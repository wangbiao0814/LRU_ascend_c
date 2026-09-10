// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstdint>
#include <limits>

#include <ATen/MemoryOverlap.h>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_build_lru_plan_row.h"
#include "aclrtlaunch_build_lru_plan_hit_local.h"
#include "aclrtlaunch_build_lru_plan_candidate_local.h"
#include "aclrtlaunch_build_lru_plan_materialize.h"
#include "aclrtlaunch_build_lru_plan_keep_scan_write.h"

namespace ascend_kernel {
namespace {

constexpr int64_t kInt32PerDataBlock = 8;
constexpr int64_t kInt32PerCacheLine = 128;
constexpr int64_t kMaximumTileLength = 256;
constexpr int64_t kUbReserveBytes = 8 * 1024;

int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

int64_t AlignDown(int64_t value, int64_t alignment)
{
    return value / alignment * alignment;
}

int64_t MakeParallelTileLength(int64_t logicalLength, int64_t batchSize,
                               int64_t coreNum)
{
    int64_t desiredTilesPerRow = (coreNum + batchSize - 1) / batchSize;
    int64_t tileLength = AlignDown(
        logicalLength / desiredTilesPerRow, kInt32PerDataBlock);
    tileLength = std::max(tileLength, kInt32PerDataBlock);
    tileLength = std::min(tileLength, kMaximumTileLength);
    return tileLength;
}

uint32_t MakeBlockDim(int64_t taskCount, int64_t coreNum)
{
    int64_t usedCoreNum = std::min(taskCount, coreNum);
    usedCoreNum = std::max<int64_t>(usedCoreNum, 1);
    return static_cast<uint32_t>(usedCoreNum);
}

int64_t MaximumKernelUbBytes(int64_t k, int64_t lruStride,
                             int64_t hitTileLength,
                             int64_t lruTileLength, int64_t hitMetaStride,
                             int64_t lruMetaStride, int64_t valueStride)
{
    int64_t scalarBlockBytes = kInt32PerDataBlock *
                               static_cast<int64_t>(sizeof(int32_t));
    int64_t metadataBytes = (hitMetaStride + lruMetaStride) *
                            static_cast<int64_t>(sizeof(int32_t));
    int64_t maskTileBytes = AlignUp(hitTileLength, 32);

    int64_t hitLocalStride = AlignUp(k, kInt32PerDataBlock);
    int64_t hitLocalBytes = (lruStride + hitLocalStride + hitMetaStride) *
                            static_cast<int64_t>(sizeof(int32_t));
    int64_t candidateLocalBytes = lruStride * static_cast<int64_t>(sizeof(float)) +
                                  2 * lruTileLength *
                                      static_cast<int64_t>(sizeof(int32_t)) +
                                  scalarBlockBytes;
    int64_t materializeHitBytes = valueStride *
                                      static_cast<int64_t>(sizeof(int32_t)) +
                                  metadataBytes +
                                  2 * hitTileLength *
                                      static_cast<int64_t>(sizeof(int32_t)) +
                                  maskTileBytes + scalarBlockBytes;
    int64_t materializeLruBytes = lruStride *
                                      static_cast<int64_t>(sizeof(float)) +
                                  metadataBytes +
                                  2 * lruTileLength *
                                      static_cast<int64_t>(sizeof(int32_t)) +
                                  scalarBlockBytes;
    int64_t keepWriteBytes = lruMetaStride *
                                 static_cast<int64_t>(sizeof(int32_t)) +
                             lruTileLength *
                                 static_cast<int64_t>(sizeof(int32_t));

    return std::max(
        std::max(hitLocalBytes, candidateLocalBytes),
        std::max(std::max(materializeHitBytes, materializeLruBytes),
                 keepWriteBytes));
}

int64_t FusedKernelUbBytes(int64_t bitmapElements, int64_t maskBytes,
                           int64_t hitElements, int64_t lruElements)
{
    return bitmapElements * static_cast<int64_t>(sizeof(int16_t)) + maskBytes +
           2 * hitElements * static_cast<int64_t>(sizeof(int32_t)) +
           lruElements * static_cast<int64_t>(sizeof(int32_t));
}

}  // namespace

at::Tensor build_lru_plan(const at::Tensor &lru, at::Tensor &hit,
                          const at::Tensor &hitMask)
{
    TORCH_CHECK(lru.device().type() == at::DeviceType::PrivateUse1,
                "build_lru_plan: lru must be on an NPU device");
    TORCH_CHECK(hit.device().type() == at::DeviceType::PrivateUse1,
                "build_lru_plan: hit must be on an NPU device");
    TORCH_CHECK(hitMask.device().type() == at::DeviceType::PrivateUse1,
                "build_lru_plan: hit_mask must be on an NPU device");
    TORCH_CHECK(lru.device() == hit.device() &&
                    lru.device() == hitMask.device(),
                "build_lru_plan: all inputs must be on the same NPU device");
    TORCH_CHECK(lru.scalar_type() == at::kInt && hit.scalar_type() == at::kInt,
                "build_lru_plan: lru and hit must be int32 tensors");
    TORCH_CHECK(hitMask.scalar_type() == at::kBool,
                "build_lru_plan: hit_mask must be a bool tensor");
    TORCH_CHECK(lru.dim() == 2 && hit.dim() == 2 && hitMask.dim() == 2,
                "build_lru_plan: all inputs must be rank-2 tensors");
    TORCH_CHECK(lru.size(0) == hit.size(0),
                "build_lru_plan: batch dimensions must match");
    TORCH_CHECK(hitMask.sizes() == hit.sizes(),
                "build_lru_plan: hit_mask must have the same shape as hit");
    TORCH_CHECK(hit.size(0) > 0 && hit.size(1) > 0,
                "build_lru_plan: B and K must both be positive");
    TORCH_CHECK(hit.is_contiguous(),
                "build_lru_plan: hit must be contiguous for in-place update");
    TORCH_CHECK(at::get_overlap_status(hit, lru) == at::MemOverlapStatus::No,
                "build_lru_plan: hit must not overlap lru storage");
    TORCH_CHECK(at::get_overlap_status(hit, hitMask) == at::MemOverlapStatus::No,
                "build_lru_plan: hit must not overlap hit_mask storage");

    int64_t batchSize = hit.size(0);
    int64_t k = hit.size(1);
    TORCH_CHECK(k <= std::numeric_limits<int32_t>::max() / 2,
                "build_lru_plan: K is too large for int32 IDs");
    int64_t lruLength = 2 * k;
    TORCH_CHECK(lru.size(1) == lruLength,
                "build_lru_plan: lru.shape[1] must equal 2 * hit.shape[1]");

    at::Tensor lruContiguous = lru.contiguous();
    at::Tensor hitMaskContiguous = hitMask.contiguous();
    at::Tensor newLru = at::empty_like(lruContiguous);

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = static_cast<int64_t>(ascendcPlatform->GetCoreNumAiv());
    uint64_t ubSizeRaw = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeRaw);
    int64_t ubSizeBytes = static_cast<int64_t>(ubSizeRaw);
    TORCH_CHECK(coreNum > 0, "build_lru_plan: platform reports no AIV cores");
    TORCH_CHECK(ubSizeBytes > kUbReserveBytes,
                "build_lru_plan: platform UB is smaller than the safety reserve");

    // Fast path: one AIV task owns a complete row.  K/2K inputs stay resident
    // in UB, so all four logical stages can be fused without GM workspace.
    int64_t bitmapElements = AlignUp(lruLength, 16);
    int64_t maskBytes = AlignUp(k, 32);
    int64_t hitElements = AlignUp(k, kInt32PerDataBlock);
    int64_t lruElements = AlignUp(lruLength, kInt32PerDataBlock);
    int64_t fusedUbBytes = FusedKernelUbBytes(
        bitmapElements, maskBytes, hitElements, lruElements);
    if (fusedUbBytes + kUbReserveBytes <= ubSizeBytes) {
        uint32_t rowBlockDim = MakeBlockDim(batchSize, coreNum);
        EXEC_KERNEL_CMD(build_lru_plan_row, rowBlockDim,
                        lruContiguous, hit, hitMaskContiguous, newLru,
                        batchSize, k, lruLength, bitmapElements, maskBytes,
                        hitElements, lruElements);
        return newLru;
    }
}

}  // namespace ascend_kernel
