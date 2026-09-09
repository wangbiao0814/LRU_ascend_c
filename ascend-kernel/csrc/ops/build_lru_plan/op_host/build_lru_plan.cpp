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
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_build_lru_plan_hit_local.h"
#include "aclrtlaunch_build_lru_plan_hit_scan.h"
#include "aclrtlaunch_build_lru_plan_candidate_local.h"
#include "aclrtlaunch_build_lru_plan_candidate_scan.h"
#include "aclrtlaunch_build_lru_plan_fill.h"
#include "aclrtlaunch_build_lru_plan_keep_local.h"
#include "aclrtlaunch_build_lru_plan_keep_scan.h"
#include "aclrtlaunch_build_lru_plan_write.h"

namespace ascend_kernel {
namespace {

constexpr int64_t kInt32PerDataBlock = 8;
constexpr int64_t kInt32PerCacheLine = 128;
constexpr int64_t kPreferredTileLength = 256;
constexpr int64_t kMinimumTileLength = 128;
constexpr int64_t kUbReserveBytes = 8 * 1024;

int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

uint32_t MakeBlockDim(int64_t taskCount, int64_t coreNum)
{
    int64_t usedCoreNum = std::min(taskCount, coreNum);
    usedCoreNum = std::max<int64_t>(usedCoreNum, 1);
    return static_cast<uint32_t>(usedCoreNum);
}

int64_t FillKernelUbBytes(int64_t lruStride, int64_t hitStride,
                          int64_t lruTileLength, int64_t hitTileLength)
{
    // bitmap + miss values + two LRU tiles + three hit tiles + one 32B scalar block.
    int64_t elements = lruStride + hitStride + 2 * lruTileLength +
                       3 * hitTileLength + kInt32PerDataBlock;
    return elements * static_cast<int64_t>(sizeof(int32_t));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> build_lru_plan(
    const at::Tensor &lru, const at::Tensor &hit)
{
    TORCH_CHECK(lru.device().type() == at::DeviceType::PrivateUse1,
                "build_lru_plan: lru must be on an NPU device");
    TORCH_CHECK(hit.device().type() == at::DeviceType::PrivateUse1,
                "build_lru_plan: hit must be on an NPU device");
    TORCH_CHECK(lru.device() == hit.device(),
                "build_lru_plan: lru and hit must be on the same NPU device");
    TORCH_CHECK(lru.scalar_type() == at::kInt && hit.scalar_type() == at::kInt,
                "build_lru_plan: only int32 inputs are supported");
    TORCH_CHECK(lru.dim() == 2 && hit.dim() == 2,
                "build_lru_plan: lru and hit must both be rank-2 tensors");
    TORCH_CHECK(lru.size(0) == hit.size(0),
                "build_lru_plan: batch dimensions must match");
    TORCH_CHECK(hit.size(0) > 0 && hit.size(1) > 0,
                "build_lru_plan: B and K must both be positive");

    int64_t batchSize = hit.size(0);
    int64_t k = hit.size(1);
    TORCH_CHECK(k <= std::numeric_limits<int32_t>::max() / 2,
                "build_lru_plan: K is too large for int32 IDs");
    int64_t lruLength = 2 * k;
    TORCH_CHECK(lru.size(1) == lruLength,
                "build_lru_plan: lru.shape[1] must equal 2 * hit.shape[1]");

    at::Tensor lruContiguous = lru.contiguous();
    at::Tensor hitContiguous = hit.contiguous();
    at::Tensor newLru = at::empty_like(lruContiguous);
    at::Tensor hitAndMiss = at::empty_like(hitContiguous);

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = static_cast<int64_t>(ascendcPlatform->GetCoreNumAiv());
    uint64_t ubSizeRaw = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeRaw);
    int64_t ubSizeBytes = static_cast<int64_t>(ubSizeRaw);
    TORCH_CHECK(coreNum > 0, "build_lru_plan: platform reports no AIV cores");
    TORCH_CHECK(ubSizeBytes > kUbReserveBytes,
                "build_lru_plan: platform UB is smaller than the safety reserve");

    int64_t lruStride = AlignUp(lruLength, kInt32PerCacheLine);
    int64_t hitStride = AlignUp(k, kInt32PerCacheLine);
    int64_t lruTileLength = kPreferredTileLength;
    int64_t hitTileLength = kPreferredTileLength;

    int64_t fillUbBytes = FillKernelUbBytes(
        lruStride, hitStride, lruTileLength, hitTileLength);
    if (fillUbBytes + kUbReserveBytes > ubSizeBytes) {
        lruTileLength = kMinimumTileLength;
        hitTileLength = kMinimumTileLength;
        fillUbBytes = FillKernelUbBytes(
            lruStride, hitStride, lruTileLength, hitTileLength);
    }
    TORCH_CHECK(fillUbBytes + kUbReserveBytes <= ubSizeBytes,
                "build_lru_plan: K=", k,
                " requires ", fillUbBytes,
                " UB bytes for the row bitmap path, but only ",
                ubSizeBytes - kUbReserveBytes, " bytes are available");

    int64_t lruTileCount = (lruLength + lruTileLength - 1) / lruTileLength;
    int64_t hitTileCount = (k + hitTileLength - 1) / hitTileLength;
    // Give each tile count an exclusive 32B slot so parallel MTE3 writes never
    // share a data block. The first int32 in each slot stores the count/offset.
    int64_t tileCountStride =
        std::max(lruTileCount, hitTileCount) * kInt32PerDataBlock;
    int64_t missCountStride = kInt32PerDataBlock;

    TORCH_CHECK(batchSize <= std::numeric_limits<int64_t>::max() / lruStride,
                "build_lru_plan: workspace size overflow");
    TORCH_CHECK(batchSize <= std::numeric_limits<int64_t>::max() / tileCountStride,
                "build_lru_plan: tile workspace size overflow");

    at::Tensor bitmap = at::empty({batchSize, lruStride}, lruContiguous.options());
    at::Tensor hitRank = at::empty({batchSize, hitStride}, lruContiguous.options());
    at::Tensor lruRank = at::empty({batchSize, lruStride}, lruContiguous.options());
    at::Tensor tileCounts = at::empty(
        {batchSize, tileCountStride}, lruContiguous.options());
    at::Tensor missCount = at::empty(
        {batchSize, missCountStride}, lruContiguous.options());

    int64_t rowTaskCount = batchSize;
    int64_t lruTileTaskCount = batchSize * lruTileCount;
    uint32_t rowBlockDim = MakeBlockDim(rowTaskCount, coreNum);
    uint32_t lruTileBlockDim = MakeBlockDim(lruTileTaskCount, coreNum);

    // All tensors and scalar launcher arguments are named lvalues by construction.
    EXEC_KERNEL_CMD(build_lru_plan_hit_local, rowBlockDim,
                    hitContiguous, bitmap, hitRank, tileCounts,
                    batchSize, k, lruStride, hitStride, tileCountStride,
                    hitTileLength, hitTileCount);

    EXEC_KERNEL_CMD(build_lru_plan_hit_scan, rowBlockDim,
                    hitRank, tileCounts, missCount,
                    batchSize, k, hitStride, tileCountStride,
                    missCountStride, hitTileLength, hitTileCount);

    EXEC_KERNEL_CMD(build_lru_plan_candidate_local, lruTileBlockDim,
                    lruContiguous, bitmap, lruRank, tileCounts,
                    batchSize, lruLength, lruStride, tileCountStride,
                    lruTileLength, lruTileCount);

    EXEC_KERNEL_CMD(build_lru_plan_candidate_scan, rowBlockDim,
                    lruRank, tileCounts,
                    batchSize, lruLength, lruStride, tileCountStride,
                    lruTileLength, lruTileCount);

    EXEC_KERNEL_CMD(build_lru_plan_fill, rowBlockDim,
                    lruContiguous, hitContiguous, bitmap, lruRank,
                    hitRank, missCount, hitAndMiss,
                    batchSize, k, lruLength, lruStride, hitStride,
                    missCountStride, lruTileLength, hitTileLength,
                    lruTileCount, hitTileCount);

    EXEC_KERNEL_CMD(build_lru_plan_keep_local, lruTileBlockDim,
                    lruContiguous, bitmap, lruRank, tileCounts,
                    batchSize, lruLength, lruStride, tileCountStride,
                    lruTileLength, lruTileCount);

    EXEC_KERNEL_CMD(build_lru_plan_keep_scan, rowBlockDim,
                    lruRank, tileCounts,
                    batchSize, lruLength, lruStride, tileCountStride,
                    lruTileLength, lruTileCount);

    EXEC_KERNEL_CMD(build_lru_plan_write, rowBlockDim,
                    lruContiguous, hitAndMiss, lruRank, tileCounts, newLru,
                    batchSize, k, lruLength, lruStride, tileCountStride,
                    lruTileLength, hitTileLength, lruTileCount, hitTileCount);

    return std::make_tuple(newLru, hitAndMiss);
}

}  // namespace ascend_kernel
