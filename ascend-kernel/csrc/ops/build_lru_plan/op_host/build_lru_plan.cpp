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

int64_t MaximumKernelUbBytes(int64_t lruStride, int64_t hitTileLength,
                             int64_t lruTileLength, int64_t hitMetaStride,
                             int64_t lruMetaStride, int64_t valueStride)
{
    // materialize: row bitmap + three tile buffers + compact hit/candidate
    // metadata + all candidate values + one aligned scalar block.
    int64_t maxTileLength = std::max(hitTileLength, lruTileLength);
    int64_t elements = lruStride + 3 * maxTileLength + hitMetaStride +
                       lruMetaStride + valueStride + kInt32PerDataBlock;
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
    int64_t lruTileLength = MakeParallelTileLength(
        lruLength, batchSize, coreNum);
    int64_t hitTileLength = MakeParallelTileLength(k, batchSize, coreNum);
    int64_t lruTileCount = (lruLength + lruTileLength - 1) / lruTileLength;
    int64_t hitTileCount = (k + hitTileLength - 1) / hitTileLength;

    // DataCopyPad writes only the requested logical bytes to GM. Counts can
    // therefore be packed at one int32 per tile; the row stride remains 32B
    // aligned so consumers can load the complete metadata row efficiently.
    int64_t hitMetaStride = AlignUp(hitTileCount, kInt32PerDataBlock);
    int64_t lruMetaStride = AlignUp(lruTileCount, kInt32PerDataBlock);
    int64_t valueStride = lruTileCount * lruTileLength;

    int64_t maximumUbBytes = MaximumKernelUbBytes(
        lruStride, hitTileLength, lruTileLength, hitMetaStride,
        lruMetaStride, valueStride);
    TORCH_CHECK(maximumUbBytes + kUbReserveBytes <= ubSizeBytes,
                "build_lru_plan: K=", k,
                " requires ", maximumUbBytes,
                " UB bytes for the fused path, but only ",
                ubSizeBytes - kUbReserveBytes, " bytes are available");

    TORCH_CHECK(batchSize <= std::numeric_limits<int64_t>::max() / lruStride,
                "build_lru_plan: bitmap workspace size overflow");
    TORCH_CHECK(batchSize <= std::numeric_limits<int64_t>::max() / valueStride,
                "build_lru_plan: value workspace size overflow");
    TORCH_CHECK(batchSize <= std::numeric_limits<int64_t>::max() / hitMetaStride,
                "build_lru_plan: hit metadata workspace size overflow");
    TORCH_CHECK(batchSize <= std::numeric_limits<int64_t>::max() / lruMetaStride,
                "build_lru_plan: lru metadata workspace size overflow");

    auto bitmapOptions = lruContiguous.options().dtype(at::kFloat);
    at::Tensor hitBitmap = at::zeros({batchSize, lruStride}, bitmapOptions);
    at::Tensor hitRank = at::empty({batchSize, hitStride}, lruContiguous.options());
    at::Tensor hitCounts = at::empty(
        {batchSize, hitMetaStride}, lruContiguous.options());
    at::Tensor candidateCounts = at::empty(
        {batchSize, lruMetaStride}, lruContiguous.options());
    at::Tensor candidateValues = at::empty(
        {batchSize, valueStride}, lruContiguous.options());
    // candidateValues is read while keepValues is written inside materialize;
    // these workspaces must not alias.
    at::Tensor keepCounts = at::empty(
        {batchSize, lruMetaStride}, lruContiguous.options());
    at::Tensor keepValues = at::empty(
        {batchSize, valueStride}, lruContiguous.options());

    int64_t hitTaskCount = batchSize * hitTileCount;
    int64_t lruTaskCount = batchSize * lruTileCount;
    int64_t materializeTaskCount = batchSize * (hitTileCount + lruTileCount);
    uint32_t hitBlockDim = MakeBlockDim(hitTaskCount, coreNum);
    uint32_t lruBlockDim = MakeBlockDim(lruTaskCount, coreNum);
    uint32_t materializeBlockDim = MakeBlockDim(materializeTaskCount, coreNum);

    EXEC_KERNEL_CMD(build_lru_plan_hit_local, hitBlockDim,
                    hitContiguous, hitBitmap, hitRank, hitCounts,
                    batchSize, k, lruStride, hitStride, hitMetaStride,
                    hitTileLength, hitTileCount);

    EXEC_KERNEL_CMD(build_lru_plan_candidate_local, lruBlockDim,
                    lruContiguous, hitBitmap, candidateValues, candidateCounts,
                    batchSize, lruLength, lruStride, valueStride,
                    lruMetaStride, lruTileLength, lruTileCount);

    EXEC_KERNEL_CMD(build_lru_plan_materialize, materializeBlockDim,
                    lruContiguous, hitContiguous, hitBitmap, hitRank,
                    hitCounts, candidateValues, candidateCounts,
                    keepValues, keepCounts, newLru, hitAndMiss,
                    batchSize, k, lruLength, lruStride, hitStride,
                    hitMetaStride, lruMetaStride, valueStride,
                    hitTileLength, lruTileLength,
                    hitTileCount, lruTileCount);

    EXEC_KERNEL_CMD(build_lru_plan_keep_scan_write, lruBlockDim,
                    keepValues, keepCounts, newLru,
                    batchSize, k, lruLength, valueStride, lruMetaStride,
                    lruTileLength, lruTileCount);

    return std::make_tuple(newLru, hitAndMiss);
}

}  // namespace ascend_kernel
