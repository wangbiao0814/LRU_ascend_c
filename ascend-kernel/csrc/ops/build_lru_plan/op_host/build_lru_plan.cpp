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

#include "aclrtlaunch_build_lru_plan_hit_fused.h"
#include "aclrtlaunch_build_lru_plan_candidate_fill_fused.h"
#include "aclrtlaunch_build_lru_plan_keep_write_fused.h"

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
    // Produce at least coreNum tasks whenever the logical shape has enough
    // 32B data blocks. B=1,K=2048 yields TH=48 and TL=96 on a 40-AIV device.
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

int64_t MaximumKernelUbBytes(int64_t lruStride, int64_t hitStride,
                             int64_t hitTileLength,
                             int64_t lruTileLength, int64_t lruTileCount,
                             int64_t tileCountStride, int64_t valueStride,
                             int64_t missCountStride, int64_t syncStride)
{
    int64_t syncBytes = syncStride * static_cast<int64_t>(sizeof(int32_t));

    int64_t hitLocalElements = 2 * hitTileLength + kInt32PerDataBlock;
    int64_t hitLeaderElements =
        hitStride + tileCountStride + missCountStride;
    int64_t hitWorkElements = std::max(hitLocalElements, hitLeaderElements);
    int64_t hitKernelBytes =
        hitWorkElements * static_cast<int64_t>(sizeof(int32_t)) +
        lruStride * static_cast<int64_t>(sizeof(int8_t)) + syncBytes;

    int64_t candidateElements = 2 * lruTileLength + kInt32PerDataBlock;
    int64_t candidateLeaderElements =
        valueStride + tileCountStride + hitStride + missCountStride;
    int64_t fillElements =
        3 * hitTileLength + hitStride + 2 * kInt32PerDataBlock;
    int64_t candidateWorkElements = std::max(
        candidateElements, std::max(candidateLeaderElements, fillElements));
    int64_t candidateKernelBytes =
        candidateWorkElements * static_cast<int64_t>(sizeof(int32_t)) +
        lruStride * static_cast<int64_t>(sizeof(int8_t)) + syncBytes;

    int64_t keepLocalElements = 2 * lruTileLength + kInt32PerDataBlock;
    int64_t keepScanElements =
        lruTileCount * kInt32PerDataBlock + kInt32PerDataBlock;
    int64_t keepWriteElements = lruTileLength + kInt32PerDataBlock;
    int64_t keepWorkElements = std::max(
        keepLocalElements, std::max(keepScanElements, keepWriteElements));
    int64_t keepKernelBytes =
        keepWorkElements * static_cast<int64_t>(sizeof(int32_t)) +
        lruStride * static_cast<int64_t>(sizeof(int8_t)) + syncBytes;

    return std::max(hitKernelBytes,
                    std::max(candidateKernelBytes, keepKernelBytes));
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

    // Each count/offset owns a 32B slot. Lane 0 is offset/count and lane 1
    // preserves the count after the parallel scan stage.
    int64_t tileCountStride =
        std::max(lruTileCount, hitTileCount) * kInt32PerDataBlock;
    int64_t valueStride = lruTileCount * lruTileLength;
    int64_t missCountStride = kInt32PerDataBlock;
    // Soft SyncAll requires one 32B slot per physical AIV. Fused kernels with
    // two barriers receive two independently zero-initialized GM regions.
    int64_t syncStride = coreNum * kInt32PerDataBlock;

    int64_t maximumUbBytes = MaximumKernelUbBytes(
        lruStride, hitStride, hitTileLength,
        lruTileLength, lruTileCount, tileCountStride,
        valueStride, missCountStride, syncStride);
    TORCH_CHECK(maximumUbBytes + kUbReserveBytes <= ubSizeBytes,
                "build_lru_plan: K=", k,
                " requires ", maximumUbBytes,
                " UB bytes for the fully parallel path, but only ",
                ubSizeBytes - kUbReserveBytes, " bytes are available");

    TORCH_CHECK(batchSize <= std::numeric_limits<int64_t>::max() / lruStride,
                "build_lru_plan: bitmap workspace size overflow");
    TORCH_CHECK(batchSize <= std::numeric_limits<int64_t>::max() / valueStride,
                "build_lru_plan: value workspace size overflow");
    TORCH_CHECK(batchSize <= std::numeric_limits<int64_t>::max() / tileCountStride,
                "build_lru_plan: count workspace size overflow");

    auto bitmapOptions = lruContiguous.options().dtype(at::kChar);
    at::Tensor hitBitmap = at::zeros({batchSize, lruStride}, bitmapOptions);
    at::Tensor selectedBitmap = at::zeros({batchSize, lruStride}, bitmapOptions);
    at::Tensor hitRank = at::empty({batchSize, hitStride}, lruContiguous.options());
    at::Tensor hitTileInfo = at::empty(
        {batchSize, tileCountStride}, lruContiguous.options());
    at::Tensor tileCounts = at::empty(
        {batchSize, tileCountStride}, lruContiguous.options());
    at::Tensor tileValues = at::empty(
        {batchSize, valueStride}, lruContiguous.options());
    at::Tensor candidatePacked = at::empty(
        {batchSize, hitStride}, lruContiguous.options());
    at::Tensor missCount = at::empty(
        {batchSize, missCountStride}, lruContiguous.options());
    at::Tensor hitSync = at::zeros(
        {syncStride}, lruContiguous.options());
    at::Tensor candidateSync = at::zeros(
        {2, syncStride}, lruContiguous.options());
    at::Tensor keepSync = at::zeros(
        {2, syncStride}, lruContiguous.options());

    int64_t hitTaskCount = batchSize * hitTileCount;
    int64_t lruTaskCount = batchSize * lruTileCount;
    uint32_t hitBlockDim = MakeBlockDim(hitTaskCount, coreNum);
    uint32_t lruBlockDim = MakeBlockDim(lruTaskCount, coreNum);
    int64_t candidateTaskCount = std::max(hitTaskCount, lruTaskCount);
    uint32_t candidateBlockDim = MakeBlockDim(candidateTaskCount, coreNum);

    EXEC_KERNEL_CMD(build_lru_plan_hit_fused, hitBlockDim,
                    hitContiguous, hitBitmap, hitRank, hitTileInfo,
                    missCount, hitSync,
                    batchSize, k, lruStride, hitStride, tileCountStride,
                    missCountStride, hitTileLength, hitTileCount, syncStride);

    EXEC_KERNEL_CMD(build_lru_plan_candidate_fill_fused, candidateBlockDim,
                    lruContiguous, hitContiguous, hitBitmap, hitRank,
                    hitTileInfo, missCount, tileValues, tileCounts,
                    candidatePacked, selectedBitmap, hitAndMiss, newLru,
                    candidateSync,
                    batchSize, k, lruLength, lruStride, hitStride,
                    tileCountStride, valueStride, missCountStride,
                    lruTileLength, hitTileLength,
                    lruTileCount, hitTileCount, syncStride);

    EXEC_KERNEL_CMD(build_lru_plan_keep_write_fused, lruBlockDim,
                    lruContiguous, selectedBitmap, tileValues, tileCounts,
                    newLru, keepSync,
                    batchSize, k, lruLength, lruStride, valueStride,
                    tileCountStride, lruTileLength, lruTileCount, syncStride);

    return std::make_tuple(newLru, hitAndMiss);
}

}  // namespace ascend_kernel
