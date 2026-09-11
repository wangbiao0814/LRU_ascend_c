// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
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
#include "tiling/tiling_api.h"

#include "aclrtlaunch_build_lru_plan_row.h"

namespace ascend_kernel {
namespace {

constexpr int64_t kUbAlignment = 32;
constexpr int64_t kUbReserveBytes = 8 * 1024;
constexpr int64_t kMaximumK =
    (static_cast<int64_t>(std::numeric_limits<int16_t>::max()) + 1) / 2;
constexpr int64_t kMaximumSortRepeats = 255;

int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

int64_t NextPowerOfTwo(int64_t value)
{
    int64_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

int64_t IntegerLog2(int64_t powerOfTwo)
{
    int64_t result = 0;
    while (powerOfTwo > 1) {
        powerOfTwo >>= 1;
        ++result;
    }
    return result;
}

uint32_t MakeBlockDim(int64_t batchSize, int64_t coreNum)
{
    int64_t usedCoreNum = std::min(batchSize, coreNum);
    usedCoreNum = std::max<int64_t>(usedCoreNum, 1);
    return static_cast<uint32_t>(usedCoreNum);
}

struct UbPlan {
    int64_t hitElements;
    int64_t vectorLength;
    int64_t sortLength;
    int64_t binarySearchSteps;
    int64_t sortTmpBytes;
    int64_t workBytes;
};

UbPlan MakeUbPlan(int64_t k, int64_t ubSizeBytes,
                  const platform_ascendc::PlatformAscendC &platform)
{
    // int16 Compare/GatherMask process 128 elements per 256-byte repeat.
    const int64_t hitElements = AlignUp(k, 128);
    const int64_t vectorLength = AlignUp(2 * k, 128);
    // A power-of-two sort extent enables branch-free binary lifting. A
    // minimum of 32 is required by Sort.
    const int64_t sortLength =
        std::max<int64_t>(32, NextPowerOfTwo(k + 1));
    const int64_t sortRepeats = sortLength / 32;
    TORCH_CHECK(sortRepeats <= kMaximumSortRepeats,
                "build_lru_plan: vector Sort requires at most ",
                kMaximumSortRepeats, " repeats, got ", sortRepeats,
                " for K=", k);
    TORCH_CHECK(vectorLength / 128 <= kMaximumSortRepeats,
                "build_lru_plan: vector GatherMask requires at most ",
                kMaximumSortRepeats, " repeats, got ", vectorLength / 128,
                " for K=", k);

    const int64_t sortTmpBytes = AlignUp(
        static_cast<int64_t>(AscendC::GetSortTmpSize(
            platform, static_cast<uint32_t>(sortLength), sizeof(float))),
        kUbAlignment);

    const int64_t binaryMaskBytes =
        AlignUp((vectorLength + 7) / 8, kUbAlignment);
    // Custom int16 GatherMask patterns are padded to one 32-byte block per
    // 128 source elements so src1RepeatStride remains an integer block.
    const int64_t gatherPatternBytes = vectorLength / 128 * kUbAlignment;
    const int64_t hitMaskBytes =
        AlignUp((hitElements + 7) / 8, kUbAlignment);

    const int64_t persistentBytes =
        hitElements * static_cast<int64_t>(sizeof(int16_t)) +
        sortLength * static_cast<int64_t>(sizeof(float)) +
        vectorLength * static_cast<int64_t>(sizeof(int16_t)) +
        vectorLength * static_cast<int64_t>(sizeof(int16_t));
    const int64_t sortScratchBytes = 16 * sortLength + sortTmpBytes;
    const int64_t membershipScratchBytes =
        20 * vectorLength + binaryMaskBytes + gatherPatternBytes;
    // Integer Hillis-Steele scan: 36H bytes cover float mask creation,
    // zero-padded int32 scan storage, shifted values, indices/offsets,
    // gathered candidates, and the final int16 output.
    const int64_t fillScratchBytes = 36 * hitElements + hitMaskBytes;

    const int64_t minimumScratchBytes = std::max(
        {sortScratchBytes, membershipScratchBytes, fillScratchBytes});
    TORCH_CHECK(persistentBytes + minimumScratchBytes + kUbReserveBytes <=
                    ubSizeBytes,
                "build_lru_plan: K=", k, " requires at least ",
                persistentBytes + minimumScratchBytes,
                " UB bytes for the vector Sort/GatherMask/prefix-scan path, but ",
                ubSizeBytes - kUbReserveBytes,
                " bytes remain after the safety reserve");

    return {hitElements,
            vectorLength,
            sortLength,
            IntegerLog2(sortLength),
            sortTmpBytes,
            persistentBytes + minimumScratchBytes};
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
    TORCH_CHECK(lru.scalar_type() == at::kShort &&
                    hit.scalar_type() == at::kShort,
                "build_lru_plan: lru and hit must be int16 tensors");
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

    const int64_t batchSize = hit.size(0);
    const int64_t k = hit.size(1);
    TORCH_CHECK(k <= kMaximumK,
                "build_lru_plan: int16 IDs require K <= ", kMaximumK,
                ", got ", k);
    const int64_t lruLength = 2 * k;
    TORCH_CHECK(lru.size(1) == lruLength,
                "build_lru_plan: lru.shape[1] must equal 2 * hit.shape[1]");

    at::Tensor lruContiguous = lru.contiguous();
    at::Tensor newLru = at::empty_like(lruContiguous);

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum =
        static_cast<int64_t>(ascendcPlatform->GetCoreNumAiv());
    uint64_t ubSizeRaw = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                    ubSizeRaw);
    const int64_t ubSizeBytes = static_cast<int64_t>(ubSizeRaw);
    TORCH_CHECK(coreNum > 0, "build_lru_plan: platform reports no AIV cores");
    TORCH_CHECK(ubSizeBytes > kUbReserveBytes,
                "build_lru_plan: platform UB is smaller than the safety reserve");

    const UbPlan plan = MakeUbPlan(k, ubSizeBytes, *ascendcPlatform);
    const uint32_t rowBlockDim = MakeBlockDim(batchSize, coreNum);
    EXEC_KERNEL_CMD(build_lru_plan_row, rowBlockDim,
                    lruContiguous, hit, hitMask, newLru,
                    batchSize, k, lruLength, plan.hitElements,
                    plan.vectorLength, plan.sortLength,
                    plan.binarySearchSteps, plan.sortTmpBytes,
                    plan.workBytes);
    return newLru;
}

}  // namespace ascend_kernel
