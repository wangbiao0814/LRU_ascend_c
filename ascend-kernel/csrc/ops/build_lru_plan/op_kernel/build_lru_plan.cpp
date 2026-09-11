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

#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr CumSumConfig kCumSumConfig{true, false, false};

__aicore__ inline void CopyInInt16(LocalTensor<int16_t> dst,
                                   GlobalTensor<int16_t> &src,
                                   int64_t srcOffset, int64_t length)
{
    DataCopyExtParams params{};
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(length * sizeof(int16_t));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<int16_t> padParams{};
    padParams.isPad = false;
    DataCopyPad(dst, src[srcOffset], params, padParams);
}

__aicore__ inline void CopyOutInt16(GlobalTensor<int16_t> &dst,
                                    int64_t dstOffset,
                                    LocalTensor<int16_t> src,
                                    int64_t length)
{
    DataCopyExtParams params{};
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(length * sizeof(int16_t));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPad(dst[dstOffset], src, params);
}

template <HardEvent Event>
__aicore__ inline void PipeBarrier()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(eventId);
    WaitFlag<Event>(eventId);
}

__aicore__ inline LocalTensor<uint8_t> ByteSlice(LocalTensor<uint8_t> tensor,
                                                 int64_t byteOffset)
{
    return tensor[byteOffset];
}

}  // namespace

extern "C" __global__ __aicore__ void build_lru_plan_row(
    GM_ADDR lru, GM_ADDR hit, GM_ADDR hitMask, GM_ADDR newLru,
    int64_t batchSize, int64_t k, int64_t lruLength,
    int64_t hitElements, int64_t vectorLength, int64_t sortLength,
    int64_t binarySearchSteps, int64_t sortTmpBytes,
    int64_t cumSumTmpBytes, int64_t workBytes)
{
    SetAtomicNone();
    (void)hitMask;
    (void)sortTmpBytes;

    GlobalTensor<int16_t> lruGm;
    GlobalTensor<int16_t> hitGm;
    GlobalTensor<int16_t> newLruGm;
    lruGm.SetGlobalBuffer((__gm__ int16_t *)lru, batchSize * lruLength);
    hitGm.SetGlobalBuffer((__gm__ int16_t *)hit, batchSize * k);
    newLruGm.SetGlobalBuffer((__gm__ int16_t *)newLru,
                             batchSize * lruLength);

    const int64_t hitOffset = 0;
    const int64_t sortedHitOffset =
        hitElements * static_cast<int64_t>(sizeof(int16_t));
    const int64_t lruOffset =
        sortedHitOffset + sortLength * static_cast<int64_t>(sizeof(float));
    const int64_t nonHitOffset =
        lruOffset + vectorLength * static_cast<int64_t>(sizeof(int16_t));
    const int64_t scratchOffset =
        nonHitOffset + vectorLength * static_cast<int64_t>(sizeof(int16_t));

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workBytes));
    LocalTensor<uint8_t> work = workBuf.Get<uint8_t>();
    LocalTensor<int16_t> hitLocal =
        ByteSlice(work, hitOffset).ReinterpretCast<int16_t>();
    LocalTensor<float> sortedHit =
        ByteSlice(work, sortedHitOffset).ReinterpretCast<float>();
    LocalTensor<int16_t> lruLocal =
        ByteSlice(work, lruOffset).ReinterpretCast<int16_t>();
    LocalTensor<int16_t> nonHitForward =
        ByteSlice(work, nonHitOffset).ReinterpretCast<int16_t>();
    LocalTensor<uint8_t> scratch = ByteSlice(work, scratchOffset);

    const int64_t binaryMaskBytes = ((vectorLength + 255) / 256) * 32;
    const int64_t gatherRepeatTimes = vectorLength / 128;
    const int64_t gatherPatternBytes = gatherRepeatTimes * 32;

    int64_t blockNum = GetBlockNum();
    for (int64_t b = GetBlockIdx(); b < batchSize; b += blockNum) {
        // Pad before MTE2 overwrites the valid prefix. LRU padding uses -1,
        // which is present in sortedHit and therefore never survives compact.
        Duplicate(hitLocal, static_cast<int16_t>(0), hitElements);
        Duplicate(lruLocal, static_cast<int16_t>(-1), vectorLength);
        PipeBarrier<HardEvent::V_MTE2>();
        CopyInInt16(hitLocal, hitGm, b * k, k);
        CopyInInt16(lruLocal, lruGm, b * lruLength, lruLength);
        PipeBarrier<HardEvent::MTE2_V>();

        // Phase A: descending hit-ID sort. Misses and padding are -1 and act
        // as sentinels for every non-negative LRU query.
        LocalTensor<float> sortScore = scratch.ReinterpretCast<float>();
        LocalTensor<int32_t> sortIndex =
            ByteSlice(scratch, sortLength * sizeof(float))
                .ReinterpretCast<int32_t>();
        LocalTensor<float> sortRecords =
            ByteSlice(scratch, 2 * sortLength * sizeof(float))
                .ReinterpretCast<float>();
        LocalTensor<float> sortTmp =
            ByteSlice(scratch, 4 * sortLength * sizeof(float))
                .ReinterpretCast<float>();

        Duplicate(sortScore, -1.0f, sortLength);
        Cast(sortScore, hitLocal, RoundMode::CAST_NONE,
             static_cast<uint32_t>(k));
        CreateVecIndex(sortIndex, static_cast<int32_t>(0),
                       static_cast<uint32_t>(sortLength));
        Sort<float, true>(sortRecords, sortScore,
                          sortIndex.ReinterpretCast<uint32_t>(), sortTmp,
                          static_cast<int32_t>(sortLength / 32));
        Extract(sortedHit, sortIndex.ReinterpretCast<uint32_t>(), sortRecords,
                static_cast<int32_t>(sortLength / 32));

        // Phase B: vector-parallel descending lower_bound. base is the last
        // sorted position whose value is greater than the current LRU ID.
        LocalTensor<float> query = scratch.ReinterpretCast<float>();
        LocalTensor<int32_t> base =
            ByteSlice(scratch, vectorLength * 4)
                .ReinterpretCast<int32_t>();
        LocalTensor<int32_t> candidate =
            ByteSlice(scratch, vectorLength * 8)
                .ReinterpretCast<int32_t>();
        LocalTensor<int32_t> byteOffset =
            ByteSlice(scratch, vectorLength * 12)
                .ReinterpretCast<int32_t>();
        LocalTensor<float> probe =
            ByteSlice(scratch, vectorLength * 16)
                .ReinterpretCast<float>();
        LocalTensor<uint8_t> binaryMask =
            ByteSlice(scratch, vectorLength * 20);
        LocalTensor<uint8_t> gatherPattern =
            ByteSlice(scratch, vectorLength * 20 + binaryMaskBytes);

        Cast(query, lruLocal, RoundMode::CAST_NONE,
             static_cast<uint32_t>(vectorLength));
        Duplicate(base, static_cast<int32_t>(-1), vectorLength);
        int32_t step = static_cast<int32_t>(sortLength >> 1);
        for (int64_t round = 0; round < binarySearchSteps;
             ++round, step >>= 1) {
            Adds(candidate, base, step, vectorLength);
            Muls(byteOffset, candidate,
                 static_cast<int32_t>(sizeof(float)), vectorLength);
            Gather(probe, sortedHit,
                   byteOffset.ReinterpretCast<uint32_t>(), 0,
                   static_cast<uint32_t>(vectorLength));
            Compare(binaryMask, probe, query, CMPMODE::GT,
                    static_cast<uint32_t>(vectorLength));
            Select(base, binaryMask, candidate, base,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   static_cast<uint32_t>(vectorLength));
        }
        Adds(candidate, base, static_cast<int32_t>(1), vectorLength);
        Muls(byteOffset, candidate, static_cast<int32_t>(sizeof(float)),
             vectorLength);
        Gather(probe, sortedHit, byteOffset.ReinterpretCast<uint32_t>(), 0,
               static_cast<uint32_t>(vectorLength));

        // Compare writes one useful 128-bit mask into each padded 32-byte
        // pattern block, matching GatherMask's int16 custom-mask stride.
        Duplicate(gatherPattern.ReinterpretCast<uint16_t>(),
                  static_cast<uint16_t>(0), gatherPatternBytes / 2);
        for (int64_t repeat = 0; repeat < gatherRepeatTimes; ++repeat) {
            Compare(gatherPattern[repeat * 32], probe[repeat * 128],
                    query[repeat * 128], CMPMODE::NE,
                    static_cast<uint32_t>(128));
        }
        uint64_t nonHitCount = 0;
        GatherMaskParams gatherParams{
            1, static_cast<uint8_t>(gatherRepeatTimes), 8, 1};
        GatherMask(nonHitForward, lruLocal,
                   gatherPattern.ReinterpretCast<uint16_t>(), false, 0,
                   gatherParams, nonHitCount);
        PipeBarrier<HardEvent::V_S>();

        // Phase C: CumSum creates one-based miss ranks. F[nonHitCount-prefix]
        // is the reverse-suffix eviction candidate for each miss position.
        LocalTensor<float> hitFloat = scratch.ReinterpretCast<float>();
        LocalTensor<float> oneOrMinusOne =
            ByteSlice(scratch, hitElements * 4).ReinterpretCast<float>();
        LocalTensor<float> missFlag =
            ByteSlice(scratch, hitElements * 8).ReinterpretCast<float>();
        LocalTensor<float> prefixOrIndex =
            ByteSlice(scratch, hitElements * 12).ReinterpretCast<float>();
        LocalTensor<int32_t> missByteOffset =
            ByteSlice(scratch, hitElements * 16).ReinterpretCast<int32_t>();
        LocalTensor<int16_t> candidateForPos =
            ByteSlice(scratch, hitElements * 20).ReinterpretCast<int16_t>();
        LocalTensor<int16_t> filledHit =
            ByteSlice(scratch, hitElements * 22).ReinterpretCast<int16_t>();
        const int64_t fillMaskOffset = hitElements * 24;
        const int64_t fillMaskBytes = ((hitElements + 255) / 256) * 32;
        LocalTensor<uint8_t> missMask = ByteSlice(scratch, fillMaskOffset);
        LocalTensor<uint8_t> cumSumTmp =
            ByteSlice(scratch, fillMaskOffset + fillMaskBytes);
        LocalTensor<float> lastRow =
            ByteSlice(scratch, fillMaskOffset + fillMaskBytes +
                                   cumSumTmpBytes)
                .ReinterpretCast<float>();

        Cast(hitFloat, hitLocal, RoundMode::CAST_NONE,
             static_cast<uint32_t>(hitElements));
        Duplicate(oneOrMinusOne, -1.0f, hitElements);
        Compare(missMask, hitFloat, oneOrMinusOne, CMPMODE::EQ,
                static_cast<uint32_t>(hitElements));
        Duplicate(hitFloat, 0.0f, hitElements);
        Duplicate(oneOrMinusOne, 1.0f, hitElements);
        Select(missFlag, missMask, oneOrMinusOne, hitFloat,
               SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(hitElements));

        CumSumInfo cumSumInfo{1, static_cast<uint32_t>(hitElements)};
        CumSum<float, kCumSumConfig>(prefixOrIndex, lastRow, missFlag,
                                     cumSumTmp, cumSumInfo);
        PipeBarrier<HardEvent::S_V>();
        Muls(prefixOrIndex, prefixOrIndex, -1.0f, hitElements);
        Adds(prefixOrIndex, prefixOrIndex,
             static_cast<float>(nonHitCount), hitElements);
        Mins(prefixOrIndex, prefixOrIndex,
             static_cast<float>(nonHitCount - 1), hitElements);
        Maxs(prefixOrIndex, prefixOrIndex, 0.0f, hitElements);
        Cast(missByteOffset, prefixOrIndex, RoundMode::CAST_RINT,
             static_cast<uint32_t>(hitElements));
        Muls(missByteOffset, missByteOffset,
             static_cast<int32_t>(sizeof(int16_t)), hitElements);
        Gather(candidateForPos, nonHitForward,
               missByteOffset.ReinterpretCast<uint32_t>(), 0,
               static_cast<uint32_t>(hitElements));

        Duplicate(filledHit, static_cast<int16_t>(-1), hitElements);
        Compare(missMask, hitLocal, filledHit, CMPMODE::NE,
                static_cast<uint32_t>(hitElements));
        Select(filledHit, missMask, hitLocal, candidateForPos,
               SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<uint32_t>(hitElements));

        PipeBarrier<HardEvent::V_MTE3>();
        CopyOutInt16(hitGm, b * k, filledHit, k);
        CopyOutInt16(newLruGm, b * lruLength, filledHit, k);
        CopyOutInt16(newLruGm, b * lruLength + k, nonHitForward, k);
        PipeBarrier<HardEvent::MTE3_V>();
    }
}
