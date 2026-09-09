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

#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr int64_t kDataBlockElements = 8;

__aicore__ inline int64_t MinInt64(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline void LoadInt32(LocalTensor<int32_t> dst,
                                  GlobalTensor<int32_t> &src,
                                  int64_t srcOffset, int64_t length)
{
    DataCopyExtParams params{};
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(length * sizeof(int32_t));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<int32_t> padParams{};
    padParams.isPad = false;
    DataCopyPad(dst, src[srcOffset], params, padParams);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void LoadFloat(LocalTensor<float> dst,
                                 GlobalTensor<float> &src,
                                 int64_t srcOffset, int64_t length)
{
    DataCopyExtParams params{};
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(length * sizeof(float));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<float> padParams{};
    padParams.isPad = false;
    DataCopyPad(dst, src[srcOffset], params, padParams);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void StoreInt32(GlobalTensor<int32_t> &dst,
                                   int64_t dstOffset,
                                   LocalTensor<int32_t> src,
                                   int64_t length)
{
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams params{};
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(length * sizeof(int32_t));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPad(dst[dstOffset], src, params);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void StoreFloatAtomic(GlobalTensor<float> &dst,
                                         int64_t dstOffset,
                                         LocalTensor<float> src,
                                         int64_t length)
{
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams params{};
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(length * sizeof(float));
    params.srcStride = 0;
    params.dstStride = 0;
    SetAtomicAdd<float>();
    DataCopyPad(dst[dstOffset], src, params);
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
}

}  // namespace

extern "C" __global__ __aicore__ void build_lru_plan_hit_local(
    GM_ADDR hit, GM_ADDR hitBitmap, GM_ADDR hitRank, GM_ADDR tileCounts,
    int64_t batchSize, int64_t k, int64_t lruStride, int64_t hitStride,
    int64_t tileCountStride, int64_t hitTileLength, int64_t hitTileCount)
{
    SetAtomicNone();
    GlobalTensor<int32_t> hitGm;
    GlobalTensor<float> bitmapGm;
    GlobalTensor<int32_t> hitRankGm;
    GlobalTensor<int32_t> tileCountsGm;
    hitGm.SetGlobalBuffer((__gm__ int32_t *)hit, batchSize * k);
    bitmapGm.SetGlobalBuffer((__gm__ float *)hitBitmap,
                             batchSize * lruStride);
    hitRankGm.SetGlobalBuffer((__gm__ int32_t *)hitRank,
                              batchSize * hitStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECOUT> bitmapBuf;
    pipe.InitBuffer(workBuf,
                    static_cast<uint32_t>((2 * hitTileLength +
                                           kDataBlockElements) * sizeof(int32_t)));
    pipe.InitBuffer(bitmapBuf,
                    static_cast<uint32_t>(lruStride * sizeof(float)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> hitLocal = work;
    LocalTensor<int32_t> rankLocal = work[hitTileLength];
    LocalTensor<int32_t> countLocal = work[2 * hitTileLength];
    LocalTensor<float> bitmapLocal = bitmapBuf.Get<float>();

    int64_t taskCount = batchSize * hitTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / hitTileCount;
        int64_t ht = task - b * hitTileCount;
        int64_t start = ht * hitTileLength;
        int64_t validLen = MinInt64(hitTileLength, k - start);

        Duplicate(bitmapLocal, 0.0f, lruStride);
        Duplicate(countLocal, static_cast<int32_t>(0), kDataBlockElements);
        PipeBarrier<PIPE_ALL>();
        LoadInt32(hitLocal, hitGm, b * k + start, validLen);

        int32_t prefix = 0;
        for (int64_t p = 0; p < validLen; ++p) {
            int32_t id = hitLocal.GetValue(p);
            if (id == -1) {
                rankLocal.SetValue(p, prefix);
                ++prefix;
            } else {
                rankLocal.SetValue(p, static_cast<int32_t>(-1));
                bitmapLocal.SetValue(id, 1.0f);
            }
        }
        countLocal.SetValue(0, prefix);
        countLocal.SetValue(1, prefix);

        StoreInt32(hitRankGm, b * hitStride + start, rankLocal, validLen);
        StoreInt32(tileCountsGm,
                   b * tileCountStride + ht * kDataBlockElements,
                   countLocal, kDataBlockElements);
        StoreFloatAtomic(bitmapGm, b * lruStride, bitmapLocal, lruStride);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_hit_scan(
    GM_ADDR hitRank, GM_ADDR tileCounts, GM_ADDR missCount,
    int64_t batchSize, int64_t k, int64_t hitStride,
    int64_t tileCountStride, int64_t missCountStride,
    int64_t hitTileLength, int64_t hitTileCount)
{
    SetAtomicNone();
    GlobalTensor<int32_t> hitRankGm;
    GlobalTensor<int32_t> tileCountsGm;
    GlobalTensor<int32_t> missCountGm;
    hitRankGm.SetGlobalBuffer((__gm__ int32_t *)hitRank,
                              batchSize * hitStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);
    missCountGm.SetGlobalBuffer((__gm__ int32_t *)missCount,
                                batchSize * missCountStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = hitTileLength + tileCountStride + missCountStride;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> rankLocal = work;
    LocalTensor<int32_t> countLocal = work[hitTileLength];
    LocalTensor<int32_t> scalarLocal = work[hitTileLength + tileCountStride];

    int64_t taskCount = batchSize * hitTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / hitTileCount;
        int64_t ht = task - b * hitTileCount;
        int64_t start = ht * hitTileLength;
        int64_t validLen = MinInt64(hitTileLength, k - start);
        LoadInt32(countLocal, tileCountsGm,
                  b * tileCountStride, tileCountStride);
        LoadInt32(rankLocal, hitRankGm,
                  b * hitStride + start, validLen);

        int32_t offset = 0;
        for (int64_t tile = 0; tile < ht; ++tile) {
            offset += countLocal.GetValue(tile * kDataBlockElements);
        }
        int32_t ownCount = countLocal.GetValue(ht * kDataBlockElements);
        for (int64_t p = 0; p < validLen; ++p) {
            int32_t rank = rankLocal.GetValue(p);
            if (rank >= 0) {
                rankLocal.SetValue(p, rank + offset);
            }
        }
        StoreInt32(hitRankGm, b * hitStride + start, rankLocal, validLen);

        if (ht + 1 == hitTileCount) {
            Duplicate(scalarLocal, static_cast<int32_t>(0), missCountStride);
            PipeBarrier<PIPE_ALL>();
            scalarLocal.SetValue(0, offset + ownCount);
            StoreInt32(missCountGm, b * missCountStride,
                       scalarLocal, missCountStride);
        }
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_candidate_local(
    GM_ADDR lru, GM_ADDR hitBitmap, GM_ADDR tileValues, GM_ADDR tileCounts,
    int64_t batchSize, int64_t lruLength, int64_t lruStride,
    int64_t valueStride, int64_t tileCountStride,
    int64_t lruTileLength, int64_t lruTileCount)
{
    SetAtomicNone();
    GlobalTensor<int32_t> lruGm;
    GlobalTensor<float> bitmapGm;
    GlobalTensor<int32_t> tileValuesGm;
    GlobalTensor<int32_t> tileCountsGm;
    lruGm.SetGlobalBuffer((__gm__ int32_t *)lru, batchSize * lruLength);
    bitmapGm.SetGlobalBuffer((__gm__ float *)hitBitmap,
                             batchSize * lruStride);
    tileValuesGm.SetGlobalBuffer((__gm__ int32_t *)tileValues,
                                 batchSize * valueStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = lruStride + 2 * lruTileLength + kDataBlockElements;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<float> bitmapLocal = work.ReinterpretCast<float>();
    LocalTensor<int32_t> lruLocal = work[lruStride];
    LocalTensor<int32_t> valuesLocal = work[lruStride + lruTileLength];
    LocalTensor<int32_t> countLocal =
        work[lruStride + 2 * lruTileLength];

    int64_t taskCount = batchSize * lruTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / lruTileCount;
        int64_t lt = task - b * lruTileCount;
        int64_t start = lt * lruTileLength;
        int64_t validLen = MinInt64(lruTileLength, lruLength - start);
        LoadFloat(bitmapLocal, bitmapGm, b * lruStride, lruLength);
        LoadInt32(lruLocal, lruGm, b * lruLength + start, validLen);
        Duplicate(valuesLocal, static_cast<int32_t>(0), lruTileLength);
        Duplicate(countLocal, static_cast<int32_t>(0), kDataBlockElements);
        PipeBarrier<PIPE_ALL>();

        int32_t suffix = 0;
        for (int64_t p = validLen; p > 0; --p) {
            int64_t index = p - 1;
            int32_t id = lruLocal.GetValue(index);
            if (bitmapLocal.GetValue(id) == 0.0f) {
                valuesLocal.SetValue(suffix, id);
                ++suffix;
            }
        }
        countLocal.SetValue(0, suffix);
        countLocal.SetValue(1, suffix);
        StoreInt32(tileValuesGm,
                   b * valueStride + lt * lruTileLength,
                   valuesLocal, lruTileLength);
        StoreInt32(tileCountsGm,
                   b * tileCountStride + lt * kDataBlockElements,
                   countLocal, kDataBlockElements);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_candidate_scan(
    GM_ADDR tileCounts, int64_t batchSize,
    int64_t tileCountStride, int64_t lruTileCount)
{
    SetAtomicNone();
    GlobalTensor<int32_t> tileCountsGm;
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = tileCountStride + kDataBlockElements;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> countLocal = work;
    LocalTensor<int32_t> scalarLocal = work[tileCountStride];

    int64_t taskCount = batchSize * lruTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / lruTileCount;
        int64_t lt = task - b * lruTileCount;
        LoadInt32(countLocal, tileCountsGm,
                  b * tileCountStride, tileCountStride);

        int32_t offset = 0;
        for (int64_t tile = lt + 1; tile < lruTileCount; ++tile) {
            offset += countLocal.GetValue(tile * kDataBlockElements + 1);
        }
        int32_t ownCount =
            countLocal.GetValue(lt * kDataBlockElements + 1);
        Duplicate(scalarLocal, static_cast<int32_t>(0), kDataBlockElements);
        PipeBarrier<PIPE_ALL>();
        scalarLocal.SetValue(0, offset);
        scalarLocal.SetValue(1, ownCount);
        StoreInt32(tileCountsGm,
                   b * tileCountStride + lt * kDataBlockElements,
                   scalarLocal, kDataBlockElements);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_fill(
    GM_ADDR hit, GM_ADDR hitRank, GM_ADDR tileCounts, GM_ADDR tileValues,
    GM_ADDR missCount, GM_ADDR selectedBitmap, GM_ADDR hitAndMiss,
    int64_t batchSize, int64_t k, int64_t lruStride, int64_t hitStride,
    int64_t tileCountStride, int64_t valueStride, int64_t missCountStride,
    int64_t lruTileLength, int64_t hitTileLength,
    int64_t lruTileCount, int64_t hitTileCount)
{
    SetAtomicNone();
    GlobalTensor<int32_t> hitGm;
    GlobalTensor<int32_t> hitRankGm;
    GlobalTensor<int32_t> tileCountsGm;
    GlobalTensor<int32_t> tileValuesGm;
    GlobalTensor<int32_t> missCountGm;
    GlobalTensor<float> selectedBitmapGm;
    GlobalTensor<int32_t> hitAndMissGm;
    hitGm.SetGlobalBuffer((__gm__ int32_t *)hit, batchSize * k);
    hitRankGm.SetGlobalBuffer((__gm__ int32_t *)hitRank,
                              batchSize * hitStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);
    tileValuesGm.SetGlobalBuffer((__gm__ int32_t *)tileValues,
                                 batchSize * valueStride);
    missCountGm.SetGlobalBuffer((__gm__ int32_t *)missCount,
                                batchSize * missCountStride);
    selectedBitmapGm.SetGlobalBuffer((__gm__ float *)selectedBitmap,
                                     batchSize * lruStride);
    hitAndMissGm.SetGlobalBuffer((__gm__ int32_t *)hitAndMiss,
                                 batchSize * k);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECOUT> bitmapBuf;
    int64_t workElements = 3 * hitTileLength + valueStride +
                           tileCountStride + missCountStride;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    pipe.InitBuffer(bitmapBuf,
                    static_cast<uint32_t>(lruStride * sizeof(float)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> hitLocal = work;
    LocalTensor<int32_t> rankLocal = work[hitTileLength];
    LocalTensor<int32_t> outputLocal = work[2 * hitTileLength];
    int64_t valueBase = 3 * hitTileLength;
    LocalTensor<int32_t> valuesLocal = work[valueBase];
    LocalTensor<int32_t> countLocal = work[valueBase + valueStride];
    LocalTensor<int32_t> scalarLocal =
        work[valueBase + valueStride + tileCountStride];
    LocalTensor<float> bitmapLocal = bitmapBuf.Get<float>();

    int64_t taskCount = batchSize * hitTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / hitTileCount;
        int64_t ht = task - b * hitTileCount;
        int64_t start = ht * hitTileLength;
        int64_t validLen = MinInt64(hitTileLength, k - start);
        LoadInt32(hitLocal, hitGm, b * k + start, validLen);
        LoadInt32(rankLocal, hitRankGm,
                  b * hitStride + start, validLen);
        LoadInt32(valuesLocal, tileValuesGm,
                  b * valueStride, valueStride);
        LoadInt32(countLocal, tileCountsGm,
                  b * tileCountStride, tileCountStride);
        LoadInt32(scalarLocal, missCountGm,
                  b * missCountStride, missCountStride);
        int32_t rowMissCount = scalarLocal.GetValue(0);
        Duplicate(bitmapLocal, 0.0f, lruStride);
        PipeBarrier<PIPE_ALL>();

        for (int64_t p = 0; p < validLen; ++p) {
            int32_t id = hitLocal.GetValue(p);
            int32_t outputId = id;
            if (id == -1) {
                int32_t rank = rankLocal.GetValue(p);
                if (rank >= 0 && rank < rowMissCount) {
                    for (int64_t lt = 0; lt < lruTileCount; ++lt) {
                        int32_t offset =
                            countLocal.GetValue(lt * kDataBlockElements);
                        int32_t count =
                            countLocal.GetValue(lt * kDataBlockElements + 1);
                        if (rank >= offset && rank < offset + count) {
                            outputId = valuesLocal.GetValue(
                                lt * lruTileLength + rank - offset);
                            break;
                        }
                    }
                }
            }
            outputLocal.SetValue(p, outputId);
            bitmapLocal.SetValue(outputId, 1.0f);
        }

        StoreInt32(hitAndMissGm, b * k + start,
                   outputLocal, validLen);
        StoreFloatAtomic(selectedBitmapGm, b * lruStride,
                         bitmapLocal, lruStride);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_keep_local(
    GM_ADDR lru, GM_ADDR selectedBitmap, GM_ADDR tileValues,
    GM_ADDR tileCounts, int64_t batchSize, int64_t lruLength,
    int64_t lruStride, int64_t valueStride, int64_t tileCountStride,
    int64_t lruTileLength, int64_t lruTileCount)
{
    SetAtomicNone();
    GlobalTensor<int32_t> lruGm;
    GlobalTensor<float> bitmapGm;
    GlobalTensor<int32_t> tileValuesGm;
    GlobalTensor<int32_t> tileCountsGm;
    lruGm.SetGlobalBuffer((__gm__ int32_t *)lru, batchSize * lruLength);
    bitmapGm.SetGlobalBuffer((__gm__ float *)selectedBitmap,
                             batchSize * lruStride);
    tileValuesGm.SetGlobalBuffer((__gm__ int32_t *)tileValues,
                                 batchSize * valueStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = lruStride + 2 * lruTileLength + kDataBlockElements;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<float> bitmapLocal = work.ReinterpretCast<float>();
    LocalTensor<int32_t> lruLocal = work[lruStride];
    LocalTensor<int32_t> valuesLocal = work[lruStride + lruTileLength];
    LocalTensor<int32_t> countLocal =
        work[lruStride + 2 * lruTileLength];

    int64_t taskCount = batchSize * lruTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / lruTileCount;
        int64_t lt = task - b * lruTileCount;
        int64_t start = lt * lruTileLength;
        int64_t validLen = MinInt64(lruTileLength, lruLength - start);
        LoadFloat(bitmapLocal, bitmapGm, b * lruStride, lruLength);
        LoadInt32(lruLocal, lruGm, b * lruLength + start, validLen);
        Duplicate(valuesLocal, static_cast<int32_t>(0), lruTileLength);
        Duplicate(countLocal, static_cast<int32_t>(0), kDataBlockElements);
        PipeBarrier<PIPE_ALL>();

        int32_t prefix = 0;
        for (int64_t p = 0; p < validLen; ++p) {
            int32_t id = lruLocal.GetValue(p);
            if (bitmapLocal.GetValue(id) == 0.0f) {
                valuesLocal.SetValue(prefix, id);
                ++prefix;
            }
        }
        countLocal.SetValue(0, prefix);
        countLocal.SetValue(1, prefix);
        StoreInt32(tileValuesGm,
                   b * valueStride + lt * lruTileLength,
                   valuesLocal, lruTileLength);
        StoreInt32(tileCountsGm,
                   b * tileCountStride + lt * kDataBlockElements,
                   countLocal, kDataBlockElements);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_keep_scan(
    GM_ADDR tileCounts, int64_t batchSize,
    int64_t tileCountStride, int64_t lruTileCount)
{
    SetAtomicNone();
    GlobalTensor<int32_t> tileCountsGm;
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = tileCountStride + kDataBlockElements;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> countLocal = work;
    LocalTensor<int32_t> scalarLocal = work[tileCountStride];

    int64_t taskCount = batchSize * lruTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / lruTileCount;
        int64_t lt = task - b * lruTileCount;
        LoadInt32(countLocal, tileCountsGm,
                  b * tileCountStride, tileCountStride);

        int32_t offset = 0;
        for (int64_t tile = 0; tile < lt; ++tile) {
            offset += countLocal.GetValue(tile * kDataBlockElements + 1);
        }
        int32_t ownCount =
            countLocal.GetValue(lt * kDataBlockElements + 1);
        Duplicate(scalarLocal, static_cast<int32_t>(0), kDataBlockElements);
        PipeBarrier<PIPE_ALL>();
        scalarLocal.SetValue(0, offset);
        scalarLocal.SetValue(1, ownCount);
        StoreInt32(tileCountsGm,
                   b * tileCountStride + lt * kDataBlockElements,
                   scalarLocal, kDataBlockElements);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_write(
    GM_ADDR hitAndMiss, GM_ADDR tileValues, GM_ADDR tileCounts,
    GM_ADDR newLru, int64_t batchSize, int64_t k, int64_t lruLength,
    int64_t valueStride, int64_t tileCountStride,
    int64_t lruTileLength, int64_t hitTileLength,
    int64_t lruTileCount, int64_t hitTileCount)
{
    SetAtomicNone();
    GlobalTensor<int32_t> hitAndMissGm;
    GlobalTensor<int32_t> tileValuesGm;
    GlobalTensor<int32_t> tileCountsGm;
    GlobalTensor<int32_t> newLruGm;
    hitAndMissGm.SetGlobalBuffer((__gm__ int32_t *)hitAndMiss,
                                 batchSize * k);
    tileValuesGm.SetGlobalBuffer((__gm__ int32_t *)tileValues,
                                 batchSize * valueStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);
    newLruGm.SetGlobalBuffer((__gm__ int32_t *)newLru,
                             batchSize * lruLength);

    int64_t maxTileLength = lruTileLength > hitTileLength
        ? lruTileLength : hitTileLength;
    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    pipe.InitBuffer(workBuf,
                    static_cast<uint32_t>((maxTileLength +
                                           kDataBlockElements) * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> valuesLocal = work;
    LocalTensor<int32_t> scalarLocal = work[maxTileLength];

    int64_t tasksPerRow = hitTileCount + lruTileCount;
    int64_t taskCount = batchSize * tasksPerRow;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / tasksPerRow;
        int64_t localTask = task - b * tasksPerRow;
        if (localTask < hitTileCount) {
            int64_t start = localTask * hitTileLength;
            int64_t validLen = MinInt64(hitTileLength, k - start);
            LoadInt32(valuesLocal, hitAndMissGm, b * k + start, validLen);
            StoreInt32(newLruGm, b * lruLength + start,
                       valuesLocal, validLen);
        } else {
            int64_t lt = localTask - hitTileCount;
            LoadInt32(scalarLocal, tileCountsGm,
                      b * tileCountStride + lt * kDataBlockElements,
                      kDataBlockElements);
            int32_t offset = scalarLocal.GetValue(0);
            int32_t count = scalarLocal.GetValue(1);
            if (count > 0) {
                LoadInt32(valuesLocal, tileValuesGm,
                          b * valueStride + lt * lruTileLength, count);
                StoreInt32(newLruGm, b * lruLength + k + offset,
                           valuesLocal, count);
            }
        }
    }
}
