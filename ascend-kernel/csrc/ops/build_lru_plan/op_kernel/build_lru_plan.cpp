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

__aicore__ inline int64_t MinInt64(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline void LoadInt32(LocalTensor<int32_t> dst,
                                  GlobalTensor<int32_t> &src,
                                  int64_t srcOffset,
                                  int64_t length)
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

}  // namespace

extern "C" __global__ __aicore__ void build_lru_plan_hit_local(
    GM_ADDR hit, GM_ADDR bitmap, GM_ADDR hitRank, GM_ADDR tileCounts,
    int64_t batchSize, int64_t k, int64_t lruStride, int64_t hitStride,
    int64_t tileCountStride, int64_t hitTileLength, int64_t hitTileCount)
{
    GlobalTensor<int32_t> hitGm;
    GlobalTensor<int32_t> bitmapGm;
    GlobalTensor<int32_t> hitRankGm;
    GlobalTensor<int32_t> tileCountsGm;
    hitGm.SetGlobalBuffer((__gm__ int32_t *)hit, batchSize * k);
    bitmapGm.SetGlobalBuffer((__gm__ int32_t *)bitmap,
                             batchSize * lruStride);
    hitRankGm.SetGlobalBuffer((__gm__ int32_t *)hitRank,
                              batchSize * hitStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = lruStride + 2 * hitTileLength + tileCountStride;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> bitmapLocal = work;
    LocalTensor<int32_t> hitLocal = work[lruStride];
    LocalTensor<int32_t> rankLocal = work[lruStride + hitTileLength];
    LocalTensor<int32_t> countLocal = work[lruStride + 2 * hitTileLength];

    int64_t blockNum = GetBlockNum();
    for (int64_t b = GetBlockIdx(); b < batchSize; b += blockNum) {
        Duplicate(bitmapLocal, static_cast<int32_t>(0), lruStride);
        Duplicate(countLocal, static_cast<int32_t>(0), tileCountStride);
        PipeBarrier<PIPE_ALL>();

        for (int64_t ht = 0; ht < hitTileCount; ++ht) {
            int64_t start = ht * hitTileLength;
            int64_t validLen = MinInt64(hitTileLength, k - start);
            LoadInt32(hitLocal, hitGm, b * k + start, validLen);

            int32_t prefix = 0;
            for (int64_t p = 0; p < validLen; ++p) {
                int32_t id = hitLocal.GetValue(p);
                if (id == -1) {
                    rankLocal.SetValue(p, prefix);
                    ++prefix;
                } else {
                    rankLocal.SetValue(p, static_cast<int32_t>(-1));
                    bitmapLocal.SetValue(id, static_cast<int32_t>(1));
                }
            }
            countLocal.SetValue(ht * 8, prefix);
            StoreInt32(hitRankGm, b * hitStride + start, rankLocal, validLen);
        }

        StoreInt32(bitmapGm, b * lruStride, bitmapLocal, lruStride);
        StoreInt32(tileCountsGm, b * tileCountStride, countLocal, tileCountStride);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_hit_scan(
    GM_ADDR hitRank, GM_ADDR tileCounts, GM_ADDR missCount,
    int64_t batchSize, int64_t k, int64_t hitStride,
    int64_t tileCountStride, int64_t missCountStride,
    int64_t hitTileLength, int64_t hitTileCount)
{
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
    int64_t workElements = hitStride + tileCountStride + missCountStride;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> rankLocal = work;
    LocalTensor<int32_t> countLocal = work[hitStride];
    LocalTensor<int32_t> scalarLocal = work[hitStride + tileCountStride];

    int64_t blockNum = GetBlockNum();
    for (int64_t b = GetBlockIdx(); b < batchSize; b += blockNum) {
        LoadInt32(rankLocal, hitRankGm, b * hitStride, k);
        LoadInt32(countLocal, tileCountsGm, b * tileCountStride, tileCountStride);

        int32_t offset = 0;
        for (int64_t ht = 0; ht < hitTileCount; ++ht) {
            int32_t count = countLocal.GetValue(ht * 8);
            countLocal.SetValue(ht * 8, offset);
            int64_t start = ht * hitTileLength;
            int64_t validLen = MinInt64(hitTileLength, k - start);
            for (int64_t p = 0; p < validLen; ++p) {
                int32_t rank = rankLocal.GetValue(start + p);
                if (rank >= 0) {
                    rankLocal.SetValue(start + p, rank + offset);
                }
            }
            offset += count;
        }

        Duplicate(scalarLocal, static_cast<int32_t>(0), missCountStride);
        PipeBarrier<PIPE_ALL>();
        scalarLocal.SetValue(0, offset);
        StoreInt32(hitRankGm, b * hitStride, rankLocal, k);
        StoreInt32(tileCountsGm, b * tileCountStride, countLocal, tileCountStride);
        StoreInt32(missCountGm, b * missCountStride, scalarLocal, missCountStride);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_candidate_local(
    GM_ADDR lru, GM_ADDR bitmap, GM_ADDR lruRank, GM_ADDR tileCounts,
    int64_t batchSize, int64_t lruLength, int64_t lruStride,
    int64_t tileCountStride, int64_t lruTileLength, int64_t lruTileCount)
{
    GlobalTensor<int32_t> lruGm;
    GlobalTensor<int32_t> bitmapGm;
    GlobalTensor<int32_t> lruRankGm;
    GlobalTensor<int32_t> tileCountsGm;
    lruGm.SetGlobalBuffer((__gm__ int32_t *)lru,
                          batchSize * lruLength);
    bitmapGm.SetGlobalBuffer((__gm__ int32_t *)bitmap,
                             batchSize * lruStride);
    lruRankGm.SetGlobalBuffer((__gm__ int32_t *)lruRank,
                              batchSize * lruStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = lruStride + 2 * lruTileLength + 8;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> bitmapLocal = work;
    LocalTensor<int32_t> lruLocal = work[lruStride];
    LocalTensor<int32_t> rankLocal = work[lruStride + lruTileLength];
    LocalTensor<int32_t> scalarLocal = work[lruStride + 2 * lruTileLength];

    int64_t taskCount = batchSize * lruTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / lruTileCount;
        int64_t lt = task - b * lruTileCount;
        int64_t start = lt * lruTileLength;
        int64_t validLen = MinInt64(lruTileLength, lruLength - start);

        LoadInt32(bitmapLocal, bitmapGm, b * lruStride, lruLength);
        LoadInt32(lruLocal, lruGm, b * lruLength + start, validLen);

        int32_t suffix = 0;
        for (int64_t p = validLen; p > 0; --p) {
            int64_t index = p - 1;
            int32_t id = lruLocal.GetValue(index);
            if (bitmapLocal.GetValue(id) == 0) {
                rankLocal.SetValue(index, suffix);
                ++suffix;
            } else {
                rankLocal.SetValue(index, static_cast<int32_t>(-1));
            }
        }

        Duplicate(scalarLocal, static_cast<int32_t>(0), 8);
        PipeBarrier<PIPE_ALL>();
        scalarLocal.SetValue(0, suffix);
        StoreInt32(lruRankGm, b * lruStride + start, rankLocal, validLen);
        StoreInt32(tileCountsGm, b * tileCountStride + lt * 8, scalarLocal, 8);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_candidate_scan(
    GM_ADDR lruRank, GM_ADDR tileCounts,
    int64_t batchSize, int64_t lruLength, int64_t lruStride,
    int64_t tileCountStride, int64_t lruTileLength, int64_t lruTileCount)
{
    GlobalTensor<int32_t> lruRankGm;
    GlobalTensor<int32_t> tileCountsGm;
    lruRankGm.SetGlobalBuffer((__gm__ int32_t *)lruRank,
                              batchSize * lruStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = lruStride + tileCountStride;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> rankLocal = work;
    LocalTensor<int32_t> countLocal = work[lruStride];

    int64_t blockNum = GetBlockNum();
    for (int64_t b = GetBlockIdx(); b < batchSize; b += blockNum) {
        LoadInt32(rankLocal, lruRankGm, b * lruStride, lruLength);
        LoadInt32(countLocal, tileCountsGm, b * tileCountStride, tileCountStride);

        int32_t offset = 0;
        for (int64_t tile = lruTileCount; tile > 0; --tile) {
            int64_t lt = tile - 1;
            int32_t count = countLocal.GetValue(lt * 8);
            countLocal.SetValue(lt * 8, offset);
            int64_t start = lt * lruTileLength;
            int64_t validLen = MinInt64(lruTileLength, lruLength - start);
            for (int64_t p = 0; p < validLen; ++p) {
                int32_t rank = rankLocal.GetValue(start + p);
                if (rank >= 0) {
                    rankLocal.SetValue(start + p, rank + offset);
                }
            }
            offset += count;
        }

        StoreInt32(lruRankGm, b * lruStride, rankLocal, lruLength);
        StoreInt32(tileCountsGm, b * tileCountStride, countLocal, tileCountStride);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_fill(
    GM_ADDR lru, GM_ADDR hit, GM_ADDR bitmap, GM_ADDR lruRank,
    GM_ADDR hitRank, GM_ADDR missCount, GM_ADDR hitAndMiss,
    int64_t batchSize, int64_t k, int64_t lruLength,
    int64_t lruStride, int64_t hitStride, int64_t missCountStride,
    int64_t lruTileLength, int64_t hitTileLength,
    int64_t lruTileCount, int64_t hitTileCount)
{
    GlobalTensor<int32_t> lruGm;
    GlobalTensor<int32_t> hitGm;
    GlobalTensor<int32_t> bitmapGm;
    GlobalTensor<int32_t> lruRankGm;
    GlobalTensor<int32_t> hitRankGm;
    GlobalTensor<int32_t> missCountGm;
    GlobalTensor<int32_t> hitAndMissGm;
    lruGm.SetGlobalBuffer((__gm__ int32_t *)lru,
                          batchSize * lruLength);
    hitGm.SetGlobalBuffer((__gm__ int32_t *)hit, batchSize * k);
    bitmapGm.SetGlobalBuffer((__gm__ int32_t *)bitmap,
                             batchSize * lruStride);
    lruRankGm.SetGlobalBuffer((__gm__ int32_t *)lruRank,
                              batchSize * lruStride);
    hitRankGm.SetGlobalBuffer((__gm__ int32_t *)hitRank,
                              batchSize * hitStride);
    missCountGm.SetGlobalBuffer((__gm__ int32_t *)missCount,
                                batchSize * missCountStride);
    hitAndMissGm.SetGlobalBuffer((__gm__ int32_t *)hitAndMiss,
                                 batchSize * k);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = lruStride + hitStride + 2 * lruTileLength +
                           3 * hitTileLength + missCountStride;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> bitmapLocal = work;
    LocalTensor<int32_t> missValuesLocal = work[lruStride];
    LocalTensor<int32_t> lruLocal = work[lruStride + hitStride];
    LocalTensor<int32_t> candidateRankLocal =
        work[lruStride + hitStride + lruTileLength];
    int64_t hitBase = lruStride + hitStride + 2 * lruTileLength;
    LocalTensor<int32_t> hitLocal = work[hitBase];
    LocalTensor<int32_t> missRankLocal = work[hitBase + hitTileLength];
    LocalTensor<int32_t> hitOutLocal = work[hitBase + 2 * hitTileLength];
    LocalTensor<int32_t> scalarLocal = work[hitBase + 3 * hitTileLength];

    int64_t blockNum = GetBlockNum();
    for (int64_t b = GetBlockIdx(); b < batchSize; b += blockNum) {
        Duplicate(missValuesLocal, static_cast<int32_t>(0), hitStride);
        LoadInt32(scalarLocal, missCountGm, b * missCountStride, missCountStride);
        int32_t rowMissCount = scalarLocal.GetValue(0);

        for (int64_t lt = 0; lt < lruTileCount; ++lt) {
            int64_t start = lt * lruTileLength;
            int64_t validLen = MinInt64(lruTileLength, lruLength - start);
            LoadInt32(lruLocal, lruGm, b * lruLength + start, validLen);
            LoadInt32(candidateRankLocal, lruRankGm,
                      b * lruStride + start, validLen);
            for (int64_t p = 0; p < validLen; ++p) {
                int32_t rank = candidateRankLocal.GetValue(p);
                if (rank >= 0 && rank < rowMissCount) {
                    missValuesLocal.SetValue(rank, lruLocal.GetValue(p));
                }
            }
        }

        Duplicate(bitmapLocal, static_cast<int32_t>(0), lruStride);
        PipeBarrier<PIPE_ALL>();
        for (int64_t ht = 0; ht < hitTileCount; ++ht) {
            int64_t start = ht * hitTileLength;
            int64_t validLen = MinInt64(hitTileLength, k - start);
            LoadInt32(hitLocal, hitGm, b * k + start, validLen);
            LoadInt32(missRankLocal, hitRankGm,
                      b * hitStride + start, validLen);
            for (int64_t p = 0; p < validLen; ++p) {
                int32_t id = hitLocal.GetValue(p);
                int32_t outputId = id;
                if (id == -1) {
                    int32_t rank = missRankLocal.GetValue(p);
                    outputId = missValuesLocal.GetValue(rank);
                }
                hitOutLocal.SetValue(p, outputId);
                bitmapLocal.SetValue(outputId, static_cast<int32_t>(1));
            }
            StoreInt32(hitAndMissGm, b * k + start, hitOutLocal, validLen);
        }
        StoreInt32(bitmapGm, b * lruStride, bitmapLocal, lruLength);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_keep_local(
    GM_ADDR lru, GM_ADDR bitmap, GM_ADDR lruRank, GM_ADDR tileCounts,
    int64_t batchSize, int64_t lruLength, int64_t lruStride,
    int64_t tileCountStride, int64_t lruTileLength, int64_t lruTileCount)
{
    GlobalTensor<int32_t> lruGm;
    GlobalTensor<int32_t> bitmapGm;
    GlobalTensor<int32_t> lruRankGm;
    GlobalTensor<int32_t> tileCountsGm;
    lruGm.SetGlobalBuffer((__gm__ int32_t *)lru,
                          batchSize * lruLength);
    bitmapGm.SetGlobalBuffer((__gm__ int32_t *)bitmap,
                             batchSize * lruStride);
    lruRankGm.SetGlobalBuffer((__gm__ int32_t *)lruRank,
                              batchSize * lruStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = lruStride + 2 * lruTileLength + 8;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> bitmapLocal = work;
    LocalTensor<int32_t> lruLocal = work[lruStride];
    LocalTensor<int32_t> rankLocal = work[lruStride + lruTileLength];
    LocalTensor<int32_t> scalarLocal = work[lruStride + 2 * lruTileLength];

    int64_t taskCount = batchSize * lruTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / lruTileCount;
        int64_t lt = task - b * lruTileCount;
        int64_t start = lt * lruTileLength;
        int64_t validLen = MinInt64(lruTileLength, lruLength - start);

        LoadInt32(bitmapLocal, bitmapGm, b * lruStride, lruLength);
        LoadInt32(lruLocal, lruGm, b * lruLength + start, validLen);

        int32_t prefix = 0;
        for (int64_t p = 0; p < validLen; ++p) {
            int32_t id = lruLocal.GetValue(p);
            if (bitmapLocal.GetValue(id) == 0) {
                rankLocal.SetValue(p, prefix);
                ++prefix;
            } else {
                rankLocal.SetValue(p, static_cast<int32_t>(-1));
            }
        }

        Duplicate(scalarLocal, static_cast<int32_t>(0), 8);
        PipeBarrier<PIPE_ALL>();
        scalarLocal.SetValue(0, prefix);
        StoreInt32(lruRankGm, b * lruStride + start, rankLocal, validLen);
        StoreInt32(tileCountsGm, b * tileCountStride + lt * 8, scalarLocal, 8);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_keep_scan(
    GM_ADDR lruRank, GM_ADDR tileCounts,
    int64_t batchSize, int64_t lruLength, int64_t lruStride,
    int64_t tileCountStride, int64_t lruTileLength, int64_t lruTileCount)
{
    GlobalTensor<int32_t> lruRankGm;
    GlobalTensor<int32_t> tileCountsGm;
    lruRankGm.SetGlobalBuffer((__gm__ int32_t *)lruRank,
                              batchSize * lruStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = lruStride + tileCountStride;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> rankLocal = work;
    LocalTensor<int32_t> countLocal = work[lruStride];

    int64_t blockNum = GetBlockNum();
    for (int64_t b = GetBlockIdx(); b < batchSize; b += blockNum) {
        LoadInt32(rankLocal, lruRankGm, b * lruStride, lruLength);
        LoadInt32(countLocal, tileCountsGm, b * tileCountStride, tileCountStride);

        int32_t offset = 0;
        for (int64_t lt = 0; lt < lruTileCount; ++lt) {
            int32_t count = countLocal.GetValue(lt * 8);
            countLocal.SetValue(lt * 8, offset);
            int64_t start = lt * lruTileLength;
            int64_t validLen = MinInt64(lruTileLength, lruLength - start);
            for (int64_t p = 0; p < validLen; ++p) {
                int32_t rank = rankLocal.GetValue(start + p);
                if (rank >= 0) {
                    rankLocal.SetValue(start + p, rank + offset);
                }
            }
            offset += count;
        }

        StoreInt32(lruRankGm, b * lruStride, rankLocal, lruLength);
        StoreInt32(tileCountsGm, b * tileCountStride, countLocal, tileCountStride);
    }
}

extern "C" __global__ __aicore__ void build_lru_plan_write(
    GM_ADDR lru, GM_ADDR hitAndMiss, GM_ADDR lruRank,
    GM_ADDR tileCounts, GM_ADDR newLru,
    int64_t batchSize, int64_t k, int64_t lruLength,
    int64_t lruStride, int64_t tileCountStride,
    int64_t lruTileLength, int64_t hitTileLength,
    int64_t lruTileCount, int64_t hitTileCount)
{
    GlobalTensor<int32_t> lruGm;
    GlobalTensor<int32_t> hitAndMissGm;
    GlobalTensor<int32_t> lruRankGm;
    GlobalTensor<int32_t> tileCountsGm;
    GlobalTensor<int32_t> newLruGm;
    lruGm.SetGlobalBuffer((__gm__ int32_t *)lru,
                          batchSize * lruLength);
    hitAndMissGm.SetGlobalBuffer((__gm__ int32_t *)hitAndMiss,
                                 batchSize * k);
    lruRankGm.SetGlobalBuffer((__gm__ int32_t *)lruRank,
                              batchSize * lruStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);
    newLruGm.SetGlobalBuffer((__gm__ int32_t *)newLru,
                             batchSize * lruLength);

    int64_t maxTileLength = lruTileLength > hitTileLength
        ? lruTileLength : hitTileLength;
    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = 3 * maxTileLength + 8;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> inputLocal = work;
    LocalTensor<int32_t> rankLocal = work[maxTileLength];
    LocalTensor<int32_t> outputLocal = work[2 * maxTileLength];
    LocalTensor<int32_t> scalarLocal = work[3 * maxTileLength];

    int64_t blockNum = GetBlockNum();
    for (int64_t b = GetBlockIdx(); b < batchSize; b += blockNum) {
        // One block owns the complete output row. This prevents adjacent,
        // non-32B-aligned compacted tile ranges from being written by
        // different blocks at the same time.
        for (int64_t ht = 0; ht < hitTileCount; ++ht) {
            int64_t start = ht * hitTileLength;
            int64_t validLen = MinInt64(hitTileLength, k - start);
            LoadInt32(inputLocal, hitAndMissGm, b * k + start, validLen);
            StoreInt32(newLruGm, b * lruLength + start, inputLocal, validLen);
        }

        for (int64_t lt = 0; lt < lruTileCount; ++lt) {
            int64_t start = lt * lruTileLength;
            int64_t validLen = MinInt64(lruTileLength, lruLength - start);
            LoadInt32(inputLocal, lruGm, b * lruLength + start, validLen);
            LoadInt32(rankLocal, lruRankGm, b * lruStride + start, validLen);
            LoadInt32(scalarLocal, tileCountsGm,
                      b * tileCountStride + lt * 8, 8);
            int32_t tileOffset = scalarLocal.GetValue(0);
            int64_t compactLen = 0;
            for (int64_t p = 0; p < validLen; ++p) {
                if (rankLocal.GetValue(p) >= 0) {
                    outputLocal.SetValue(compactLen, inputLocal.GetValue(p));
                    ++compactLen;
                }
            }
            if (compactLen > 0) {
                StoreInt32(newLruGm,
                           b * lruLength + k + tileOffset,
                           outputLocal, compactLen);
            }
        }
    }
}
