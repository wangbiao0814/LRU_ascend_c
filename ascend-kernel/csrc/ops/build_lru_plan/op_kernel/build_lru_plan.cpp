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

__aicore__ inline int64_t MaxInt64(int64_t lhs, int64_t rhs)
{
    return lhs > rhs ? lhs : rhs;
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

__aicore__ inline void LoadInt8(LocalTensor<int8_t> dst,
                                 GlobalTensor<int8_t> &src,
                                 int64_t srcOffset, int64_t length)
{
    DataCopyExtParams params{};
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(length * sizeof(int8_t));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<int8_t> padParams{};
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

__aicore__ inline void StoreInt8(GlobalTensor<int8_t> &dst,
                                  int64_t dstOffset,
                                  LocalTensor<int8_t> src,
                                  int64_t length)
{
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams params{};
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(length * sizeof(int8_t));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPad(dst[dstOffset], src, params);
    PipeBarrier<PIPE_ALL>();
}

}  // namespace

// Fused K1 + K2. hitRank remains tile-local; the fill kernel applies the
// per-tile offset produced by the row leader after the barrier.
extern "C" __global__ __aicore__ void build_lru_plan_hit_fused(
    GM_ADDR hit, GM_ADDR hitBitmap, GM_ADDR hitRank, GM_ADDR hitTileInfo,
    GM_ADDR missCount, GM_ADDR syncWorkspace,
    int64_t batchSize, int64_t k, int64_t lruStride, int64_t hitStride,
    int64_t tileCountStride, int64_t missCountStride,
    int64_t hitTileLength, int64_t hitTileCount, int64_t syncStride)
{
    SetAtomicNone();
    GlobalTensor<int32_t> hitGm;
    GlobalTensor<int8_t> bitmapGm;
    GlobalTensor<int32_t> hitRankGm;
    GlobalTensor<int32_t> hitTileInfoGm;
    GlobalTensor<int32_t> missCountGm;
    GlobalTensor<int32_t> syncGm;
    hitGm.SetGlobalBuffer((__gm__ int32_t *)hit, batchSize * k);
    bitmapGm.SetGlobalBuffer((__gm__ int8_t *)hitBitmap,
                             batchSize * lruStride);
    hitRankGm.SetGlobalBuffer((__gm__ int32_t *)hitRank,
                              batchSize * hitStride);
    hitTileInfoGm.SetGlobalBuffer((__gm__ int32_t *)hitTileInfo,
                                  batchSize * tileCountStride);
    missCountGm.SetGlobalBuffer((__gm__ int32_t *)missCount,
                                batchSize * missCountStride);
    syncGm.SetGlobalBuffer((__gm__ int32_t *)syncWorkspace, syncStride);

    int64_t localElements = 2 * hitTileLength + kDataBlockElements;
    int64_t leaderElements = hitStride + tileCountStride + missCountStride;
    int64_t workElements = MaxInt64(localElements, leaderElements);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECOUT> bitmapBuf;
    TBuf<TPosition::VECCALC> syncBuf;
    pipe.InitBuffer(workBuf,
                    static_cast<uint32_t>(workElements * sizeof(int32_t)));
    pipe.InitBuffer(bitmapBuf, static_cast<uint32_t>(lruStride));
    pipe.InitBuffer(syncBuf,
                    static_cast<uint32_t>(syncStride * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int8_t> bitmapLocal = bitmapBuf.Get<int8_t>();
    LocalTensor<int32_t> syncLocal = syncBuf.Get<int32_t>();

    int64_t taskCount = batchSize * hitTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / hitTileCount;
        int64_t ht = task - b * hitTileCount;
        int64_t start = ht * hitTileLength;
        int64_t validLen = MinInt64(hitTileLength, k - start);
        LocalTensor<int32_t> hitLocal = work;
        LocalTensor<int32_t> rankLocal = work[hitTileLength];
        LocalTensor<int32_t> countLocal = work[2 * hitTileLength];

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
            }
        }
        countLocal.SetValue(0, prefix);
        countLocal.SetValue(1, prefix);
        StoreInt32(hitRankGm, b * hitStride + start, rankLocal, validLen);
        StoreInt32(hitTileInfoGm,
                   b * tileCountStride + ht * kDataBlockElements,
                   countLocal, kDataBlockElements);
    }

    SyncAll(syncGm, syncLocal, static_cast<int32_t>(blockNum));

    for (int64_t b = GetBlockIdx(); b < batchSize; b += blockNum) {
        LocalTensor<int32_t> hitLocal = work;
        LocalTensor<int32_t> countLocal = work[hitStride];
        LocalTensor<int32_t> scalarLocal = work[hitStride + tileCountStride];
        int64_t countElements = hitTileCount * kDataBlockElements;
        LoadInt32(hitLocal, hitGm, b * k, k);
        LoadInt32(countLocal, hitTileInfoGm,
                  b * tileCountStride, countElements);
        Duplicate(bitmapLocal, static_cast<int8_t>(0), lruStride);
        Duplicate(scalarLocal, static_cast<int32_t>(0), missCountStride);
        PipeBarrier<PIPE_ALL>();

        int32_t offset = 0;
        for (int64_t ht = 0; ht < hitTileCount; ++ht) {
            int64_t slot = ht * kDataBlockElements;
            int32_t count = countLocal.GetValue(slot + 1);
            countLocal.SetValue(slot, offset);
            countLocal.SetValue(slot + 1, count);
            offset += count;
        }
        for (int64_t p = 0; p < k; ++p) {
            int32_t id = hitLocal.GetValue(p);
            if (id >= 0) {
                bitmapLocal.SetValue(id, static_cast<int8_t>(1));
            }
        }
        scalarLocal.SetValue(0, offset);

        StoreInt32(hitTileInfoGm, b * tileCountStride,
                   countLocal, countElements);
        StoreInt32(missCountGm, b * missCountStride,
                   scalarLocal, missCountStride);
        StoreInt8(bitmapGm, b * lruStride, bitmapLocal, lruStride);
    }
}

// Fused K3 + K4 + K5. A row leader packs candidates and builds the selected
// bitmap once, eliminating the old per-hit-tile bitmap atomic writes.
extern "C" __global__ __aicore__ void build_lru_plan_candidate_fill_fused(
    GM_ADDR lru, GM_ADDR hit, GM_ADDR hitBitmap, GM_ADDR hitRank,
    GM_ADDR hitTileInfo, GM_ADDR missCount, GM_ADDR tileValues,
    GM_ADDR tileCounts, GM_ADDR candidatePacked, GM_ADDR selectedBitmap,
    GM_ADDR hitAndMiss, GM_ADDR newLru, GM_ADDR syncWorkspace,
    int64_t batchSize, int64_t k, int64_t lruLength,
    int64_t lruStride, int64_t hitStride, int64_t tileCountStride,
    int64_t valueStride, int64_t missCountStride,
    int64_t lruTileLength, int64_t hitTileLength,
    int64_t lruTileCount, int64_t hitTileCount, int64_t syncStride)
{
    SetAtomicNone();
    GlobalTensor<int32_t> lruGm;
    GlobalTensor<int32_t> hitGm;
    GlobalTensor<int8_t> hitBitmapGm;
    GlobalTensor<int32_t> hitRankGm;
    GlobalTensor<int32_t> hitTileInfoGm;
    GlobalTensor<int32_t> missCountGm;
    GlobalTensor<int32_t> tileValuesGm;
    GlobalTensor<int32_t> tileCountsGm;
    GlobalTensor<int32_t> candidatePackedGm;
    GlobalTensor<int8_t> selectedBitmapGm;
    GlobalTensor<int32_t> hitAndMissGm;
    GlobalTensor<int32_t> newLruGm;
    GlobalTensor<int32_t> sync0Gm;
    GlobalTensor<int32_t> sync1Gm;
    lruGm.SetGlobalBuffer((__gm__ int32_t *)lru, batchSize * lruLength);
    hitGm.SetGlobalBuffer((__gm__ int32_t *)hit, batchSize * k);
    hitBitmapGm.SetGlobalBuffer((__gm__ int8_t *)hitBitmap,
                                batchSize * lruStride);
    hitRankGm.SetGlobalBuffer((__gm__ int32_t *)hitRank,
                              batchSize * hitStride);
    hitTileInfoGm.SetGlobalBuffer((__gm__ int32_t *)hitTileInfo,
                                  batchSize * tileCountStride);
    missCountGm.SetGlobalBuffer((__gm__ int32_t *)missCount,
                                batchSize * missCountStride);
    tileValuesGm.SetGlobalBuffer((__gm__ int32_t *)tileValues,
                                 batchSize * valueStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);
    candidatePackedGm.SetGlobalBuffer((__gm__ int32_t *)candidatePacked,
                                      batchSize * hitStride);
    selectedBitmapGm.SetGlobalBuffer((__gm__ int8_t *)selectedBitmap,
                                     batchSize * lruStride);
    hitAndMissGm.SetGlobalBuffer((__gm__ int32_t *)hitAndMiss,
                                 batchSize * k);
    newLruGm.SetGlobalBuffer((__gm__ int32_t *)newLru,
                             batchSize * lruLength);
    sync0Gm.SetGlobalBuffer((__gm__ int32_t *)syncWorkspace, syncStride);
    sync1Gm.SetGlobalBuffer(((__gm__ int32_t *)syncWorkspace) + syncStride,
                            syncStride);

    int64_t candidateElements = 2 * lruTileLength + kDataBlockElements;
    int64_t leaderElements = valueStride + tileCountStride +
                             hitStride + missCountStride;
    int64_t fillElements = 3 * hitTileLength + hitStride +
                           2 * kDataBlockElements;
    int64_t workElements = MaxInt64(candidateElements,
                                    MaxInt64(leaderElements, fillElements));

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> bitmapBuf;
    TBuf<TPosition::VECCALC> syncBuf;
    pipe.InitBuffer(workBuf,
                    static_cast<uint32_t>(workElements * sizeof(int32_t)));
    pipe.InitBuffer(bitmapBuf, static_cast<uint32_t>(lruStride));
    pipe.InitBuffer(syncBuf,
                    static_cast<uint32_t>(syncStride * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int8_t> bitmapLocal = bitmapBuf.Get<int8_t>();
    LocalTensor<int32_t> syncLocal = syncBuf.Get<int32_t>();

    int64_t lruTaskCount = batchSize * lruTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < lruTaskCount; task += blockNum) {
        int64_t b = task / lruTileCount;
        int64_t lt = task - b * lruTileCount;
        int64_t start = lt * lruTileLength;
        int64_t validLen = MinInt64(lruTileLength, lruLength - start);
        LocalTensor<int32_t> lruLocal = work;
        LocalTensor<int32_t> valuesLocal = work[lruTileLength];
        LocalTensor<int32_t> countLocal = work[2 * lruTileLength];
        LoadInt8(bitmapLocal, hitBitmapGm, b * lruStride, lruLength);
        LoadInt32(lruLocal, lruGm, b * lruLength + start, validLen);
        Duplicate(valuesLocal, static_cast<int32_t>(0), lruTileLength);
        Duplicate(countLocal, static_cast<int32_t>(0), kDataBlockElements);
        PipeBarrier<PIPE_ALL>();

        int32_t suffix = 0;
        for (int64_t p = validLen; p > 0; --p) {
            int64_t index = p - 1;
            int32_t id = lruLocal.GetValue(index);
            if (bitmapLocal.GetValue(id) == static_cast<int8_t>(0)) {
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

    SyncAll(sync0Gm, syncLocal, static_cast<int32_t>(blockNum));

    for (int64_t b = GetBlockIdx(); b < batchSize; b += blockNum) {
        LocalTensor<int32_t> valuesLocal = work;
        LocalTensor<int32_t> countLocal = work[valueStride];
        LocalTensor<int32_t> packedLocal =
            work[valueStride + tileCountStride];
        LocalTensor<int32_t> scalarLocal =
            work[valueStride + tileCountStride + hitStride];
        int64_t countElements = lruTileCount * kDataBlockElements;
        LoadInt32(valuesLocal, tileValuesGm, b * valueStride, valueStride);
        LoadInt32(countLocal, tileCountsGm,
                  b * tileCountStride, countElements);
        LoadInt32(scalarLocal, missCountGm,
                  b * missCountStride, missCountStride);
        LoadInt8(bitmapLocal, hitBitmapGm, b * lruStride, lruLength);
        Duplicate(packedLocal, static_cast<int32_t>(0), hitStride);
        PipeBarrier<PIPE_ALL>();

        int32_t rowMissCount = scalarLocal.GetValue(0);
        int32_t packedCount = 0;
        for (int64_t lt = lruTileCount; lt > 0; --lt) {
            int64_t tile = lt - 1;
            int32_t count =
                countLocal.GetValue(tile * kDataBlockElements + 1);
            for (int32_t p = 0; p < count && packedCount < rowMissCount; ++p) {
                int32_t id = valuesLocal.GetValue(tile * lruTileLength + p);
                packedLocal.SetValue(packedCount, id);
                bitmapLocal.SetValue(id, static_cast<int8_t>(1));
                ++packedCount;
            }
        }
        StoreInt32(candidatePackedGm, b * hitStride, packedLocal, k);
        StoreInt8(selectedBitmapGm, b * lruStride,
                  bitmapLocal, lruLength);
    }

    SyncAll(sync1Gm, syncLocal, static_cast<int32_t>(blockNum));

    int64_t hitTaskCount = batchSize * hitTileCount;
    for (int64_t task = GetBlockIdx(); task < hitTaskCount; task += blockNum) {
        int64_t b = task / hitTileCount;
        int64_t ht = task - b * hitTileCount;
        int64_t start = ht * hitTileLength;
        int64_t validLen = MinInt64(hitTileLength, k - start);
        LocalTensor<int32_t> hitLocal = work;
        LocalTensor<int32_t> rankLocal = work[hitTileLength];
        LocalTensor<int32_t> outputLocal = work[2 * hitTileLength];
        LocalTensor<int32_t> packedLocal = work[3 * hitTileLength];
        LocalTensor<int32_t> infoLocal =
            work[3 * hitTileLength + hitStride];
        LocalTensor<int32_t> scalarLocal =
            work[3 * hitTileLength + hitStride + kDataBlockElements];
        LoadInt32(hitLocal, hitGm, b * k + start, validLen);
        LoadInt32(rankLocal, hitRankGm,
                  b * hitStride + start, validLen);
        LoadInt32(packedLocal, candidatePackedGm,
                  b * hitStride, k);
        LoadInt32(infoLocal, hitTileInfoGm,
                  b * tileCountStride + ht * kDataBlockElements,
                  kDataBlockElements);
        LoadInt32(scalarLocal, missCountGm,
                  b * missCountStride, missCountStride);

        int32_t tileOffset = infoLocal.GetValue(0);
        int32_t rowMissCount = scalarLocal.GetValue(0);
        for (int64_t p = 0; p < validLen; ++p) {
            int32_t id = hitLocal.GetValue(p);
            int32_t outputId = id;
            if (id == -1) {
                int32_t rank = rankLocal.GetValue(p) + tileOffset;
                if (rank >= 0 && rank < rowMissCount) {
                    outputId = packedLocal.GetValue(rank);
                }
            }
            outputLocal.SetValue(p, outputId);
        }
        StoreInt32(hitAndMissGm, b * k + start,
                   outputLocal, validLen);
        StoreInt32(newLruGm, b * lruLength + start,
                   outputLocal, validLen);
    }
}

// Fused K6 + K7 + K8. The last phase writes only newLru[:, K:] because the
// fused fill kernel already writes the first K values.
extern "C" __global__ __aicore__ void build_lru_plan_keep_write_fused(
    GM_ADDR lru, GM_ADDR selectedBitmap, GM_ADDR tileValues,
    GM_ADDR tileCounts, GM_ADDR newLru, GM_ADDR syncWorkspace,
    int64_t batchSize, int64_t k, int64_t lruLength,
    int64_t lruStride, int64_t valueStride, int64_t tileCountStride,
    int64_t lruTileLength, int64_t lruTileCount, int64_t syncStride)
{
    SetAtomicNone();
    GlobalTensor<int32_t> lruGm;
    GlobalTensor<int8_t> bitmapGm;
    GlobalTensor<int32_t> tileValuesGm;
    GlobalTensor<int32_t> tileCountsGm;
    GlobalTensor<int32_t> newLruGm;
    GlobalTensor<int32_t> sync0Gm;
    GlobalTensor<int32_t> sync1Gm;
    lruGm.SetGlobalBuffer((__gm__ int32_t *)lru, batchSize * lruLength);
    bitmapGm.SetGlobalBuffer((__gm__ int8_t *)selectedBitmap,
                             batchSize * lruStride);
    tileValuesGm.SetGlobalBuffer((__gm__ int32_t *)tileValues,
                                 batchSize * valueStride);
    tileCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileCounts,
                                 batchSize * tileCountStride);
    newLruGm.SetGlobalBuffer((__gm__ int32_t *)newLru,
                             batchSize * lruLength);
    sync0Gm.SetGlobalBuffer((__gm__ int32_t *)syncWorkspace, syncStride);
    sync1Gm.SetGlobalBuffer(((__gm__ int32_t *)syncWorkspace) + syncStride,
                            syncStride);

    int64_t localElements = 2 * lruTileLength + kDataBlockElements;
    int64_t scanElements =
        lruTileCount * kDataBlockElements + kDataBlockElements;
    int64_t writeElements = lruTileLength + kDataBlockElements;
    int64_t workElements = MaxInt64(localElements,
                                    MaxInt64(scanElements, writeElements));

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> bitmapBuf;
    TBuf<TPosition::VECCALC> syncBuf;
    pipe.InitBuffer(workBuf,
                    static_cast<uint32_t>(workElements * sizeof(int32_t)));
    pipe.InitBuffer(bitmapBuf, static_cast<uint32_t>(lruStride));
    pipe.InitBuffer(syncBuf,
                    static_cast<uint32_t>(syncStride * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int8_t> bitmapLocal = bitmapBuf.Get<int8_t>();
    LocalTensor<int32_t> syncLocal = syncBuf.Get<int32_t>();

    int64_t taskCount = batchSize * lruTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / lruTileCount;
        int64_t lt = task - b * lruTileCount;
        int64_t start = lt * lruTileLength;
        int64_t validLen = MinInt64(lruTileLength, lruLength - start);
        LocalTensor<int32_t> lruLocal = work;
        LocalTensor<int32_t> valuesLocal = work[lruTileLength];
        LocalTensor<int32_t> countLocal = work[2 * lruTileLength];
        LoadInt8(bitmapLocal, bitmapGm, b * lruStride, lruLength);
        LoadInt32(lruLocal, lruGm, b * lruLength + start, validLen);
        Duplicate(valuesLocal, static_cast<int32_t>(0), lruTileLength);
        Duplicate(countLocal, static_cast<int32_t>(0), kDataBlockElements);
        PipeBarrier<PIPE_ALL>();

        int32_t prefix = 0;
        for (int64_t p = 0; p < validLen; ++p) {
            int32_t id = lruLocal.GetValue(p);
            if (bitmapLocal.GetValue(id) == static_cast<int8_t>(0)) {
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

    SyncAll(sync0Gm, syncLocal, static_cast<int32_t>(blockNum));

    for (int64_t b = GetBlockIdx(); b < batchSize; b += blockNum) {
        LocalTensor<int32_t> countLocal = work;
        int64_t countElements = lruTileCount * kDataBlockElements;
        LoadInt32(countLocal, tileCountsGm,
                  b * tileCountStride, countElements);
        int32_t offset = 0;
        for (int64_t lt = 0; lt < lruTileCount; ++lt) {
            int64_t slot = lt * kDataBlockElements;
            int32_t count = countLocal.GetValue(slot + 1);
            countLocal.SetValue(slot, offset);
            countLocal.SetValue(slot + 1, count);
            offset += count;
        }
        StoreInt32(tileCountsGm, b * tileCountStride,
                   countLocal, countElements);
    }

    SyncAll(sync1Gm, syncLocal, static_cast<int32_t>(blockNum));

    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / lruTileCount;
        int64_t lt = task - b * lruTileCount;
        LocalTensor<int32_t> valuesLocal = work;
        LocalTensor<int32_t> scalarLocal = work[lruTileLength];
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
