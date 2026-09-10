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

__aicore__ inline void CopyInInt32(LocalTensor<int32_t> dst,
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
}

__aicore__ inline void CopyInFloat(LocalTensor<float> dst,
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
}

__aicore__ inline void CopyInUint8(LocalTensor<uint8_t> dst,
                                   GlobalTensor<uint8_t> &src,
                                   int64_t srcOffset, int64_t length)
{
    DataCopyExtParams params{};
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(length * sizeof(uint8_t));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<uint8_t> padParams{};
    padParams.isPad = false;
    DataCopyPad(dst, src[srcOffset], params, padParams);
}

__aicore__ inline void CopyOutInt32(GlobalTensor<int32_t> &dst,
                                    int64_t dstOffset,
                                    LocalTensor<int32_t> src,
                                    int64_t length)
{
    DataCopyExtParams params{};
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(length * sizeof(int32_t));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPad(dst[dstOffset], src, params);
}

__aicore__ inline void CopyOutFloat(GlobalTensor<float> &dst,
                                    int64_t dstOffset,
                                    LocalTensor<float> src,
                                    int64_t length)
{
    DataCopyExtParams params{};
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(length * sizeof(float));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPad(dst[dstOffset], src, params);
}

__aicore__ inline void WaitMte2ToScalar()
{
    event_t eventId = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
}

__aicore__ inline void WaitScalarToMte3()
{
    event_t eventId = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId);
    WaitFlag<HardEvent::S_MTE3>(eventId);
}

__aicore__ inline void WaitMte3ToScalar()
{
    event_t eventId = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventId);
    WaitFlag<HardEvent::MTE3_S>(eventId);
}

}  // namespace

// Stage 1: build the hit bitmap, tile-local miss ranks, and compact counts.
extern "C" __global__ __aicore__ void build_lru_plan_hit_local(
    GM_ADDR hit, GM_ADDR hitBitmap, GM_ADDR hitCounts,
    int64_t batchSize, int64_t k, int64_t lruStride,
    int64_t hitMetaStride, int64_t hitTileLength, int64_t hitTileCount)
{
    SetAtomicNone();
    GlobalTensor<int32_t> hitGm;
    GlobalTensor<float> bitmapGm;
    GlobalTensor<int32_t> hitCountsGm;
    hitGm.SetGlobalBuffer((__gm__ int32_t *)hit, batchSize * k);
    bitmapGm.SetGlobalBuffer((__gm__ float *)hitBitmap,
                             batchSize * lruStride);
    hitCountsGm.SetGlobalBuffer((__gm__ int32_t *)hitCounts,
                                batchSize * hitMetaStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    pipe.InitBuffer(workBuf,
                    static_cast<uint32_t>((2 * hitTileLength +
                                           kDataBlockElements) * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> hitLocal = work;
    LocalTensor<float> oneLocal = work[hitTileLength].ReinterpretCast<float>();
    LocalTensor<int32_t> countLocal = work[2 * hitTileLength];

    int64_t taskCount = batchSize * hitTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / hitTileCount;
        int64_t ht = task - b * hitTileCount;
        int64_t start = ht * hitTileLength;
        int64_t validLen = MinInt64(hitTileLength, k - start);

        CopyInInt32(hitLocal, hitGm, b * k + start, validLen);
        WaitMte2ToScalar();

        int32_t prefix = 0;
        for (int64_t p = 0; p < validLen; ++p) {
            int32_t id = hitLocal.GetValue(p);
            if (id == -1) {
                ++prefix;
            } else {
                oneLocal.SetValue(p, 1.0f);
            }
        }
        countLocal.SetValue(0, prefix);

        WaitScalarToMte3();
        CopyOutInt32(hitCountsGm, b * hitMetaStride + ht,
                     countLocal, 1);
        // Valid hit IDs are unique within each row by contract. Every sparse
        // store therefore owns a distinct logical float in hitBitmap, even
        // when different hit tiles execute concurrently. DataCopyPad writes
        // only the requested four bytes, so no atomic merge is required.
        for (int64_t p = 0; p < validLen; ++p) {
            int32_t id = hitLocal.GetValue(p);
            if (id != -1) {
                CopyOutFloat(bitmapGm, b * lruStride + id,
                             oneLocal[p], 1);
            }
        }
        WaitMte3ToScalar();
    }
}

// Stage 2: compact non-hit LRU IDs per tile in eviction order.
extern "C" __global__ __aicore__ void build_lru_plan_candidate_local(
    GM_ADDR lru, GM_ADDR hitBitmap, GM_ADDR candidateValues,
    GM_ADDR candidateCounts, int64_t batchSize, int64_t lruLength,
    int64_t lruStride, int64_t valueStride, int64_t lruMetaStride,
    int64_t lruTileLength, int64_t lruTileCount)
{
    SetAtomicNone();
    GlobalTensor<int32_t> lruGm;
    GlobalTensor<float> bitmapGm;
    GlobalTensor<int32_t> candidateValuesGm;
    GlobalTensor<int32_t> candidateCountsGm;
    lruGm.SetGlobalBuffer((__gm__ int32_t *)lru,
                           batchSize * lruLength);
    bitmapGm.SetGlobalBuffer((__gm__ float *)hitBitmap,
                             batchSize * lruStride);
    candidateValuesGm.SetGlobalBuffer((__gm__ int32_t *)candidateValues,
                                      batchSize * valueStride);
    candidateCountsGm.SetGlobalBuffer((__gm__ int32_t *)candidateCounts,
                                      batchSize * lruMetaStride);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = lruStride + 2 * lruTileLength +
                           kDataBlockElements;
    pipe.InitBuffer(workBuf,
                    static_cast<uint32_t>(workElements * sizeof(int32_t)));
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

        CopyInFloat(bitmapLocal, bitmapGm, b * lruStride, lruLength);
        CopyInInt32(lruLocal, lruGm, b * lruLength + start, validLen);
        WaitMte2ToScalar();

        int32_t count = 0;
        for (int64_t p = validLen; p > 0; --p) {
            int32_t id = lruLocal.GetValue(p - 1);
            if (bitmapLocal.GetValue(id) == 0.0f) {
                valuesLocal.SetValue(count, id);
                ++count;
            }
        }
        countLocal.SetValue(0, count);

        WaitScalarToMte3();
        if (count > 0) {
            CopyOutInt32(candidateValuesGm,
                         b * valueStride + lt * lruTileLength,
                         valuesLocal, count);
        }
        CopyOutInt32(candidateCountsGm,
                     b * lruMetaStride + lt, countLocal, 1);
        WaitMte3ToScalar();
    }
}

// Stage 3: fill misses in-place and, in independent LRU-tile tasks,
// compact the IDs that remain after eviction.
extern "C" __global__ __aicore__ void build_lru_plan_materialize(
    GM_ADDR lru, GM_ADDR hit, GM_ADDR hitMask, GM_ADDR hitBitmap,
    GM_ADDR hitCounts, GM_ADDR candidateValues, GM_ADDR candidateCounts,
    GM_ADDR keepValues, GM_ADDR keepCounts, GM_ADDR newLru,
    int64_t batchSize, int64_t k, int64_t lruLength,
    int64_t lruStride, int64_t hitMetaStride, int64_t lruMetaStride,
    int64_t valueStride, int64_t hitTileLength, int64_t lruTileLength,
    int64_t hitTileCount, int64_t lruTileCount)
{
    SetAtomicNone();
    GlobalTensor<int32_t> lruGm;
    GlobalTensor<int32_t> hitGm;
    GlobalTensor<uint8_t> hitMaskGm;
    GlobalTensor<float> bitmapGm;
    GlobalTensor<int32_t> hitCountsGm;
    GlobalTensor<int32_t> candidateValuesGm;
    GlobalTensor<int32_t> candidateCountsGm;
    GlobalTensor<int32_t> keepValuesGm;
    GlobalTensor<int32_t> keepCountsGm;
    GlobalTensor<int32_t> newLruGm;
    lruGm.SetGlobalBuffer((__gm__ int32_t *)lru,
                           batchSize * lruLength);
    hitGm.SetGlobalBuffer((__gm__ int32_t *)hit, batchSize * k);
    hitMaskGm.SetGlobalBuffer((__gm__ uint8_t *)hitMask, batchSize * k);
    bitmapGm.SetGlobalBuffer((__gm__ float *)hitBitmap,
                             batchSize * lruStride);
    hitCountsGm.SetGlobalBuffer((__gm__ int32_t *)hitCounts,
                                batchSize * hitMetaStride);
    candidateValuesGm.SetGlobalBuffer((__gm__ int32_t *)candidateValues,
                                      batchSize * valueStride);
    candidateCountsGm.SetGlobalBuffer((__gm__ int32_t *)candidateCounts,
                                      batchSize * lruMetaStride);
    keepValuesGm.SetGlobalBuffer((__gm__ int32_t *)keepValues,
                                 batchSize * valueStride);
    keepCountsGm.SetGlobalBuffer((__gm__ int32_t *)keepCounts,
                                 batchSize * lruMetaStride);
    newLruGm.SetGlobalBuffer((__gm__ int32_t *)newLru,
                             batchSize * lruLength);

    int64_t metadataElements = hitMetaStride + lruMetaStride;
    int64_t maskBytes = (hitTileLength + 31) / 32 * 32;
    int64_t hitBranchBytes = valueStride * sizeof(int32_t) +
                             metadataElements * sizeof(int32_t) +
                             2 * hitTileLength * sizeof(int32_t) +
                             kDataBlockElements * sizeof(int32_t) + maskBytes;
    int64_t lruBranchBytes = lruStride * sizeof(float) +
                             metadataElements * sizeof(int32_t) +
                             2 * lruTileLength * sizeof(int32_t) +
                             kDataBlockElements * sizeof(int32_t);
    int64_t workBytes = hitBranchBytes > lruBranchBytes
        ? hitBranchBytes : lruBranchBytes;

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workBytes));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<uint8_t> workBytesLocal = work.ReinterpretCast<uint8_t>();

    int64_t tasksPerRow = hitTileCount + lruTileCount;
    int64_t taskCount = batchSize * tasksPerRow;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / tasksPerRow;
        int64_t localTask = task - b * tasksPerRow;

        if (localTask < hitTileCount) {
            int64_t ht = localTask;
            int64_t start = ht * hitTileLength;
            int64_t validLen = MinInt64(hitTileLength, k - start);

            LocalTensor<int32_t> candidateValueLocal = work;
            int64_t hitBase = valueStride;
            LocalTensor<int32_t> hitLocal = work[hitBase];
            int64_t outputBase = hitBase + hitTileLength;
            LocalTensor<int32_t> outputLocal = work[outputBase];
            int64_t hitCountBase = outputBase + hitTileLength;
            LocalTensor<int32_t> hitCountLocal = work[hitCountBase];
            int64_t candidateCountBase = hitCountBase + hitMetaStride;
            LocalTensor<int32_t> candidateCountLocal = work[candidateCountBase];
            int64_t scalarBase = candidateCountBase + lruMetaStride;
            LocalTensor<uint8_t> maskLocal = workBytesLocal[
                (scalarBase + kDataBlockElements) * sizeof(int32_t)];

            CopyInInt32(hitLocal, hitGm, b * k + start, validLen);
            CopyInUint8(maskLocal, hitMaskGm, b * k + start, validLen);
            CopyInInt32(hitCountLocal, hitCountsGm,
                        b * hitMetaStride, hitMetaStride);
            CopyInInt32(candidateCountLocal, candidateCountsGm,
                        b * lruMetaStride, lruMetaStride);
            CopyInInt32(candidateValueLocal, candidateValuesGm,
                        b * valueStride, valueStride);
            WaitMte2ToScalar();

            int32_t missBase = 0;
            for (int64_t tile = 0; tile < ht; ++tile) {
                missBase += hitCountLocal.GetValue(tile);
            }

            int32_t localMissRank = 0;
            for (int64_t p = 0; p < validLen; ++p) {
                int32_t outputId = hitLocal.GetValue(p);
                if (maskLocal.GetValue(p) == 0) {
                    int32_t rank = missBase + localMissRank;
                    for (int64_t tile = lruTileCount; tile > 0; --tile) {
                        int64_t lt = tile - 1;
                        int32_t count = candidateCountLocal.GetValue(lt);
                        if (rank < count) {
                            outputId = candidateValueLocal.GetValue(
                                lt * lruTileLength + rank);
                            break;
                        }
                        rank -= count;
                    }
                    ++localMissRank;
                }
                outputLocal.SetValue(p, outputId);
            }

            WaitScalarToMte3();
            CopyOutInt32(hitGm, b * k + start, outputLocal, validLen);
            CopyOutInt32(newLruGm, b * lruLength + start,
                         outputLocal, validLen);
            WaitMte3ToScalar();
        } else {
            int64_t lt = localTask - hitTileCount;
            int64_t start = lt * lruTileLength;
            int64_t validLen = MinInt64(lruTileLength, lruLength - start);

            LocalTensor<float> bitmapLocal = work.ReinterpretCast<float>();
            int64_t lruBase = lruStride;
            LocalTensor<int32_t> lruLocal = work[lruBase];
            int64_t outputBase = lruBase + lruTileLength;
            LocalTensor<int32_t> outputLocal = work[outputBase];
            int64_t hitCountBase = outputBase + lruTileLength;
            LocalTensor<int32_t> hitCountLocal = work[hitCountBase];
            int64_t candidateCountBase = hitCountBase + hitMetaStride;
            LocalTensor<int32_t> candidateCountLocal = work[candidateCountBase];
            LocalTensor<int32_t> scalarLocal =
                work[candidateCountBase + lruMetaStride];

            CopyInFloat(bitmapLocal, bitmapGm,
                        b * lruStride, lruLength);
            CopyInInt32(lruLocal, lruGm,
                        b * lruLength + start, validLen);
            CopyInInt32(hitCountLocal, hitCountsGm,
                        b * hitMetaStride, hitMetaStride);
            CopyInInt32(candidateCountLocal, candidateCountsGm,
                        b * lruMetaStride, lruMetaStride);
            WaitMte2ToScalar();

            int32_t rowMissCount = 0;
            for (int64_t tile = 0; tile < hitTileCount; ++tile) {
                rowMissCount += hitCountLocal.GetValue(tile);
            }
            int32_t suffixOffset = 0;
            for (int64_t tile = lt + 1; tile < lruTileCount; ++tile) {
                suffixOffset += candidateCountLocal.GetValue(tile);
            }
            int32_t withinReverse = candidateCountLocal.GetValue(lt);
            int32_t keepCount = 0;
            for (int64_t p = 0; p < validLen; ++p) {
                int32_t id = lruLocal.GetValue(p);
                if (bitmapLocal.GetValue(id) == 0.0f) {
                    --withinReverse;
                    int32_t reverseRank = suffixOffset + withinReverse;
                    if (reverseRank >= rowMissCount) {
                        outputLocal.SetValue(keepCount, id);
                        ++keepCount;
                    }
                }
            }
            scalarLocal.SetValue(0, keepCount);

            WaitScalarToMte3();
            if (keepCount > 0) {
                CopyOutInt32(keepValuesGm,
                             b * valueStride + lt * lruTileLength,
                             outputLocal, keepCount);
            }
            CopyOutInt32(keepCountsGm,
                         b * lruMetaStride + lt, scalarLocal, 1);
            WaitMte3ToScalar();
        }
    }
}

// Stage 4: compute each keep tile's compact prefix and immediately write it.
extern "C" __global__ __aicore__ void build_lru_plan_keep_scan_write(
    GM_ADDR keepValues, GM_ADDR keepCounts, GM_ADDR newLru,
    int64_t batchSize, int64_t k, int64_t lruLength,
    int64_t valueStride, int64_t lruMetaStride,
    int64_t lruTileLength, int64_t lruTileCount)
{
    SetAtomicNone();
    GlobalTensor<int32_t> keepValuesGm;
    GlobalTensor<int32_t> keepCountsGm;
    GlobalTensor<int32_t> newLruGm;
    keepValuesGm.SetGlobalBuffer((__gm__ int32_t *)keepValues,
                                 batchSize * valueStride);
    keepCountsGm.SetGlobalBuffer((__gm__ int32_t *)keepCounts,
                                 batchSize * lruMetaStride);
    newLruGm.SetGlobalBuffer((__gm__ int32_t *)newLru,
                             batchSize * lruLength);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    int64_t workElements = lruMetaStride + lruTileLength;
    pipe.InitBuffer(workBuf,
                    static_cast<uint32_t>(workElements * sizeof(int32_t)));
    LocalTensor<int32_t> work = workBuf.Get<int32_t>();
    LocalTensor<int32_t> countLocal = work;
    LocalTensor<int32_t> valuesLocal = work[lruMetaStride];

    int64_t taskCount = batchSize * lruTileCount;
    int64_t blockNum = GetBlockNum();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += blockNum) {
        int64_t b = task / lruTileCount;
        int64_t lt = task - b * lruTileCount;
        CopyInInt32(countLocal, keepCountsGm,
                    b * lruMetaStride, lruMetaStride);
        WaitMte2ToScalar();

        int32_t offset = 0;
        for (int64_t tile = 0; tile < lt; ++tile) {
            offset += countLocal.GetValue(tile);
        }
        int32_t count = countLocal.GetValue(lt);
        if (count > 0) {
            CopyInInt32(valuesLocal, keepValuesGm,
                        b * valueStride + lt * lruTileLength, count);
            WaitMte2ToScalar();
            WaitScalarToMte3();
            CopyOutInt32(newLruGm,
                         b * lruLength + k + offset,
                         valuesLocal, count);
            WaitMte3ToScalar();
        }
    }
}
