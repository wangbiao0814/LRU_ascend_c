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

__aicore__ inline void WaitMte2ToScalar()
{
    event_t eventId = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
}

__aicore__ inline void WaitVectorToScalar()
{
    event_t eventId = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
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

extern "C" __global__ __aicore__ void build_lru_plan_row(
    GM_ADDR lru, GM_ADDR hit, GM_ADDR hitMask, GM_ADDR newLru,
    int64_t batchSize, int64_t k, int64_t lruLength,
    int64_t bitmapElements, int64_t maskBytes,
    int64_t hitElements, int64_t lruElements)
{
    SetAtomicNone();
    GlobalTensor<int16_t> lruGm;
    GlobalTensor<int16_t> hitGm;
    GlobalTensor<uint8_t> hitMaskGm;
    GlobalTensor<int16_t> newLruGm;
    lruGm.SetGlobalBuffer((__gm__ int16_t *)lru,
                          batchSize * lruLength);
    hitGm.SetGlobalBuffer((__gm__ int16_t *)hit, batchSize * k);
    hitMaskGm.SetGlobalBuffer((__gm__ uint8_t *)hitMask, batchSize * k);
    newLruGm.SetGlobalBuffer((__gm__ int16_t *)newLru,
                             batchSize * lruLength);

    int64_t bitmapOffset = 0;
    int64_t maskOffset =
        bitmapOffset + bitmapElements * sizeof(int16_t);
    int64_t hitOffset = maskOffset + maskBytes;
    int64_t lruOffset = hitOffset + hitElements * sizeof(int16_t);
    int64_t candidateOffset = lruOffset + lruElements * sizeof(int16_t);
    int64_t workBytes = candidateOffset + hitElements * sizeof(int16_t);

    TPipe pipe;
    TBuf<TPosition::VECCALC> workBuf;
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(workBytes));
    LocalTensor<uint8_t> work = workBuf.Get<uint8_t>();
    LocalTensor<int16_t> bitmapLocal =
        work[bitmapOffset].ReinterpretCast<int16_t>();
    LocalTensor<uint8_t> maskLocal = work[maskOffset];
    LocalTensor<int16_t> hitLocal =
        work[hitOffset].ReinterpretCast<int16_t>();
    LocalTensor<int16_t> lruLocal =
        work[lruOffset].ReinterpretCast<int16_t>();
    LocalTensor<int16_t> candidateLocal =
        work[candidateOffset].ReinterpretCast<int16_t>();

    int64_t blockNum = GetBlockNum();
    for (int64_t b = GetBlockIdx(); b < batchSize; b += blockNum) {
        Duplicate(bitmapLocal, static_cast<int16_t>(0), bitmapElements);
        CopyInInt16(hitLocal, hitGm, b * k, k);
        CopyInUint8(maskLocal, hitMaskGm, b * k, k);
        CopyInInt16(lruLocal, lruGm, b * lruLength, lruLength);
        WaitMte2ToScalar();
        WaitVectorToScalar();

        for (int64_t i = 0; i < k; ++i) {
            if (maskLocal.GetValue(i) != 0) {
                int32_t id = static_cast<int32_t>(hitLocal.GetValue(i));
                bitmapLocal.SetValue(id, static_cast<int16_t>(1));
            }
        }

        int64_t candidateCount = 0;
        for (int64_t p = lruLength; p > 0 && candidateCount < k; --p) {
            int16_t id = lruLocal.GetValue(p - 1);
            if (bitmapLocal.GetValue(static_cast<int32_t>(id)) == 0) {
                candidateLocal.SetValue(candidateCount, id);
                ++candidateCount;
            }
        }

        int64_t missRank = 0;
        for (int64_t i = 0; i < k; ++i) {
            if (maskLocal.GetValue(i) == 0) {
                int16_t id = candidateLocal.GetValue(missRank);
                ++missRank;
                hitLocal.SetValue(i, id);
                bitmapLocal.SetValue(static_cast<int32_t>(id),
                                     static_cast<int16_t>(1));
            }
        }

        int64_t keepCount = 0;
        for (int64_t p = 0; p < lruLength; ++p) {
            int16_t id = lruLocal.GetValue(p);
            if (bitmapLocal.GetValue(static_cast<int32_t>(id)) == 0) {
                candidateLocal.SetValue(keepCount, id);
                ++keepCount;
            }
        }

        for (int64_t i = 0; i < k; ++i) {
            lruLocal.SetValue(i, hitLocal.GetValue(i));
            lruLocal.SetValue(k + i, candidateLocal.GetValue(i));
        }

        WaitScalarToMte3();
        CopyOutInt16(hitGm, b * k, hitLocal, k);
        CopyOutInt16(newLruGm, b * lruLength, lruLocal, lruLength);
        WaitMte3ToScalar();
    }
}
