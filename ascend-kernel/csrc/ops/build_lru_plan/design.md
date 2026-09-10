# build_lru_plan：hit_mask 输入与 hit 原地回填设计

## 1. 目标与设计结论

本次接口升级在现有 4-Kernel V2 流水上完成两项变更：

1. 新增与 `hit` 同 shape 的 `hit_mask` 输入。原始 `hit[b, i] == -1` 时
   `hit_mask[b, i] == false`，否则为 `true`。
2. 删除 `hit_and_miss` 输出；算子执行结束后，直接把淘汰 ID 写入 `hit` 的 miss
   位置，`hit` 成为填充完成的 hit/miss 序列。

推荐实现仍为纯 **AscendC Vector Kernel**，不涉及 Cube、CATLASS 或 ACLNN 封装。
PyTorch 和 NumPy 均没有 `build_lru_plan` 同名标准接口，因此采用本项目的自定义原地
语义，并在 dispatcher schema 中显式声明 `hit` 可变。

本设计的核心访存优化不是简单地把 `hit_and_miss` 重命名为 `hit`，而是利用
`hit_mask` 在 materialize 阶段现场计算 Tile 内 miss rank，从而删除 `hit_rank`
workspace 及其一次 GM 写、一次 GM 读。`hit_mask` 必须由上游直接提供为连续
`torch.bool`；若为本算子单独启动 `(hit != -1)` Kernel 生成，额外 launch 和 MTE
可能抵消收益。

---

## 2. 算子接口

### 2.1 C++ 函数签名

```cpp
at::Tensor build_lru_plan(
    const at::Tensor &lru,
    at::Tensor &hit,
    const at::Tensor &hit_mask);
```

### 2.2 PyTorch Dispatcher Schema

```cpp
m.def(
    "build_lru_plan(Tensor lru, Tensor(a!) hit, Tensor hit_mask) "
    "-> Tensor new_lru");
```

`Tensor(a!) hit` 表示算子会修改 `hit` 的底层存储。返回值只有 `new_lru`，调用方式为：

```python
new_lru = torch.ops.npu.build_lru_plan(lru, hit, hit_mask)
# 此时 hit 中原来的 -1 已被淘汰 ID 替换
```

### 2.3 参数与约束

| 名称 | 方向 | Shape | dtype | 说明 |
|---|---|---:|---|---|
| `lru` | 输入 | `[B, 2K]` | `int32` | 每行按 MRU 到 LRU 排列 |
| `hit` | 输入/原地输出 | `[B, K]` | `int32` | 输入时 `-1` 表示 miss；输出时 miss 已用淘汰 ID 填充 |
| `hit_mask` | 输入 | `[B, K]` | `bool` | 原始 `hit != -1` 的 0/1 掩码；算子不修改 |
| `new_lru` | 输出 | `[B, 2K]` | `int32` | 更新后的 LRU，且 `new_lru[:, :K] == hit` |

Host 端必须检查：

- 三个 Tensor 均位于同一 NPU device；
- `lru`、`hit` 为 `int32`，`hit_mask` 为 `torch.bool`；
- 三者均为二维，`hit_mask.sizes() == hit.sizes()`；
- `B > 0`、`K > 0`、`lru.size(0) == B`、`lru.size(1) == 2K`；
- `hit` 必须连续。原地语义下不能用 `hit.contiguous()` 的临时副本代替原 Tensor，
  否则调用者看不到回填结果；
- `lru`、`hit_mask` 是只读输入，可保留 `.contiguous()` 兼容路径，但性能路径要求上游
  直接提供连续 Tensor，避免隐藏的格式转换和额外 MTE；
- `hit` 的存储不能与 `lru` 或 `hit_mask` 重叠。materialize Kernel 中存在并行的
  `lru` 读取和 `hit` 写回，重叠会产生核间数据竞争。

性能路径不把数据复制到 CPU 做值域或一致性检查。调用方必须保证：

```text
hit_mask[b, i] == false  <=>  hit[b, i] == -1       # 调用前
hit_mask[b, i] == true   <=>  hit[b, i] != -1
```

每行 `lru` 必须是 `0..2K-1` 的排列，所有有效 hit ID 位于该范围且互不重复。
不满足这些条件时行为未定义。`hit_mask` 描述的是调用前的 `hit`，算子结束后不会随
原地回填而改变。

---

## 3. 计算语义

对每个 batch 行独立执行：

```python
hit_ids = {hit[i] for i in range(K) if hit_mask[i]}
candidates = [id for id in reversed(lru) if id not in hit_ids]

candidate_it = iter(candidates)
for i in range(K):
    if not hit_mask[i]:
        hit[i] = next(candidate_it)       # 原地回填

selected = set(hit)
new_lru = hit + [id for id in lru if id not in selected]
```

输出不再包含 `hit_and_miss`；其旧语义完全由原地更新后的 `hit` 承担。

### 3.1 四 Kernel 数据流

```text
K1 hit_local
  hit -> hit_bitmap + hit_counts
                |
                v
K2 candidate_local
  lru + hit_bitmap -> candidate_values + candidate_counts
                |
                v
K3 materialize
  hit + hit_mask + counts/candidates -> hit(in-place) + new_lru[:K]
  lru + hit_bitmap + counts          -> keep_values + keep_counts
                |
                v
K4 keep_scan_write
  keep_values + keep_counts -> new_lru[K:]
```

普通 AscendC Kernel 内没有跨 AIV 全局 barrier，因此 K1→K2、K2→K3、K3→K4 的
全局生产者/消费者边界仍通过同一 stream 上的顺序 Kernel launch 保证，不能继续直接
融合。

---

## 4. AscendC Kernel 设计与 API 伪代码

### 4.1 K1 `build_lru_plan_hit_local`

任务空间为 `B * NH`。K1 必须读取 `hit` 才能为有效 ID 构造按 ID 索引的 bitmap，
因此这里直接使用 `id == -1` 统计 miss，不额外搬入 `hit_mask`。与旧实现相比，删除
Tile 内 `hit_rank` 的生成和写出。

`hit_bitmap` 由 Host 以零初始化。由于有效 hit ID 在一行内互不重复，每个有效 ID 都有
唯一写者；K1 使用每 ID 一次 4B `DataCopyPad` 的稀疏短写，删除每 Tile 的整行 bitmap
初始化、整行搬出和 float atomic add。

```cpp
DataCopyPad(hitLocal, hitGm[hitOffset], hitCopyParams, noPad);
SetFlag<HardEvent::MTE2_S>(mte2ToS);
WaitFlag<HardEvent::MTE2_S>(mte2ToS);

int32_t missCount = 0;
for (int64_t p = 0; p < validLen; ++p) {
    int32_t id = hitLocal.GetValue(p);
    if (id == -1) {
        ++missCount;
    } else {
        oneLocal.SetValue(p, 1.0f);
    }
}
countLocal.SetValue(0, missCount);

SetFlag<HardEvent::S_MTE3>(sToMte3);
WaitFlag<HardEvent::S_MTE3>(sToMte3);
DataCopyPad(hitCountsGm[countOffset], countLocal, oneInt32Params);
for (int64_t p = 0; p < validLen; ++p) {
    int32_t id = hitLocal.GetValue(p);
    if (id != -1) {
        DataCopyPad(hitBitmapGm[bitmapOffset + id], oneLocal[p], oneInt32Params);
    }
}
```

普通 `DataCopyPad` 不能用一组参数表达任意 ID 的 scatter，因此这里是每个有效 ID 一次
短写，而不是整个 Tile 一次写。若 profiler 显示短 DMA 命令开销占主导，后续备选路径是
按 bitmap 地址范围分配唯一写者，每个任务一次连续写出自己的 range。

### 4.2 K2 `build_lru_plan_candidate_local`

任务空间为 `B * NL`，与现有 V2 相同。每个 LRU Tile 从右向左筛选未命中 ID，按
淘汰顺序紧凑写入自己的固定 slot。

```cpp
DataCopyPad(bitmapLocal, hitBitmapGm[row], bitmapParams, noPad);
DataCopyPad(lruLocal, lruGm[tileOffset], lruParams, noPad);
SetFlag<HardEvent::MTE2_S>(mte2ToS);
WaitFlag<HardEvent::MTE2_S>(mte2ToS);

int32_t count = 0;
for (int64_t p = validLen; p > 0; --p) {
    int32_t id = lruLocal.GetValue(p - 1);
    if (bitmapLocal.GetValue(id) == 0.0f) {
        candidateLocal.SetValue(count++, id);
    }
}

SetFlag<HardEvent::S_MTE3>(sToMte3);
WaitFlag<HardEvent::S_MTE3>(sToMte3);
DataCopyPad(candidateValuesGm[slot], candidateLocal, valueParams);
DataCopyPad(candidateCountsGm[countOffset], countLocal, oneInt32Params);
```

### 4.3 K3 `build_lru_plan_materialize`

任务空间为 `B * (NH + NL)`，包含两类互不重叠的任务。

#### Hit Tile 任务

使用 `bool hit_mask` 现场生成 Tile 内 exclusive miss rank，不再读取 `hit_rank`：

```cpp
DataCopyPad(hitLocal, hitGm[hitOffset], hitParams, noPad);
DataCopyPad(maskLocal, hitMaskGm[maskOffset], maskParams, noPad);
DataCopyPad(hitCountLocal, hitCountsGm[row], hitMetaParams, noPad);
DataCopyPad(candidateCountLocal, candidateCountsGm[row], lruMetaParams, noPad);
DataCopyPad(candidateValueLocal, candidateValuesGm[row], valueParams, noPad);
SetFlag<HardEvent::MTE2_S>(mte2ToS);
WaitFlag<HardEvent::MTE2_S>(mte2ToS);

int32_t missBase = Sum(hitCountLocal[0 : hitTileIndex]);
int32_t localMissRank = 0;
for (int64_t p = 0; p < validLen; ++p) {
    int32_t outputId = hitLocal.GetValue(p);
    if (maskLocal.GetValue(p) == 0) {
        int32_t rank = missBase + localMissRank;
        outputId = LookupCandidateInReverseTileOrder(
            rank, candidateCountLocal, candidateValueLocal);
        ++localMissRank;
    }
    outputLocal.SetValue(p, outputId);
}

SetFlag<HardEvent::S_MTE3>(sToMte3);
WaitFlag<HardEvent::S_MTE3>(sToMte3);
DataCopyPad(hitGm[hitOffset], outputLocal, outputParams);       // 原地回填
DataCopyPad(newLruGm[lruHeadOffset], outputLocal, outputParams);
```

同一个 Hit Tile 只由一个任务写，Tile 间逻辑区间互不重叠。`DataCopyPad` 只向 GM 写
`blockLen` 指定的有效字节，尾 Tile 不会覆盖相邻任务。

#### LRU Tile 任务

与现有 V2 相同。设 `q[p]` 表示该 ID 不在 hit bitmap，`cnt[lt]` 为本 Tile 的候选数：

```text
reverseRank(p) = suffixOffset[lt] + cnt[lt] - prefixInclusive(p)
keep(p)        = q[p] && reverseRank(p) >= rowMissCount
```

它只读取已完成的 `hit_bitmap`、`hit_counts`、`candidate_counts`，不会读取正在原地写回
的 `hit`，因此可以与 Hit Tile 任务在同一 Kernel 中并行。

### 4.4 K4 `build_lru_plan_keep_scan_write`

任务空间为 `B * NL`，与现有 V2 相同。每个 Tile 读取紧凑 `keep_counts`，累加左侧
count 得到 offset，再把自己的 `keep_values` 写到 `new_lru[K + offset]`。

```cpp
DataCopyPad(countLocal, keepCountsGm[row], metaParams, noPad);
SetFlag<HardEvent::MTE2_S>(mte2ToS);
WaitFlag<HardEvent::MTE2_S>(mte2ToS);
int32_t offset = Sum(countLocal[0 : lruTileIndex]);
int32_t count = countLocal.GetValue(lruTileIndex);
DataCopyPad(valuesLocal, keepValuesGm[slot], valueParams, noPad);
SetFlag<HardEvent::MTE2_S>(mte2ToS);
WaitFlag<HardEvent::MTE2_S>(mte2ToS);
SetFlag<HardEvent::S_MTE3>(sToMte3);
WaitFlag<HardEvent::S_MTE3>(sToMte3);
DataCopyPad(newLruGm[k + offset], valuesLocal, valueParams);
```

本算子只支持整数和布尔搬运/比较，不涉及 FP16/BF16，也不需要升精度 Cast 流程。

---

## 5. 两级 Tiling 策略

### 5.1 Tiling 参数结构体

```cpp
struct BuildLruPlanTilingData {
    int64_t batchSize;
    int64_t k;
    int64_t lruLength;        // 2K
    int64_t coreNum;          // 设备 AIV 数

    int64_t hitTileLength;    // TH
    int64_t lruTileLength;    // TL
    int64_t hitTileCount;     // NH
    int64_t lruTileCount;     // NL

    int64_t lruStride;        // bitmap 行 stride，int32/float 元素数
    int64_t hitMetaStride;    // hit_counts 行 stride，int32 元素数
    int64_t lruMetaStride;    // candidate/keep counts 行 stride
    int64_t valueStride;      // candidate/keep values 行 stride

    int64_t scanMode;         // 0=on-the-fly, 1=row scan, 2=hierarchical
    int64_t pipelineDepth;    // 初版固定 1；实测后可选 2
};
```

旧字段 `hitStride` 删除，因为不再存在 `hit_rank`。

### 5.2 Block 级：二维任务切分

```text
L  = 2K
C  = 设备 AIV 数
R  = ceil(C / B)

TH_parallel = clamp(AlignDown(K / R, 8), 8, 256)
TL_parallel = clamp(AlignDown(L / R, 8), 8, 256)

NH = ceil(K / TH)
NL = ceil(L / TL)
```

`8 * sizeof(int32_t) == 32B`，常规 Tile 从 32B 边界开始。每个 Kernel 交错取任务：

```cpp
for (int64_t task = GetBlockIdx(); task < taskCount;
     task += GetBlockNum()) {
    ProcessTask(task);
}
```

Host 设置 `blockDim = max(1, min(taskCount, coreNum))`。对重点场景
`B=1, K=2048, C=40`：

```text
TH = 48, NH = 43
TL = 96, NL = 43
```

因此四个阶段均至少有 40 个有效任务。

### 5.3 UB 级：容量约束与 Buffer 分配

所有 UB buffer 按 32B 对齐。令：

```text
A8(x) = AlignUp(x, 8)          # int32/float 元素对齐
A32(x) = AlignUp(x, 32)        # byte 对齐
TH = hitTileLength
TL = lruTileLength
```

#### K1 UB 分配

| Buffer | dtype | 数量 | 大小（Byte） | 用途 |
|---|---|---:|---:|---|
| `hitLocal` | int32 | 1 | `4 * TH` | hit Tile |
| `oneLocal` | float32 | 1 | `4 * TH` | 稀疏短写的稳定源槽 |
| `countLocal` | int32 | 1 | `32` | 单个 miss count 的对齐短写 |
| **总计** | | | **`8*TH + 32`** | |

Tile buffer coefficient 为 **8 Byte/TH 元素**。相较整行 atomic bitmap 方案，K1 UB
不再随 `lruStride` 增长。

#### K2 UB 分配

| Buffer | dtype | 数量 | 大小（Byte） |
|---|---|---:|---:|
| `bitmapLocal` | float32 | 1 | `4 * lruStride` |
| `lruLocal` | int32 | 1 | `4 * TL` |
| `candidateLocal` | int32 | 1 | `4 * TL` |
| `countLocal` | int32 | 1 | `32` |
| **总计** | | | **`4*lruStride + 8*TL + 32`** |

Tile buffer coefficient 为 **8 Byte/TL 元素**。

#### K3 UB 分配（分支复用）

Hit Tile 与 LRU Tile 分支不会在同一任务内同时执行，行级 resident 区允许复用：

| 区域 | Hit Tile 分支 | LRU Tile 分支 |
|---|---:|---:|
| 行级 resident | `4 * valueStride`（candidate values） | `4 * lruStride`（bitmap） |
| 紧凑 metadata | `4*(hitMetaStride+lruMetaStride)` | 同左 |
| 输入 Tile | `4*TH + A32(TH)` | `4*TL` |
| 输出 Tile | `4*TH` | `4*TL` |
| scalar block | `32` | `32` |

因此：

```text
K3_hit_bytes = 4*valueStride
             + 4*(hitMetaStride+lruMetaStride)
             + 8*TH + A32(TH) + 32

K3_lru_bytes = 4*lruStride
             + 4*(hitMetaStride+lruMetaStride)
             + 8*TL + 32

K3_ub_bytes  = max(K3_hit_bytes, K3_lru_bytes)
```

`hit_mask` 的 Tile buffer coefficient 为 **1 Byte/TH 元素**，但实际 buffer 大小向上
对齐到 32B。两个 int32 输入/输出 Tile 的 coefficient 合计为 **8 Byte/TH 元素**。

#### K4 UB 分配

```text
K4_ub_bytes = 4*lruMetaStride + 4*TL
```

Host 端取四个阶段最大值，并验证：

```text
max(K1_ub_bytes, K2_ub_bytes, K3_ub_bytes, K4_ub_bytes)
    <= deviceUbBytes - 8KB
```

对 `B=1,K=2048,C=40`，有 `TH=48`、`TL=96`、`lruStride=4096`、
`hitMetaStride=lruMetaStride=48`、`valueStride=4128`：

| Kernel | UB 使用量 |
|---|---:|
| K1 | `416 Byte` |
| K2 | `17,184 Byte` |
| K3 | `17,568 Byte` |
| K4 | `576 Byte` |

最大值约占示例 192KB UB 的 `8.94%`；再加 8KB reserve 仍满足约束。实际实现必须读取
运行设备 UB 容量，不能把 192KB 写死。

初版 `pipelineDepth=1`。只有 profiler 显示 MTE stall，且每核稳定获得至少 3 个 Tile
任务时，才把 Tile 输入/输出区扩成双缓冲；行级 bitmap、metadata 和 candidate values
不做双份分配。

---

## 6. GM Workspace 与生命周期

```text
lruStride     = AlignUp(2K, 128)
hitMetaStride = AlignUp(NH, 8)
lruMetaStride = AlignUp(NL, 8)
valueStride   = NL * TL
```

| Workspace | dtype / Shape | 生产者 -> 消费者 |
|---|---|---|
| `hit_bitmap` | float32 `[B, lruStride]` | K1 -> K2/K3 |
| `hit_counts` | int32 `[B, hitMetaStride]` | K1 -> K3 |
| `candidate_counts` | int32 `[B, lruMetaStride]` | K2 -> K3 |
| `candidate_values` | int32 `[B, valueStride]` | K2 -> K3 |
| `keep_counts` | int32 `[B, lruMetaStride]` | K3 -> K4 |
| `keep_values` | int32 `[B, valueStride]` | K3 -> K4 |

总 workspace：

```text
B * 4 * (lruStride + hitMetaStride + 2*lruMetaStride + 2*valueStride) Byte
```

旧 `hit_rank` workspace（`int32 [B, AlignUp(K,128)]`）和 `hit_and_miss` 输出均删除。
`candidate_values` 在 K3 被读取时 `keep_values` 同时被写入，两者不能别名。

不需要额外 system workspace；上述临时 Tensor 由 Host 在 current stream 上分配并由
PyTorch 生命周期管理。

---

## 7. MTE 流量收益分析

只统计受本次接口变更影响、按逻辑有效字节计算的 GM 流量：

| 路径 | 旧 V2 | 新设计 |
|---|---:|---:|
| K1 读取 `hit` | `4BK` | `4BK` |
| K1 写 `hit_rank` | `4BK` | `0` |
| K3 读取 `hit` | `4BK` | `4BK` |
| K3 读取 rank/mask | `4BK` rank | `BK` bool mask |
| K3 写填充结果 | `4BK` 到 `hit_and_miss` | `4BK` 原地写 `hit` |
| K3 写 `new_lru[:K]` | `4BK` | `4BK` |
| **合计** | **`24BK` Byte** | **`17BK` Byte** |

在 `hit_mask` 已由上游产生且为 bool 的前提下，净减少 **`7BK` Byte（约 29.2%）**
的相关逻辑 GM 流量，并减少约 `4*B*AlignUp(K,128)` Byte rank workspace 和 `4BK`
Byte 输出显存。

对 `B=1,K=2048`，表中相关流量由 `49,152 Byte` 降到 `34,816 Byte`，理论上减少
`14,336 Byte`；尾 Tile 和总线事务的实际传输量以 profiler 为准。

必须注意：

- 把 `hit_and_miss` 改为写回 `hit` 只是改变 MTE3 目的地址，单次写出的字节量不变；
- 真正的 MTE 降幅来自删除 `hit_rank` 读写，并以 1B bool mask 替代 4B rank 读取；
- 若 `hit_mask` 使用 int32，相关总流量变为 `20BK`，收益降为 `4BK`；因此接口固定
  使用 bool；
- 若在算子调用前专门生成 mask，应把 mask 生成 Kernel 的读写和 launch 一并计入
  端到端 benchmark。

算子仍是 memory/Scalar-bound：顺序 DMA 与按 ID 随机 bitmap Scalar 访问并存，性能
必须以目标 NPU profiler 的 MTE2/MTE3 stall、Scalar pipe 和端到端 latency 为准。

---

## 8. 同步与并发安全

- K1/K2/K3/K4 在同一 current stream 顺序发射，跨 Kernel 数据依赖由 stream 保证；
- 同一 Kernel 内不同任务只写独占的 Tile 逻辑区间或独占 metadata 元素；
- K1 bitmap 的每个有效 ID 只有一个任务写；并发安全来自地址互斥，`DataCopyPad`
  本身不提供冲突保护；
- GM→UB 后 Scalar 首次消费使用 `MTE2_S`，Scalar→GM 前使用 `S_MTE3`，复用同一
  UB 槽前按真实依赖使用 `MTE3_S`；禁止恢复 helper 内无条件 `PIPE_ALL`；
- K3 必须先完成 `hit` Tile 的 MTE2 搬入和 Scalar 消费，再对相同 GM 区间执行原地
  MTE3 写回；
- K3 的 LRU Tile 分支不读取 `hit`，因此与 Hit Tile 分支的原地写回无数据竞争；
- `hit` 与 `lru`/`hit_mask` 禁止存储重叠，防止跨任务读写竞争；
- `DataCopyPad(Local -> Global)` 仅写 `blockLen` 的有效字节，尾 Tile 之间只要逻辑
  区间不重叠，就不会因 32B 补齐而互相覆盖。

---

## 9. Host 与 Kernel 实施清单

### 9.1 Host 端

- [ ] `ops.h` 返回类型改为 `at::Tensor`，`hit` 改为 `at::Tensor &`，新增
      `const at::Tensor &hit_mask`；
- [ ] dispatcher schema 改为
      `build_lru_plan(Tensor lru, Tensor(a!) hit, Tensor hit_mask) -> Tensor new_lru`；
- [ ] 增加 hit_mask device/dtype/rank/shape/contiguous 检查；
- [ ] 拒绝非连续 hit，检查 hit 与只读输入不重叠；
- [ ] 删除 `hitContiguous` 临时副本、`hitAndMiss` 和 `hitRank` 分配；
- [ ] 删除 `hitStride` 的计算、UB 预算和 Kernel 参数；
- [ ] K1 launch 删除 `hitRank` 参数；
- [ ] K3 launch 新增 `hitMask`，删除 `hitRank` 和 `hitAndMiss` 参数；
- [ ] 返回 `newLru`，不再返回 tuple。

### 9.2 Kernel 端

- [ ] K1 删除 `hitRankGm`、`rankLocal` 和 rank 写出，只统计 Tile miss count；
- [ ] K3 增加 `GlobalTensor<uint8_t> hitMaskGm` 和 32B 对齐的 mask Tile buffer；
- [ ] K3 Hit Tile 以 `localMissRank` 现场计算 `missBase + localMissRank`；
- [ ] K3 把填充 Tile 同时写到 `hitGm` 与 `newLruGm`；
- [ ] 删除所有 `hitAndMissGm` 代码；
- [ ] 根据分支复用 K3 resident UB 区，按第 5.3 节公式重新校验 UB；
- [ ] 保持 MTE2→S、S→MTE3、MTE3→UB 复用的窄事件依赖。

---

## 10. 测试与验收

### 10.1 CPU Reference

Reference 必须克隆调用前的 hit，并显式验证原地效果：

```python
hit_mask = hit.ne(-1).contiguous()
hit_before = hit.clone()
expected_lru, expected_filled = reference(lru, hit_before, hit_mask)

actual_lru = torch.ops.npu.build_lru_plan(lru_npu, hit_npu, hit_mask_npu)

assert_close(actual_lru.cpu(), expected_lru)
assert_close(hit_npu.cpu(), expected_filled)
assert_close(actual_lru[:, :K].cpu(), hit_npu.cpu())
assert_equal(hit_mask_npu.cpu(), hit_mask)     # mask 不变
assert not torch.any(hit_npu == -1)
```

### 10.2 正确性矩阵

| 维度 | Case |
|---|---|
| K | `1,3,7,8,9,47,48,49,95,96,97,127,128,129,257,2048` |
| hit 分布 | 全 hit、全 miss、0/1 个 miss、约 50%、约 80%、miss 集中在首/尾 |
| batch | `B=1,2,3,40,>40` |
| Tile 边界 | hit_mask 尾 Tile 为 1B/31B/32B 附近；hit/lru 尾 Tile 为 4B/28B/32B 附近 |
| 接口错误 | mask shape/dtype/device 错误、非连续 hit、存储重叠 |
| 语义 | hit 原地更新、mask 保持不变、仅返回 new_lru |
| 并发写 | 相邻 Tile 写同一 32B 中不同逻辑 byte，重复至少 10k 次 |

生产路径不检查 mask 与 hit 的逐元素一致性；CPU 模拟测试必须增加不一致 mask 的
debug/reference case，确保文档中的未定义行为边界明确。

### 10.3 性能验收

对 `B=1,K=2048` 和实际业务 batch 分别报告：

- 端到端 p50/p90 latency 与异步 stream 平均耗时；
- 4 次 Kernel launch 是否保持不变；
- `hit_rank` GM 读写是否从 profiler 中消失；
- K3 新增 bool mask MTE2 流量是否约为 `BK` Byte；
- MTE2/MTE3 stall、Scalar pipe 占比、AIV 活跃数；
- 包含和不包含上游 mask 生成成本的两组 benchmark。

验收标准是新接口结果逐元素正确、原地 alias 语义正确，并且包含真实上游流程后的
端到端性能不劣于旧 V2；不预先承诺仅由理论字节数推导出的固定加速百分比。

---

## 11. 文件改动范围与下一步

- `csrc/ops/build_lru_plan/op_host/build_lru_plan.cpp`
- `csrc/ops/build_lru_plan/op_kernel/build_lru_plan.cpp`
- `csrc/ops.h`
- `csrc/register.cpp`
- `tests/test_build_lru_plan.py`
- `tests/benchmark_build_lru_plan.py`

设计完成后使用 `ascendc-operator-code-gen` 实现接口和 Kernel，再使用
`ascendc-operator-compile-debug` 编译、安装并在 NPU 上完成精度与 profiler 验证。
