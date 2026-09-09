# build_lru_plan 全 AIV 并行与四 Kernel 融合设计

## 1. 目标与结论

`build_lru_plan` 对每个 batch 的 LRU 队列执行一次批量更新。优化前实现采用 8 个
AscendC Vector Kernel；当前 V2 实现依据第 10--17 节融合为 4 个 Kernel。两版都将
按行串行的 bitmap、scan、fill 和 write 工作改为二维 `(batch, tile)` 任务。Host
根据运行设备的 AIV 数动态选择 Tile，使逻辑数据至少包含同等数量的 32B 数据块时，
每个阶段都有不少于 AIV 数量的有效任务。

对重点场景 `B=1, K=2048, AIV=40`：

```text
TH = 48,  NH = ceil(2048 / 48) = 43
TL = 96,  NL = ceil(4096 / 96) = 43
```

下表为优化前 8-Kernel 基线：

| Kernel | 有效任务数 | blockDim | 活跃 AIV |
|---|---:|---:|---:|
| K1 hit local | 43 | 40 | 40 |
| K2 hit scan | 43 | 40 | 40 |
| K3 candidate local | 43 | 40 | 40 |
| K4 candidate scan | 43 | 40 | 40 |
| K5 fill | 43 | 40 | 40 |
| K6 keep local | 43 | 40 | 40 |
| K7 keep scan | 43 | 40 | 40 |
| K8 write | 86 | 40 | 40 |

这里的“占满”指每个 AIV 都获得有效 Tile 任务，不代表执行单元利用率必然为 100%。
该算子仍有大量 Scalar 索引访问和 8 次 Kernel launch，最终性能需要在目标 NPU 上
通过 profiler 验证。

## 2. 接口与输入约束

```cpp
std::tuple<at::Tensor, at::Tensor> build_lru_plan(
    const at::Tensor &lru,
    const at::Tensor &hit);
```

```text
build_lru_plan(Tensor lru, Tensor hit) ->
    (Tensor new_lru, Tensor hit_and_miss)
```

| 名称 | 方向 | Shape | dtype | 说明 |
|---|---|---:|---|---|
| `lru` | 输入 | `[B, 2K]` | `int32` | 每行按 MRU 到 LRU 排列 |
| `hit` | 输入 | `[B, K]` | `int32` | `-1` 表示 miss，其余为命中 ID |
| `new_lru` | 输出 | `[B, 2K]` | `int32` | 更新后的 LRU |
| `hit_and_miss` | 输出 | `[B, K]` | `int32` | 用淘汰 ID 填充 miss 后的序列 |

Host 检查设备、dtype、二维 shape、batch、`lru.size(1) == 2K`、正数维度和
动态 UB 容量。非连续输入先执行 `.contiguous()`。

性能路径不复制输入到 CPU 做值域检查。调用方必须保证每行 `lru` 是
`0..2K-1` 的排列，`hit` 中有效 ID 位于该范围且互不重复；否则行为未定义。

## 3. 算子语义

对每一行：

1. 保留 `hit` 中非 `-1` 的 ID。
2. 从原 `lru` 的 LRU 端向 MRU 端选择未命中的 ID。
3. 按 miss 从左到右的顺序填入这些 ID，得到 `hit_and_miss`。
4. 把 `hit_and_miss` 放入 `new_lru` 的前 `K` 项。
5. 原 `lru` 中未被本轮选中的 ID 稳定压缩到后 `K` 项。

## 4. 动态二维 Tiling

```text
L  = 2K
C  = 设备 AIV 数
R  = ceil(C / B)                    # 每行目标 Tile 数
TH = clamp(AlignDown(K / R, 8), 8, 256)
TL = clamp(AlignDown(L / R, 8), 8, 256)
NH = ceil(K / TH)
NL = ceil(L / TL)
```

Tile 长度按 8 个 `int32` 对齐，保证常规 Tile 从 32B 边界开始。尾 Tile 的非对齐
逻辑长度由 `DataCopyPad` 处理。每个 Kernel 使用交错取任务：

```cpp
for (int64_t task = GetBlockIdx(); task < taskCount;
     task += GetBlockNum()) {
    ProcessTask(task);
}
```

Host 动态设置 `blockDim = min(taskCount, coreNum)`，最少为 1。若 shape 本身少于
设备 AIV 数量的 32B 数据块，无法产生足够的有效并行任务，这是小 shape 的物理
并行度上限。

## 5. 优化前八阶段 Kernel

### K1 `build_lru_plan_hit_local`

任务空间为 `B * NH`。每个 hit Tile 在 UB 中创建一份私有 float bitmap，计算 Tile
内 miss exclusive rank，并把 bitmap 通过 float atomic add 累加到预先清零的 GM
`hit_bitmap`。每个 Tile 的 miss count 写入独占 32B 槽。

### K2 `build_lru_plan_hit_scan`

任务空间为 `B * NH`。每个 hit Tile 独立读取该行全部 Tile count，只累加位于自己
之前的 count，并把 offset 加到本 Tile 的 miss rank。最后一个 Tile 写出整行
`miss_count`。该方式用少量重复读取消除原先“一行一个核”的串行扫描。

### K3 `build_lru_plan_candidate_local`

任务空间为 `B * NL`。每个 LRU Tile 读取 hit bitmap，从 Tile 右端向左筛选未命中
ID，并把候选 ID 直接紧凑写入本 Tile 的固定长度 slot；不再生成逐元素 lru rank。

### K4 `build_lru_plan_candidate_scan`

任务空间为 `B * NL`。每个 LRU Tile 读取全部 count，独立累加其右侧 Tile 的 count，
得到反向 exclusive offset。计数槽 lane 0 保存 offset，lane 1 保留原 count；并行
核心只写自己的 32B 槽，因此不存在覆盖竞争。

### K5 `build_lru_plan_fill`

任务空间为 `B * NH`。每个 hit Tile 读取候选值和 `(offset, count)`，填充本 Tile 的
miss，写出对应 `hit_and_miss` 区间。同时构造私有 selected bitmap，并通过 float
atomic add 累加到预先清零的 GM `selected_bitmap`。

### K6 `build_lru_plan_keep_local`

任务空间为 `B * NL`。每个 LRU Tile 读取 selected bitmap，把未被本轮选择的 ID
稳定压缩到本 Tile 的固定长度 value slot，并写 count。

### K7 `build_lru_plan_keep_scan`

任务空间为 `B * NL`。每个 Tile 独立累加左侧 Tile 的 count，生成最终尾部的正向
exclusive offset。lane 0 保存 offset，lane 1 保留 count。

### K8 `build_lru_plan_write`

任务空间为 `B * (NH + NL)`。前 `NH` 类任务并行复制 `hit_and_miss`，后 `NL` 类
任务把已紧凑的 keep value 写到 `new_lru[K + offset]`。逻辑区间互不重叠，非 32B
对齐的头尾由 `DataCopyPad` 精确处理。

## 6. 优化前 Workspace 与复用

```text
Kp = AlignUp(K, 128)
Lp = AlignUp(2K, 128)
Tp = 8 * max(NL, NH)
Vp = NL * TL
Mp = 8
```

| Workspace | dtype / Shape | 生命周期 |
|---|---|---|
| `hit_bitmap` | float32 `[B, Lp]` | K1 原子构建，K3 读取 |
| `selected_bitmap` | float32 `[B, Lp]` | K5 原子构建，K6 读取 |
| `hit_rank` | int32 `[B, Kp]` | K1 局部 rank，K2 全局 rank，K5 读取 |
| `tile_counts` | int32 `[B, Tp]` | K1/K2、K3-K5、K6-K8 分阶段复用 |
| `tile_values` | int32 `[B, Vp]` | K3-K5 candidate，K6-K8 keep 复用 |
| `miss_count` | int32 `[B, Mp]` | K2 写，K5 读 |

bitmap 使用 float32 是因为目标 A2 路径支持 float atomic add。bitmap 只判断
`0.0f` 和非零值；在输入有效 ID 互不重复的约束下，每个位置最终为 0 或 1。

每个 Tile 计数占一个独立 32B slot：lane 0 存 count/offset，lane 1 始终保留
count。K4/K7 的每个核心只覆盖自己的 slot，其他核心在扫描期间始终读取 lane 1，
因此不会读取到被并发改写的 lane 0。

## 7. 优化前 UB 预算

各阶段的主要 UB 元素数：

```text
K1: Lp(float) + 2*TH + 8
K2: TH + Tp + Mp
K3/K6: Lp(float) + 2*TL + 8
K4/K7: Tp + 8
K5: Lp(float) + 3*TH + Vp + Tp + Mp
K8: max(TL, TH) + 8
```

K5 为最坏阶段。Host 使用 K5 公式检查动态 UB，并预留 8KB。bitmap 原子 DMA 的
UB 源单独放在 `TPosition::VECOUT`。

## 8. 优化前同步与并发安全

- 8 个 Kernel 在同一 current stream 顺序发射，阶段间依赖由 stream 顺序保证。
- GM 与 UB 之间只使用 `DataCopyPad`。
- Kernel 不直接对 GM 使用 Scalar `GetValue/SetValue`；所有标量索引访问都在 UB。
- K1/K5 使用 `SetAtomicAdd<float>()`，DMA 完成后立即恢复 `SetAtomicNone()`。
- Kernel 入口显式调用 `SetAtomicNone()`，避免继承异常原子状态。
- 并行 scan 的 count 保存在独占 32B 槽，写本 Tile、读其他 Tile 的 lane 1。
- 最终写回的每个任务只写自己的逻辑输出区间。

## 9. 正确性与性能验证

`tests/test_build_lru_plan.py` 包含直接 CPU reference 和本设计的分阶段 CPU 模拟，
覆盖 `B=1,K=2048`、小 shape、Tile 边界、全 miss、无 miss、非连续输入与错误 dtype。
所有输出必须逐元素一致，并满足：

```text
new_lru[:, :K] == hit_and_miss
```

性能脚本 `tests/benchmark_build_lru_plan.py` 默认使用 `B=1,K=2048`、约 80% hit，
先做精度检查，再分别报告异步 stream 平均耗时和逐次同步 latency。目标设备上还应
用 profiler 确认每阶段的 AIV 活跃数、Scalar pipe 占比、GM 带宽和 atomic 开销。

源方案：`二维分Tile并行LRU算子设计方案.pdf`。

## 10. 现实现优化审计（2026-09）

本节区分“优化前 8-Kernel 实现保证的性质”和“当前 V2 实现采用的优化”。V2 代码
以本节的依赖分析为准，不能简单删除全部同步或在一个 Kernel 内假设存在跨 AIV 的
全局 barrier。

### 10.1 `DataCopyPad` 非对齐写与同步边界

`DataCopyPad(Local -> Global)` 的 `blockLen` 单位是 Byte。当 `blockLen` 不是 32B
整数倍时，硬件在 UB 侧补 dummy byte，但写入 GM 时丢弃 dummy byte。因此两个 AIV
分别写同一个 32B 区域中的不相交逻辑 byte 区间，不会像普通对齐 `DataCopy` 那样把
补齐部分写入 GM。由此可得到两个直接优化：

1. `tileCounts` 不必为每个 Tile 固定占 8 个 `int32`；可以拆成紧凑的
   `counts[N]` 和 `offsets[N]`，每项通过 4B `DataCopyPad` 精确写入。
2. K8 中相邻 Tile 的非 32B 对齐输出只要逻辑区间严格不重叠，就无需因为共享一个
   32B 数据块而使用原子操作或核间同步。

但是，`LoadInt32`/`StoreInt32` 中的同步属于**单核流水依赖**，不负责核间互斥，
也不会把 40 个 AIV 彼此串行化：

| 依赖 | 当前位置 | 是否可直接删除 | 推荐替换 |
|---|---|---|---|
| MTE2 写 UB -> Scalar `GetValue` 读 UB | `LoadInt32` 后 | 否 | `MTE2_S` 窄事件；若工程明确开启自动同步，则由编译框架生成 |
| Scalar `SetValue` 写 UB -> MTE3 读 UB | `StoreInt32` 前 | 否 | `S_MTE3` 窄事件；若自动同步已开启则不手工重复插入 |
| MTE3 读 UB -> 下一任务复用/覆盖同一 UB | `StoreInt32` 后 | 视复用关系而定 | 双缓冲时用 `MTE3_S`/队列状态保护对应 buffer，末次无需阻塞下一块独立 buffer |
| 多次独立 MTE2 搬入 | 每次 `Load*` 后 | 可合并 | 先连续发射多个 `DataCopyPad`，在首次 Scalar/Vector 消费前等待最后一个 MTE2 事件 |
| 多次独立 MTE3 搬出 | 每次 `Store*` 前后 | 可合并 | Scalar 写完全部输出后做一次 `S_MTE3`，随后连续发射互不重叠的搬出 |
| 原子模式切换 | `SetAtomicAdd` 周围 | 不按普通 Store 处理 | 先确认当前 CANN 对 atomic 状态的采样时机；保留原子搬出完成/源 UB 复用保护 |

因此 V2 不再使用在 helper 内无条件执行的 `PipeBarrier<PIPE_ALL>()`。helper 只负责发射
DMA，调用者根据真实 producer/consumer 放置 `MTE2_S`、`S_MTE3`、`V_S`、
`V_MTE3` 等窄事件。事件 ID 必须由 `FetchEventID/AllocEventID` 获取并成对使用。

建议的伪代码如下：

```cpp
// 1. 连续发射彼此独立的 GM -> UB 搬入。
DataCopyPad(hitLocal, hitGm[hitOffset], hitParams, noPad);
DataCopyPad(rankLocal, rankGm[rankOffset], rankParams, noPad);
DataCopyPad(countLocal, countGm[rowOffset], countParams, noPad);

// 2. Scalar 首次读取这些 UB 前只做一次窄同步。
SetFlag<HardEvent::MTE2_S>(mte2ToS);
WaitFlag<HardEvent::MTE2_S>(mte2ToS);
ScalarCompute(...);

// 3. Scalar 写完多个 LocalTensor 后，只做一次 S -> MTE3 同步。
SetFlag<HardEvent::S_MTE3>(sToMte3);
WaitFlag<HardEvent::S_MTE3>(sToMte3);
DataCopyPad(out0Gm[out0Offset], out0Local, out0Params);
DataCopyPad(out1Gm[out1Offset], out1Local, out1Params);
```

### 10.2 当前同步中可以优先删除或收窄的位置

按当前代码逐项审计：

- `LoadInt32/LoadFloat` 的尾部 `PIPE_ALL` 不能无条件删除，但可从每次一次改为一组
  load 后一次 `MTE2_S`。
- `StoreInt32` 前置 `PIPE_ALL` 需要保留 producer -> MTE3 的依赖语义，但应收窄为
  `S_MTE3` 或 `V_MTE3`；尾部 `PIPE_ALL` 只在同一 UB 即将复用时等待。
- K1 的 `Duplicate(bitmap/count) -> Load(hit)` 之间没有数据依赖，可并行发射；首次
  Scalar 同时读取 `hitLocal` 和写入已完成初始化的 bitmap 前再分别满足 MTE2/Vector
  依赖。
- K5 的五次 `LoadInt32` 当前被五个 `PIPE_ALL` 完全串行化，是最应优先改为批量
  MTE2 发射的位置。
- K8 的 `Load -> Store` 仍有 MTE2 -> MTE3 的真实依赖；不能因为 GM 输出不重叠就
  删除该核内依赖，但可以用两个 Tile buffer 做 ping-pong。

## 11. Kernel 融合结论

### 11.1 不能直接融合的边界

当前设备编程模型不能在普通 AscendC Kernel 内对所有 AIV 做全局 barrier。以下阶段
要求前一阶段的所有 Tile 都完成，因而不能只把函数体拼进同一 Kernel：

```text
hit bitmap 全部完成 -> candidate 过滤
candidate counts/values 全部完成 -> miss 填充
selected 集合全部完成 -> keep 过滤（当前算法）
keep counts 全部完成 -> tail offset/write
```

若强行融合，只能退回“每行一个 AIV”串行完成，或实现跨核自旋同步；两者分别损失
并行度或带来可移植性、死锁和调度风险，不作为默认设计。

### 11.2 推荐的 4-Kernel V2

可以通过改写数据依赖而不是添加跨核 barrier，将 8 次 launch 降为 4 次：

| V2 Kernel | 原阶段 | 任务空间 | 产出 |
|---|---|---:|---|
| V2-K1 `hit_local` | K1 | `B * NH` | `hit_bitmap`、Tile 内 `hit_rank`、紧凑 `hit_counts` |
| V2-K2 `candidate_local` | K3 | `B * NL` | 正向紧凑 `candidate_values`、紧凑 `candidate_counts` |
| V2-K3 `materialize` | K2+K4+K5+K6 的逻辑融合 | `B * (NH + NL)` | `hit_and_miss`、`new_lru[:K]`、`keep_values`、`keep_counts` |
| V2-K4 `keep_scan_write` | K7+K8 tail | `B * NL` | `new_lru[K:]` |

V2-K3 不是在 Kernel 内等待其他任务，而是在进入前依赖 V2-K1、V2-K2 已由 stream
顺序完成。它的两类任务均只读这两个已完成阶段的元数据：

- Hit Tile 任务：扫描紧凑 `hit_counts` 得到自己的 miss base，扫描
  `candidate_counts` 定位淘汰候选，直接同时写 `hit_and_miss` 与 `new_lru[:K]`。
- LRU Tile 任务：不再依赖 `selected_bitmap`。一个未命中 ID 是否保留，可以由它在
  “从 LRU 端开始的全局候选 rank”与整行 `miss_count` 的比较直接判断。

V2-K3 的 `candidate_values` 与 `keep_values` 必须是两个不同的 GM workspace：同一
Kernel 中 Hit Tile 仍在读取前者，而 LRU Tile 正在写后者，不能沿用当前实现跨 Kernel
复用同一个 `tile_values` 的方式。V2-K3 完成后才允许释放/复用 `candidate_values`。

设 LRU Tile `lt` 内非 hit 指示为 `q[p]`，正向 inclusive prefix 为
`prefix[p] = sum(q[0..p])`，Tile 非 hit 数为 `cnt[lt]`，右侧 Tile 非 hit 总数为
`suffixOffset[lt]`，则：

```text
reverseRank(p) = suffixOffset[lt] + cnt[lt] - prefix[p]   (q[p] == 1)
keep(p)        = q[p] == 1 && reverseRank(p) >= missCount
```

该等式直接消除 `selected_bitmap` 及其一次清零、一次 atomic 构建、一次全量读取，
同时消除原 K5 -> K6 的阶段边界。V2-K4 对每个 Tile 读取紧凑 `keep_counts`，计算
左侧 count 之和后立即写自己的 keep 区间，因此融合原 K7 和 K8 tail。

### 11.3 5-Kernel 可扩展变体

当 `NH/NL` 很大，V2-K3/V2-K4 每个任务独立累加 counts 会形成
`O(NH^2 + NL^2)` 的重复标量工作。此时增加一个每行任务的 `meta_scan`：

```text
V2-K1 hit_local
V2-K2 candidate_local
V2-K3 meta_scan: 一次生成 hit prefix、candidate suffix、miss_count
V2-K4 materialize: O(1) 读取自己的 offset
V2-K5 keep_scan_write，或根据 NL 再拆成 scan + parallel write
```

Host 根据 Tile 数选择：目标 `B=1,K=2048,NH=NL=43` 先测试 4-Kernel 版本；当
`max(NH,NL)` 超过经验阈值（初始建议 128，最终由 benchmark 决定）切换 5-Kernel
或分层 scan 版本。

## 12. Scan 优化设计

### 12.1 当前瓶颈

当前 K2/K4/K7 每个 Tile 都加载 `tileCountStride = 8 * max(NH,NL)` 个 int32，
然后只使用每个 32B slot 的 1--2 个 lane。以 `NH=NL=43` 为例，每个任务加载
344 个 int32，而有效 count 只有 43 个，且总标量累加次数随 Tile 数平方增长。

第一步应先把元数据拆成紧凑数组：

```text
hit_counts      [B, AlignUp(NH, 8)]
hit_offsets     [B, AlignUp(NH, 8)]       # 仅 5-Kernel 变体需要
candidate_counts[B, AlignUp(NL, 8)]
candidate_offsets[B, AlignUp(NL, 8)]      # 仅 5-Kernel 变体需要
keep_counts     [B, AlignUp(NL, 8)]
```

生产者用 4B `DataCopyPad` 写自己的元素；消费者可以一次对齐搬入整行。相较当前
每 Tile 8 个 int32 的 slot，count workspace 和 scan 搬入量约缩小 8 倍。

### 12.2 三档 scan 路径

| Tile 数 | 推荐路径 | 复杂度 | 说明 |
|---:|---|---:|---|
| `N <= 64/128` | 下游任务直接累加紧凑 counts | `O(N^2)`，但常数很小 | 少一次 launch，适合当前 43 Tile 目标 |
| 中等 N | 每行一个 `meta_scan`，UB 内一次前缀/后缀和 | `O(N)` | offset 写回后供所有 Tile 读取 |
| 大 N | 两级/多级 scan：chunk local scan -> chunk sum scan -> prefix add | `O(N)` | 保持多 AIV 并行且可扩展 |

若目标 CANN 与产品组合的 `CumSum<int32_t>` 可用，可对齐成 `[1, AlignUp(N,8)]`
后调用高阶 API；但 A2/A3 的默认实现是 line-by-line，Sklansky 并行算法并非所有产品
支持，因此必须与几十次 Scalar `GetValue/Add/SetValue` 实测比较，不能仅凭 API 名称
认定更快。若高阶 API 不支持当前 int32/产品组合，则保留手写 UB scan。

Tile 内 hit rank 可以使用下面的向量化候选路径；对于 `TH=48`，仍需与当前标量循环
做 A/B benchmark，因为高阶 API 的固定开销可能占主导：

```cpp
CompareScalar(missMask, hitLocal, -1, CMPMODE::EQ, alignedValidLen);
Select(miss01, missMask, oneLocal, 0, SELMODE::VSEL_TENSOR_SCALAR_MODE,
       alignedValidLen);
CumSum(prefixInclusive, lastValue, miss01, {1, alignedValidLen});
Adds(prefixExclusive, prefixInclusive, -1, alignedValidLen);
Select(hitRankLocal, missMask, prefixExclusive, -1,
       SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedValidLen);
```

## 13. Scalar 热点向量化

scan 之外，当前 K3/K6 的随机 bitmap `GetValue` 与紧凑 `SetValue` 是主要 Scalar
热点。bitmap 在目标场景为 16KB，小于 Gather 的 32KB 源约束，可尝试：

```cpp
// lruLocal 为 int32 ID；bitmapLocal 常驻 UB。
Muls(byteOffsets, lruLocal, static_cast<int32_t>(sizeof(float)), validLen);
Gather(flags, bitmapLocal, byteOffsets.ReinterpretCast<uint32_t>(), 0, validLen);
CompareScalar(selectMask, flags, 0.0f, CMPMODE::EQ, alignedValidLen);
GatherMask(valuesLocal, lruLocal, selectMask.ReinterpretCast<uint32_t>(),
           true, alignedValidLen, gatherParams, selectedCount);
```

`candidate_values` 建议按 Tile 内正向顺序紧凑存储；填 miss 时按
`count - 1 - localReverseRank` 读取，即可保持从 LRU 端向前选择的语义，无需先反转
整个 Tile。上述 API 在目标 SoC/CANN 上的具体 overload、mask layout 和临时空间必须
在 code-gen 阶段以安装头文件为准。

## 14. Kernel 内多级流水

当前所有阶段使用单份 `TBuf`，并在每个 helper 内 `PIPE_ALL`，实际执行近似：

```text
Load(task i) -> Scalar/Vector(task i) -> Store(task i) -> Load(task i+1)
```

V2 将 UB 分成“行级常驻元数据”和“Tile 级 ping-pong buffer”：

| Buffer | 大小 | 数量 | 用途 |
|---|---:|---:|---|
| row bitmap | `Lp * 4` | 1 | 本任务重复随机读取，不做双缓冲 |
| compact counts/offsets | `AlignUp(N,8) * 4` | 1 | 行级元数据 |
| input tile | `max(TH,TL) * 4` | 2 | MTE2/Scalar 或 Vector ping-pong |
| index/flag/mask temp | 依 API 临时空间计算 | 1--2 | Gather/Compare/GatherMask |
| output tile | `max(TH,TL) * 4` | 2 | Scalar/Vector 与 MTE3 ping-pong |
| scalar block | 32B | 2 | count 等短写的独立 MTE3 源 |

稳态流水目标：

```text
MTE2 : Load task i+1 ---------------- Load task i+2
S/V  :        Compute task i ---------------- Compute task i+1
MTE3 :                 Store task i-1 ---------------- Store task i
```

实现规则：

1. 使用 `TQue<..., 2>` 或两组显式 LocalTensor，并为每个槽维护对应事件。
2. 只对 Tile buffer 双缓冲，不复制整行 bitmap/value workspace，否则 UB 预算会翻倍。
3. 当前 `B=1,K=2048` 时每核通常只有 1--2 个任务，double buffer 收益可能有限；
   大 K 或较少 blockDim 时收益更明显，Host 应保留单缓冲小任务路径。
4. V2-K3 先批量加载 hit/rank/count/value 元数据，再开始 Scalar；K4 在搬出 Tile i 时
   可预取 Tile i+1 的 keep values/count。

## 15. V2 Tiling 与 UB 预算

V2 沿用动态二维 Tile 的基本公式，但 Tile 上限不再固定只看并行度，应同时考虑
Gather/GatherMask 临时空间和 double buffer：

```text
R          = ceil(C / B)
TH_parallel= clamp(AlignDown(K  / R, 8), 8, 256)
TL_parallel= clamp(AlignDown(2K / R, 8), 8, 256)
TH_ub      = AlignDown((UB_available - rowResidentBytes) / hitBufferCoeff, 8)
TL_ub      = AlignDown((UB_available - rowResidentBytes) / lruBufferCoeff, 8)
TH         = min(TH_parallel, TH_ub)
TL         = min(TL_parallel, TL_ub)
```

V2 推荐 Host 侧结构化参数：

```cpp
struct BuildLruPlanTilingData {
    int64_t batchSize;
    int64_t k;
    int64_t lruLength;
    int64_t coreNum;
    int64_t hitTileLength;
    int64_t lruTileLength;
    int64_t hitTileCount;
    int64_t lruTileCount;
    int64_t bitmapStride;
    int64_t hitMetaStride;
    int64_t lruMetaStride;
    int64_t valueStride;
    int64_t scanMode;       // 0=on-the-fly, 1=row scan, 2=hierarchical
    int64_t pipelineDepth;  // 1 or 2
};
```

V2 的最大 UB 阶段预计为 `materialize`。精确系数依最终 Gather/CumSum overload 的
临时空间查询结果确定；code-gen 前必须补齐下表中的实测/头文件计算值：

| 项 | 单缓冲 | 双缓冲 |
|---|---:|---:|
| bitmap resident | `bitmapStride * 4` | 同左 |
| hit/rank tile | `2 * TH * 4` | `4 * TH * 4` |
| lru/value tile | `2 * TL * 4` | `4 * TL * 4` |
| compact metadata | `(2*hitMetaStride + 2*lruMetaStride) * 4` | 同左 |
| mask | `AlignUp(max(TH,TL),256) / 8`，再按 32B 对齐 | API/槽数决定 |
| Gather/CumSum tmp | `Get*TmpSize` 或安装头文件规定 | 不与另一槽无条件重复 |

约束为总计 `<= deviceUbBytes - 8KB reserve`。本算子仅支持 int32，不涉及
FP16/BF16 升精度流程。

V2 GM workspace 至少包含 `hit_bitmap`、`hit_rank`、`hit_counts`、
`candidate_counts`、`candidate_values`、`keep_counts`、`keep_values`。其中两个 value
workspace 在 V2-K3 内存在读写重叠生命周期，不能别名；删除的 `selected_bitmap`、
`miss_count` 和 32B-per-Tile `tile_counts` 可抵消一部分新增空间。

## 16. 实施优先级与验收

按风险和预期收益排序：

1. **P0：同步收窄**。拆掉 helper 内无条件 `PIPE_ALL`，按依赖批量发射 DMA，并用
   窄事件/自动同步保证正确性。
2. **P0：紧凑 metadata**。将 32B count slot 改为独立 `counts/offsets` 数组；增加
   同一 32B 内相邻 4B 多核写压力测试。
3. **P1：4-Kernel 融合**。先实现数学等价的 `reverseRank >= missCount` keep 判定，
   删除 selected bitmap，再融合 scan/write。
4. **P1：Scalar 向量化**。优先评估 bitmap Gather + Compare + GatherMask；其次评估
   Tile 内 CumSum。
5. **P2：双缓冲**。只在 profiler 显示 MTE stall 且每核有至少 3 个 Tile 时开启。
6. **P2：分层 scan**。仅在大 K 导致 Tile 数明显超过阈值时启用。

每一步都必须单独保留开关并测量，不能把四类改动一次合入后只比较最终结果。
验收矩阵：

| 维度 | Case |
|---|---|
| 正确性 | `K=1,3,7,8,9,47,48,49,95,96,97,127,128,129,257,2048` |
| hit 分布 | 全 hit、全 miss、0/1 个 miss、约 50%、约 80%、miss 集中在首/尾 |
| 并发写 | 相邻 Tile 分别写同一 32B 的不同 4B/8B 区间，重复至少 10k 次 |
| batch | `B=1,2,3,40,>40` |
| 性能 | kernel launch 数、AIV occupancy、MTE2/MTE3 stall、Scalar pipe、atomic 带宽、端到端 p50/p90 |
| 回归 | 非连续输入、错误 dtype/shape、随机种子批量对拍 |

性能目标不预先写死百分比；至少应满足 V2 对目标 `B=1,K=2048` 的端到端 p50 不劣于
当前 8-Kernel 版本，并由逐项 ablation 证明每个默认开启的优化有稳定正收益。

## 17. V2 设计结论

四个优化判断的最终结论如下：

1. `DataCopyPad` 可以保证 GM 只接收逻辑有效 byte，因而可以安全压缩 32B slot；
   但这与单核流水同步是两个问题，不能删除真实的 MTE2/MTE3 与 Scalar/Vector 依赖。
2. Kernel 可以融合，推荐通过消除 `selected_bitmap` 依赖降到 4 个；普通 Kernel 内
   没有跨 AIV barrier，不能直接把有全局生产者/消费者关系的阶段拼接。
3. scan 应先压缩元数据和减少 launch，再按 Tile 数选择 on-the-fly、单行 O(N) 或
   分层 O(N) scan；`CumSum` 需要按产品支持与固定开销实测。
4. 多级流水可做，但要以 Tile buffer 双缓冲和窄事件实现；整行 bitmap/metadata 常驻，
   不盲目双份分配。目标 shape 每核任务很少，收益优先级低于同步收窄和 Kernel 融合。
