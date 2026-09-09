# build_lru_plan 全 AIV 并行设计

## 1. 目标与结论

`build_lru_plan` 对每个 batch 的 LRU 队列执行一次批量更新。当前实现采用 8 个
AscendC Vector Kernel，并把原先按行串行的 bitmap、scan、fill 和 write 阶段全部
改为二维 `(batch, tile)` 任务。Host 根据运行设备的 AIV 数动态选择 Tile，使逻辑
数据至少包含同等数量的 32B 数据块时，每个阶段都有不少于 AIV 数量的有效任务。

对重点场景 `B=1, K=2048, AIV=40`：

```text
TH = 48,  NH = ceil(2048 / 48) = 43
TL = 96,  NL = ceil(4096 / 96) = 43
```

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

## 5. 八阶段 Kernel

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

## 6. Workspace 与复用

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

## 7. UB 预算

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

## 8. 同步与并发安全

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
