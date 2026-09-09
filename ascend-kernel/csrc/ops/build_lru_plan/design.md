# build_lru_plan 设计与实现说明

## 1. 设计结论

`build_lru_plan` 对 `B` 行独立的 LRU 队列执行一次批量更新。当前实现采用
AscendC Vector Kernel，共 8 个 Kernel，依赖同一 NPU current stream 的发射顺序
完成阶段间同步。

实现保留二维分 Tile 的主要并行路径：候选选择和保留元素判断按 `(batch, tile)`
并行；跨 Tile 扫描、bitmap 构造、miss 填充和最终写回按行并行。最终写回由一个
block 独占一行，防止压缩后相邻 Tile 的非 32B 对齐区间被不同 block 同时写入。

所有 GM 与 UB 之间的传输均使用 `DataCopyPad`。Kernel 不使用生产路径的
`GlobalTensor::GetValue/SetValue`，ID 查询所需的整行 bitmap 会先搬入 UB；所有
Scalar `GetValue/SetValue` 只作用于 `LocalTensor`。

## 2. 接口与约束

Host 接口：

```cpp
std::tuple<at::Tensor, at::Tensor> build_lru_plan(
    const at::Tensor &lru,
    const at::Tensor &hit);
```

PyTorch schema：

```text
build_lru_plan(Tensor lru, Tensor hit) ->
    (Tensor new_lru, Tensor hit_and_miss)
```

| 名称 | 方向 | Shape | dtype | 说明 |
|---|---|---:|---|---|
| `lru` | 输入 | `[B, 2K]` | `int32` | 每行按 MRU 到 LRU 排列 |
| `hit` | 输入 | `[B, K]` | `int32` | `-1` 表示 miss，其余为命中 ID |
| `new_lru` | 输出 0 | `[B, 2K]` | `int32` | 更新后的 LRU |
| `hit_and_miss` | 输出 1 | `[B, K]` | `int32` | 用淘汰 ID 填充 miss 后的访问序列 |

Host 检查：

- 输入位于同一 NPU device，dtype 均为 `int32`。
- 两个输入均为二维，batch 相同，且 `lru.size(1) == 2 * hit.size(1)`。
- `B > 0`、`K > 0`、`2K <= INT32_MAX`。
- 非连续输入通过 `.contiguous()` 转换后执行。
- 实际 UB 容量能够容纳最坏阶段的 LocalTensor。

性能路径不将输入复制到 CPU 做值域检查。调用方须保证每行 `lru` 是
`0..2K-1` 的排列，`hit` 中的有效 ID 位于该范围且互不重复；违反值约束的行为
未定义。

## 3. 算子语义

对每一行执行：

1. 保留 `hit` 中非 `-1` 的 ID 及其位置。
2. 从原 `lru` 的 LRU 端向 MRU 端扫描，跳过已有 hit，选择足够多的淘汰 ID。
3. 按 `hit` 中 miss 从左到右的顺序填入淘汰 ID，得到 `hit_and_miss`。
4. 将 `hit_and_miss` 放到 `new_lru` 的前 `K` 项。
5. 原 `lru` 中未被本轮选中的 ID 稳定压缩到后 `K` 项。

数学定义：

```text
miss_rank[b,j] = sum(hit[b,q] == -1), q < j
miss_count[b]  = sum(hit[b,q] == -1)

hit_bitmap[b, hit[b,j]] = 1, if hit[b,j] != -1
candidate[b,i] = 1 - hit_bitmap[b, lru[b,i]]
candidate_rank[b,i] = sum(candidate[b,q]), q > i

miss_values[b, candidate_rank[b,i]] = lru[b,i]
    if candidate_rank[b,i] < miss_count[b]

hit_and_miss[b,j] = hit[b,j], if hit[b,j] != -1
hit_and_miss[b,j] = miss_values[b, miss_rank[b,j]], otherwise

selected_bitmap[b, hit_and_miss[b,j]] = 1
keep[b,i] = 1 - selected_bitmap[b, lru[b,i]]
keep_rank[b,i] = sum(keep[b,q]), q < i

new_lru[b,j] = hit_and_miss[b,j], 0 <= j < K
new_lru[b,K + keep_rank[b,i]] = lru[b,i], if keep[b,i]
```

## 4. 两级 Tiling

```text
L  = 2K
TL = LRU Tile 长度
TH = hit Tile 长度
NL = ceil(L / TL)
NH = ceil(K / TH)
```

Host 优先选择 `TL = TH = 256`；若 UB 不足则降为 128。Tile 长度始终是
32B 数据块和 512B cache line 的整数倍。尾 Tile 只处理 `validLen`，并由
`DataCopyPad` 支持非 32B 对齐的逻辑长度。

核间任务映射：

| Kernel | 任务空间 | 并行方式 |
|---|---:|---|
| K1 hit local | `B` | 每个 block 处理一行的全部 hit Tile |
| K2 hit scan | `B` | 每个 block 扫描一行 |
| K3 candidate local | `B * NL` | `(batch, lru_tile)` 二维并行 |
| K4 candidate scan | `B` | 每个 block 反向扫描一行 |
| K5 fill | `B` | 每个 block 填充一行并重建 bitmap |
| K6 keep local | `B * NL` | `(batch, lru_tile)` 二维并行 |
| K7 keep scan | `B` | 每个 block 正向扫描一行 |
| K8 write | `B` | 每个 block 独占一行完成最终写回 |

每个 Kernel 使用交错取任务：

```cpp
for (int64_t task = GetBlockIdx(); task < taskCount; task += GetBlockNum()) {
    ProcessTask(task);
}
```

Host 动态查询 AIV core 数，`blockDim = min(taskCount, coreNum)`，至少为 1。

## 5. 八阶段 Kernel

### K1 `build_lru_plan_hit_local`

每个 block 独占一行：在 UB 中清零整行 `hit_bitmap`，逐 hit Tile 计算局部
exclusive miss rank，并把有效 hit 标记进 bitmap。每个 Tile 的 miss 数写入
独占 32B 计数槽。

K1 按行执行而不是按 hit Tile 执行，因为多个 Tile 并发更新同一 bitmap 行会产生
数据块级写冲突。

### K2 `build_lru_plan_hit_scan`

每行读取 `hit_rank` 和 Tile 计数，在 Tile 间执行正向 exclusive scan，把局部 miss
rank 转为全局 miss rank，并写出 `miss_count`。`-1` 是有效 hit 的 rank 哨兵，扫描
时保持不变。

### K3 `build_lru_plan_candidate_local`

按 `(batch, lru_tile)` 并行。每个任务将该行 bitmap 和一个 LRU Tile 搬入 UB，
从 Tile 右端向左计算局部 candidate rank。Tile candidate 数写到该 Tile 独占的
32B 计数槽。

### K4 `build_lru_plan_candidate_scan`

每行从最后一个 LRU Tile 向前扫描 Tile 计数，生成反向 exclusive offset，并将
offset 加到有效 candidate rank 上。

### K5 `build_lru_plan_fill`

每个 block 独占一行：

1. 在 UB 中建立长度为 `K` 的 `miss_values`，它不是 GM workspace。
2. 读取 candidate global rank，保留 rank 小于 `miss_count` 的淘汰 ID。
3. 逐 hit Tile 填充 `hit_and_miss`。
4. 在 UB 中把 bitmap 清零并重建为 `selected_bitmap`，整行写回供 K6 使用。

### K6 `build_lru_plan_keep_local`

按 `(batch, lru_tile)` 并行。加载整行 selected bitmap 和一个 LRU Tile，在 Tile
内正向计算局部 keep rank，并写入独占 32B 的 Tile 计数槽。

### K7 `build_lru_plan_keep_scan`

每行正向扫描 Tile keep count，将局部 keep rank 转成全局 keep rank。计数槽中的
第一个 `int32` 随后保存该 Tile 在最终尾部中的起始 offset。

### K8 `build_lru_plan_write`

每个 block 独占一个输出行。先逐 Tile 复制 `hit_and_miss` 到头部，再逐 LRU Tile
在 UB 内稳定压缩 keep 元素，并按 K7 生成的 Tile offset 写入尾部。

尽管不同 Tile 的逻辑输出区间互不重叠，压缩长度不保证为 8 个 `int32` 的整数倍；
因此 K8 不让多个 block 同时写同一行，避免两个 MTE3 写共享一个 32B 数据块。

## 6. Workspace 与对齐

```text
Kp = AlignUp(K, 128)
Lp = AlignUp(2K, 128)
Tp = 8 * max(NL, NH)
Mp = 8
```

| Workspace | 物理 Shape | 生命周期与复用 |
|---|---:|---|
| `bitmap` | `[B, Lp]` | K1-K4 为 hit bitmap；K5-K6 为 selected bitmap |
| `hit_rank` | `[B, Kp]` | K1 写局部 rank；K2 改为全局 rank；K5 读 |
| `lru_rank` | `[B, Lp]` | K3/K4 为 candidate rank；K6/K7 覆盖为 keep rank；K8 读 |
| `tile_counts` | `[B, Tp]` | K1-K2、K3-K4、K6-K8 分阶段复用 |
| `miss_count` | `[B, Mp]` | K2 写；K5 读 |

```text
workspaceElements = B * (2*Lp + Kp + Tp + Mp)
workspaceBytes    = 4 * workspaceElements
```

每个 Tile 计数使用一个 32B slot，仅 slot 的第一个 `int32` 保存计数或 offset。
这样 K3/K6 的并行 Tile 写不会共享同一数据块。`miss_count` 同样按 32B 行 stride
存放。

## 7. UB 预算

当前实现使用一个 `TBuf<TPosition::VECCALC>`，再切分为多个 LocalTensor。各阶段
所需 `int32` 元素数为：

```text
K1: Lp + 2*TH + Tp
K2: Kp + Tp + Mp
K3/K6: Lp + 2*TL + 8
K4/K7: Lp + Tp
K5: Lp + Kp + 2*TL + 3*TH + Mp
K8: 3*max(TL, TH) + 8
```

K5 是当前参数范围内的最坏阶段。Host 使用其精确公式计算容量，并额外保留 8KB；
若 256 Tile 不满足则降为 128，仍不满足时直接报错。整行 bitmap 必须能进入 UB，
V1 不提供 GM Scalar fallback。

## 8. 同步与并发安全

- 8 个 Kernel 由 Host 在同一 current stream 顺序发射，不做 Host 设备同步。
- 基线 Kernel 在 LocalTensor 的 DMA、Vector 初始化和 Scalar 访问之间使用
  `PipeBarrier<PIPE_ALL>()`。
- 输入、rank 与 bitmap 行 stride 均按至少 32B 对齐；bitmap/rank 进一步按 512B
  对齐。
- K3/K6 的 Tile 起点由 `TL` 保证 512B 对齐；Tile count 使用独占 32B slot。
- K8 每行单 block，消除非对齐压缩边界的跨 block 写竞争。
- 不使用原子操作，不使用浮点 Cast，不使用标准库数学函数。

## 9. Host 调度与框架注册

Host 完成输入检查、连续化、动态查询 UB/Core、workspace 分配，并按以下顺序发射：

```text
hit_local -> hit_scan -> candidate_local -> candidate_scan
          -> fill -> keep_local -> keep_scan -> write
```

所有传给 `EXEC_KERNEL_CMD` 的 Tensor 和标量均先保存为具名左值。算子已在
`csrc/ops.h` 声明，在 `csrc/register.cpp` 注册，并加入 `csrc/CMakeLists.txt` 的
Host/Kernel 源文件列表。

## 10. 正确性测试

CPU 参考实现：

```python
def build_lru_plan_reference(lru, hit):
    output_lru = []
    output_hit = []
    for lru_row, hit_row in zip(lru.tolist(), hit.tolist()):
        hit_set = {x for x in hit_row if x != -1}
        misses = [x for x in reversed(lru_row) if x not in hit_set]
        miss_iter = iter(misses)
        filled = [x if x != -1 else next(miss_iter) for x in hit_row]
        selected = set(filled)
        remaining = [x for x in lru_row if x not in selected]
        output_hit.append(filled)
        output_lru.append(filled + remaining)
    return output_lru, output_hit
```

测试覆盖：

- `B=1,K=1`，以及 `K=127/128/129/257` 的 Tile 边界。
- 随机合法输入、全 miss、无 miss。
- 非 contiguous 输入。
- 错误 dtype。
- `new_lru[:, :K] == hit_and_miss` 与 CPU reference 逐元素精确比较。

## 11. 后续优化方向

在正确性和目标设备编译通过后，可依次评估：

1. 用细粒度 event 替换 `PIPE_ALL`。
2. 为 Scalar scan 引入 Vector compare/mask 快路径。
3. 增加小 shape 单 Kernel 融合路径以降低 8 次 launch 的固定开销。
4. 对重复加载整行 bitmap 的 K3/K6 测量带宽与 cache 命中率。

源方案：`二维分Tile并行LRU算子设计方案.pdf`。
