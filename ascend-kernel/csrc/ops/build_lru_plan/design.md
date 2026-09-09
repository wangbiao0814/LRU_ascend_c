# build_lru_plan 三 Kernel 融合设计

## 1. 目标与结论

`build_lru_plan` 对每个 batch 的 LRU 队列执行一次批量更新。优化后把原来的
8 个 AscendC Vector Kernel 融合为 3 个：

| 融合 Kernel | 原阶段 | 核间屏障数 |
|---|---|---:|
| `build_lru_plan_hit_fused` | K1 hit-local + K2 hit-scan | 1 |
| `build_lru_plan_candidate_fill_fused` | K3 candidate-local + K4 scan + K5 fill | 2 |
| `build_lru_plan_keep_write_fused` | K6 keep-local + K7 scan + K8 write | 2 |

融合 Kernel 内通过软 `SyncAll` 建立 GM 可见的阶段边界。Scan 阶段不再让每个
Tile 重复扫描整张 count 表，而是由每个 batch 的一个 leader 只扫描一次。

对重点场景 `B=1, K=2048, AIV=40`：

```text
TH = 48,  NH = ceil(2048 / 48) = 43
TL = 96,  NL = ceil(4096 / 96) = 43
```

三个融合 Kernel 均使用 40 个 block。所有 block 无条件执行相同次数的
`SyncAll`；没有本阶段任务的 block 只参加同步，不访问阶段数据。

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

调用方必须保证每行 `lru` 是 `0..2K-1` 的排列，`hit` 中有效 ID 位于该范围且
互不重复。非连续输入由 Host 执行 `.contiguous()`。

## 3. 算子语义

对每一行：

1. 保留 `hit` 中非 `-1` 的 ID。
2. 从原 `lru` 的 LRU 端向 MRU 端选择未命中的 ID。
3. 按 miss 从左到右的顺序填入这些 ID，得到 `hit_and_miss`。
4. 把 `hit_and_miss` 写入 `new_lru` 的前 `K` 项。
5. 原 `lru` 中未被本轮选中的 ID 稳定压缩到后 `K` 项。

## 4. 动态二维 Tiling

```text
L  = 2K
C  = 设备 AIV 数
R  = ceil(C / B)
TH = clamp(AlignDown(K / R, 8), 8, 256)
TL = clamp(AlignDown(L / R, 8), 8, 256)
NH = ceil(K / TH)
NL = ceil(L / TL)
```

Tile 长度按 8 个 `int32` 对齐。各并行阶段使用交错任务分配：

```cpp
for (int64_t task = GetBlockIdx(); task < taskCount;
     task += GetBlockNum()) {
    ProcessTask(task);
}
```

含不同任务空间的融合 Kernel 使用：

```text
blockDim = min(coreNum, max(各阶段 taskCount))
```

且 `blockDim` 必须不大于设备实际 AIV 数。

## 5. 三个融合 Kernel

### 5.1 hit_fused（原 K1 + K2）

阶段 A：`B * NH` 个任务并行计算 Tile 内 miss exclusive rank，并写出
`hitRank` 和 `(localCount, localCount)`。

阶段 B：所有 block 调用一次 `SyncAll`。随后每个 batch 的 leader：

1. 扫描该行 `NH` 个 count，在 lane 0 写 Tile exclusive offset，lane 1 保留 count；
2. 读取一次完整 hit 行，构造 `int8 hitBitmap`；
3. 写出 `missCount`。

`hitRank` 保持 Tile 内 rank。后续 fill 使用：

```text
globalMissRank = hitRank[index] + hitTileInfo[tile].offset
```

这样避免原 K2 对 `hitRank` 的一次 GM 读和一次 GM 写。

### 5.2 candidate_fill_fused（原 K3 + K4 + K5）

阶段 A：`B * NL` 个任务并行读取 `hitBitmap`，从每个 LRU Tile 右端向左筛选
未命中 ID，写固定 Tile slot 和 count。

阶段 B：第一次 `SyncAll` 后，每个 batch 的 leader：

1. 从最右 Tile 向左读取候选，仅打包前 `missCount` 项到
   `candidatePacked[B, K]`；
2. 以 `hitBitmap` 为初值，把打包候选置 1，单次写出 `selectedBitmap`。

阶段 C：第二次 `SyncAll` 后，`B * NH` 个 fill 任务直接使用全局 miss rank 索引
`candidatePacked`，同时写 `hit_and_miss` 和 `new_lru[:, :K]`。

该设计消除了旧 K4 的 `O(NL²)` 重复扫描、K5 的候选 Tile 区间搜索和
selected bitmap 原子累加。

### 5.3 keep_write_fused（原 K6 + K7 + K8）

阶段 A：`B * NL` 个任务并行读取 `selectedBitmap`，稳定压缩未选 ID，写固定
Tile slot 和 count。

阶段 B：第一次 `SyncAll` 后，每个 batch 的 leader 正向扫描 count，在 lane 0
写 exclusive offset。

阶段 C：第二次 `SyncAll` 后，Tile 任务并行把固定 slot 写到
`new_lru[K + offset]`。前 `K` 项已由上一个融合 Kernel 写好，不再复制
`hit_and_miss`。

## 6. Workspace

```text
Kp = AlignUp(K, 128)
Lp = AlignUp(2K, 128)
Tp = 8 * max(NL, NH)
Vp = NL * TL
Sp = 8 * AIV
```

| Workspace | dtype / Shape | 用途 |
|---|---|---|
| `hit_bitmap` | int8 `[B, Lp]` | leader 单次构建，candidate 阶段读取 |
| `selected_bitmap` | int8 `[B, Lp]` | leader 单次构建，keep 阶段读取 |
| `hit_rank` | int32 `[B, Kp]` | Tile 内 miss rank |
| `hit_tile_info` | int32 `[B, Tp]` | hit Tile offset/count |
| `tile_counts` | int32 `[B, Tp]` | candidate/keep 阶段复用 |
| `tile_values` | int32 `[B, Vp]` | candidate/keep 固定 Tile slot 复用 |
| `candidate_packed` | int32 `[B, Kp]` | 连续 miss 候选 |
| `miss_count` | int32 `[B, 8]` | 每行 miss 数 |
| `hit_sync` | int32 `[Sp]` | 1 个软同步区域 |
| `candidate_sync` | int32 `[2, Sp]` | 2 个独立软同步区域 |
| `keep_sync` | int32 `[2, Sp]` | 2 个独立软同步区域 |

软同步 GM 区域在 Host 侧初始化为 0，每个区域至少为 `AIV * 32B`。每个 Kernel
还在 UB 中分配同样大小的同步区。

## 7. UB 预算

Host 分别计算三个融合 Kernel 的 UB 上界，并取最大值：

```text
hit:
  max(2*TH+8, Kp+Tp+8) * 4 + Lp + Sp*4

candidate/fill:
  max(2*TL+8, Vp+Tp+Kp+8, 3*TH+Kp+16) * 4 + Lp + Sp*4

keep/write:
  max(2*TL+8, 8*NL+8, TL+8) * 4 + Lp + Sp*4
```

其中 bitmap 已由 float32 改为 int8。Host 在 UB 上界之外继续预留 8KB。

## 8. 同步与并发安全

- 三个融合 Kernel 在 current stream 顺序发射。
- Kernel 内所有 block 无条件执行相同数量、相同顺序的 `SyncAll`。
- 每个软同步点使用独立的、预先清零的 GM 区域。
- `SyncAll` 位于任务循环之外，不会因某个 block 的任务数量不同而产生死锁。
- Host 保证 `blockDim <= coreNumAiv`。
- bitmap 每行只有 leader 写，不再需要 atomic。
- 每个 Tile 只写自己的固定 value/count slot。
- 最终输出区间互不重叠。

## 9. 后续核内流水优化

融合正确性和性能基线通过后，再实施：

1. 用 `TQue<VECIN/VECOUT>` 和 `BUFFER_NUM=2` 替换通用 `PIPE_ALL` 串行搬运；
2. 扫描 leader 对长行采用分块 CopyIn/Scalar/CopyOut 流水；
3. candidate/keep 尝试 `Gather + Compare + GatherMask` 替换 Scalar 随机索引与压缩；
4. 对 `1x/2x/4x AIV` 任务波数做 profile，只有每核至少两个任务时启用跨 Tile 双缓冲；
5. 分别覆盖全 miss、无 miss、80% hit、小 shape 和 `K=2048`。

## 10. 验证要求

必须逐元素满足：

```text
new_lru[:, :K] == hit_and_miss
```

测试包含 CPU reference、三融合阶段 CPU 模拟、随机 shape、Tile 边界、全 miss、
无 miss、非连续输入和错误 dtype。目标设备还需用 profiler 对比 8-Kernel 与
3-Kernel 的 launch、SyncAll、Scalar、MTE2/MTE3 和 GM 带宽。
