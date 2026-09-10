# build_lru_plan：整行 K / 2K Row-Fused 设计

## 1. 目标与设计结论

当前实现只有 K1 使用“一个 AIV task 独占一个 batch 行”，K2、K3、K4 仍按小 Tile
切分。以 `B=1,K=2048,C=40` 为例，当前 `TH=48`、`TL=96`、`NH=NL=43`；K2/K3
的每个小 Tile 都会重复读取整行 bitmap、candidate values 或 metadata，K4 又把 K3 的
临时结果读回后写到最终输出。高 task 并行度同时带来了成倍的 GM 流量、短 DMA 和 Scalar
prefix/suffix 扫描。

用户提出的方向正确：数据对象为 hit/mask/candidate 时，单核粒度取 `K`；数据对象为
LRU 时，单核粒度取 `2K`，整行读入、计算、最后连续写回。

进一步地，当一整行由同一个 AIV core 处理时，原 K1→K2→K3→K4 的依赖全部变成核内
顺序依赖，不再需要 Kernel launch 作为全局 barrier。因此推荐快速路径不是四个“大 Tile
Kernel”，而是 **一个 row-fused Kernel**，把原四阶段变成四个核内逻辑 phase：

```text
每个 AIV task 独占一行 b

GM: hit[K] + hit_mask[K] + lru[2K]
                    |
                    | 一次整行搬入
                    v
UB Phase 1: 由原 hit 构造 selected bitmap
UB Phase 2: 逆序扫描 lru，生成前 K 个 candidate
UB Phase 3: 以 candidate 回填 miss，并更新 selected bitmap
UB Phase 4: 正序扫描 lru，生成 remaining K
                    |
                    | hit 一次写回；new_lru[2K] 一次写回
                    v
GM: hit(in-place)[K] + new_lru[2K]
```

row-fused 快速路径具有以下结果：

- Kernel launch 从 4 次降为 1 次；
- 删除所有阶段间 bitmap/candidate/count/keep workspace；
- bitmap 不再需要 atomic，可从 `float32[2K]` 改为 UB 内 `int16[2K]`；
- 每个输入对象整行只读取一次；
- 每个输出对象最终只连续写回一次；
- Block 级并行度为 `min(B,C)`，`B=1` 时只使用一个 AIV，这是减少冗余访存的明确权衡。

对整行数据无法装入 UB 的超大 K，Host 应选择保留的旧 tiled 路径；若项目只支持固定
小 K，则可以不保留 fallback，但必须明确报错，不能越界分配 UB。

实现路径为纯 **AscendC Vector Kernel**。本算子没有 PyTorch/NumPy 同名标准接口，继续
采用项目自定义的 `hit` 原地更新语义。

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

```python
new_lru = torch.ops.npu.build_lru_plan(lru, hit, hit_mask)
# hit 已原地回填，且 new_lru[:, :K] == hit
```

### 2.3 参数、数据类型与约束

| 名称 | 方向 | Shape | dtype | 说明 |
|---|---|---:|---|---|
| `lru` | 输入 | `[B, 2K]` | `int32` | 每行按 MRU 到 LRU 排列 |
| `hit` | 输入/原地输出 | `[B, K]` | `int32` | 输入 `-1` 表示 miss，输出时已回填 |
| `hit_mask` | 输入 | `[B, K]` | `bool` | 调用前 `hit != -1` 的掩码 |
| `new_lru` | 输出 | `[B, 2K]` | `int32` | 更新后的 LRU |

Host 必须检查：

- 三个 Tensor 位于同一 NPU device；
- `lru`、`hit` 为 `int32`，`hit_mask` 为 `bool`；
- 三者均为 rank 2，`B>0`、`K>0`，shape 分别为 `[B,2K]`、`[B,K]`、`[B,K]`；
- `hit` 必须连续，且不与 `lru`、`hit_mask` 存储重叠；
- `lru`、`hit_mask` 可由 Host 转连续，但性能路径要求调用方直接提供连续 Tensor；
- `K <= INT32_MAX/2`，所有 shape、offset、UB 字节数计算均检查 int64 溢出；
- row-fused 路径满足第 5 节 UB 容量约束，否则进入 tiled fallback 或明确报错。

性能路径不做 NPU→CPU 值域同步。调用方保证：

```text
hit_mask[b,i] == false  <=>  hit[b,i] == -1
lru[b] 是 0..2K-1 的排列
有效 hit ID 位于 0..2K-1，且行内互不重复
```

本算子仅涉及 int32/bool 搬运、比较和索引，不涉及 FP16/BF16，也不需要升精度 Cast。

---

## 3. 计算语义与容量证明

对每个 batch 行：

```python
original_hit_ids = {hit[i] for i in range(K) if hit_mask[i]}
candidates = [x for x in reversed(lru) if x not in original_hit_ids]

miss_rank = 0
for i in range(K):
    if not hit_mask[i]:
        hit[i] = candidates[miss_rank]
        miss_rank += 1

selected = set(hit)
new_lru = hit + [x for x in lru if x not in selected]
```

设原 hit 数为 `H`、miss 数为 `M=K-H`。LRU 行含 `2K` 个唯一 ID，H 个原 hit 都在
LRU 中，因此非原-hit ID 数为：

```text
2K - H = K + M >= K
```

所以 Phase 2 只需保存淘汰顺序的前 `K` 个 candidate，必然覆盖最坏的全 miss 情况；
Phase 3 实际只消费前 `M` 个。最终 selected 集合恰有 K 个 ID，因此 Phase 4 也恰好
产生 K 个 remaining ID。

---

## 4. Row-Fused Kernel 与 AscendC API 伪代码

推荐 Kernel 名称：

```cpp
extern "C" __global__ __aicore__ void build_lru_plan_row(...);
```

任务空间为 B，每个 task 独占一行：

```cpp
for (int64_t b = GetBlockIdx(); b < batchSize; b += GetBlockNum()) {
    ProcessRow(b);
}
```

### 4.1 整行搬入与 bitmap 初始化

```cpp
Duplicate(bitmapLocal, static_cast<int16_t>(0), bitmapElements);

DataCopyPad(hitLocal, hitGm[b * K],
            K * sizeof(int32_t), noPad);
DataCopyPad(maskLocal, hitMaskGm[b * K],
            K * sizeof(uint8_t), noPad);
DataCopyPad(lruLocal, lruGm[b * 2 * K],
            2 * K * sizeof(int32_t), noPad);

WaitMte2ToScalar();
WaitVectorToScalar();
```

`bitmapLocal` 只在本 task 内由 Scalar 读写，不需要 atomic，也不写入 GM。旧实现为了跨
task atomic bitmap 使用 float32；row-fused 路径改为 int16，取值仅为 0/1。没有使用
uint8，是因为 Ascend A2/A3 的基础 `Duplicate` 不支持 uint8 operand。

### 4.2 Phase 1：标记原 hit

```cpp
for (int64_t i = 0; i < K; ++i) {
    if (maskLocal.GetValue(i) != 0) {
        int32_t id = hitLocal.GetValue(i);
        bitmapLocal.SetValue(id, static_cast<int16_t>(1));
    }
}
```

这里以 `hit_mask` 为语义来源；输入契约保证有效位置的 ID 不是 -1。

### 4.3 Phase 2：生成 K 个淘汰候选

```cpp
int32_t candidateCount = 0;
for (int64_t p = 2 * K; p > 0 && candidateCount < K; --p) {
    int32_t id = lruLocal.GetValue(p - 1);
    if (bitmapLocal.GetValue(id) == 0) {
        candidateLocal.SetValue(candidateCount++, id);
    }
}
// 合法输入下 candidateCount == K。
```

`candidateLocal` 必须与 `lruLocal` 分离。若逆向读 lru、从低地址紧凑写同一 buffer，写
指针可能在扫描结束前覆盖尚未读取的低地址 LRU 数据。

### 4.4 Phase 3：回填 hit，并扩展 selected bitmap

```cpp
int32_t missRank = 0;
for (int64_t i = 0; i < K; ++i) {
    if (maskLocal.GetValue(i) == 0) {
        int32_t id = candidateLocal.GetValue(missRank++);
        hitLocal.SetValue(i, id);  // hitLocal 原地成为 filled hit
        bitmapLocal.SetValue(id, static_cast<int16_t>(1));
    }
}
```

Phase 1 已标记原 hit；Phase 3 再标记实际用于回填的前 M 个 candidate。此时 bitmap
准确表示最终 selected 集合，不需要第二张 bitmap。

### 4.5 Phase 4：复用 candidate buffer 生成 remaining

candidate 在 Phase 3 后已消费完，可将 `candidateLocal[K]` 原地复用为 `remainingLocal`：

```cpp
int32_t keepCount = 0;
for (int64_t p = 0; p < 2 * K; ++p) {
    int32_t id = lruLocal.GetValue(p);
    if (bitmapLocal.GetValue(id) == 0) {
        candidateLocal.SetValue(keepCount++, id);
    }
}
// 合法输入下 keepCount == K。
```

这个阶段不再需要 `keep_counts`、prefix scan 或 K4。

### 4.6 组装与最终连续写回

为了使 `new_lru` 只进行一次连续 MTE3，完成 Phase 4 后复用 `lruLocal[2K]` 作为最终输出：

```cpp
for (int64_t i = 0; i < K; ++i) {
    lruLocal.SetValue(i, hitLocal.GetValue(i));
    lruLocal.SetValue(K + i, candidateLocal.GetValue(i));
}

WaitScalarToMte3();
DataCopyPad(hitGm[b * K], hitLocal,
            K * sizeof(int32_t));
DataCopyPad(newLruGm[b * 2 * K], lruLocal,
            2 * K * sizeof(int32_t));
WaitMte3ToScalar(); // 同一 core 进入下一行并复用 UB 前必须等待
```

Phase 4 扫描完整个原始 lru 后才覆盖 `lruLocal`，因此组装不会破坏未消费输入。最终每行
只有一次 hit 原地写回和一次完整 new_lru 写回。

---

## 5. 两级 Tiling 与 UB 分配

### 5.1 Tiling 参数

```cpp
struct BuildLruPlanTilingData {
    int64_t batchSize;      // B
    int64_t k;              // K
    int64_t lruLength;      // 2K
    int64_t bitmapElements; // A16(2K)，int16 元素数
    int64_t hitElements;    // A8(K)
    int64_t lruElements;    // A8(2K)
    int64_t maskBytes;      // A32(K)
    int64_t tilingKey;      // 0=row-fused, 1=tiled-fallback
};
```

当前工程可继续通过 launch 参数传递这些值；该结构定义字段语义，供后续 code-gen 对齐。

### 5.2 Block 级：按 batch 行切分

```text
taskCount = B
blockDim  = max(1, min(B, deviceAivCoreNum))
```

`B=1` 使用 1 个 AIV，`B=16` 使用 16 个 AIV，`B>C` 时每核交错处理多行。同一行只有
一个 writer，不需要 atomic，也不存在跨 task prefix/suffix 合并。

### 5.3 UB 级：K / 2K 整行 Tile

定义：

```text
A8(x)  = AlignUp(x, 8)    # int32 的 32B 对齐元素数
A16(x) = AlignUp(x, 16)   # int16 的 32B 对齐元素数
A32(x) = AlignUp(x, 32)   # uint8 的 32B 对齐字节数

HK = A8(K)
L  = A8(2K)
BM = A16(2K)
MK = A32(K)
```

| Buffer | dtype | 数量 | 大小（Byte） | 生命周期 |
|---|---|---:|---:|---|
| `bitmapLocal` | int16 | 1 | `2*BM` | Phase 1–4 |
| `hitLocal` | int32 | 1 | `4*HK` | 搬入至最终写回 |
| `maskLocal` | uint8 | 1 | `MK` | Phase 1、3 |
| `lruLocal` | int32 | 1 | `4*L` | 搬入、Phase 2/4、最终输出 |
| `candidateLocal` | int32 | 1 | `4*HK` | Phase 2/3，Phase 4 复用为 remaining |
| **总计** | | | **`2*BM + MK + 8*HK + 4*L`** | |

这也是 row-fused 路径的 `bufferCoefficient` 精确形式。所有子区域起点按 32B 对齐，不使用
double buffer；单行各 phase 有顺序依赖，`B<=C` 时每核通常也没有下一行可供稳定流水。

Host 必须验证：

```text
fusedUbBytes = 2*A16(2K) + A32(K) + 8*A8(K) + 4*A8(2K)
fusedUbBytes + UB_RESERVE_BYTES <= deviceUbBytes
UB_RESERVE_BYTES = 8KB
```

对 `K=2048`：

```text
BM=4096, MK=2048, HK=2048, L=4096
fusedUbBytes = 8192 + 2048 + 16384 + 16384 = 43,008 Byte
fusedUbBytes + 8KB reserve = 51,200 Byte
```

约占示例 192KB UB 的 26.0%，容量充足。渐近 UB 需求约为 `21K Byte`，仍小于把 bitmap
保留为 float32 时的约 `25K Byte`。

### 5.4 超大 K fallback

为避免缩小已有接口的可接受范围，推荐把当前四 Kernel 小 Tile 实现保留为
`tilingKey=1` fallback，仅在 row-fused UB 容量检查失败时使用。Host 必须分别计算两条
路径的 workspace、blockDim 和 launch 参数。

如果项目明确只覆盖 `K<=2048` 等固定范围，可以删除 fallback，并用 `TORCH_CHECK`
输出 `K`、所需 UB、reserve 后可用 UB。禁止尝试用截断 Tile 长度启动 row-fused Kernel，
因为它依赖完整行在同一 task 内可见。

---

## 6. Workspace

row-fused 快速路径的 bitmap、candidate、counts、keep 全部只存在于 UB，**GM workspace
大小为 0 Byte**。只分配最终 `new_lru` 输出。

以下当前 workspace 在快速路径全部删除：

- `hit_bitmap`
- `hit_counts`
- `candidate_values`
- `candidate_counts`
- `keep_values`
- `keep_counts`

对 `B=1,K=2048`，当前实现按现有 stride 公式约分配 49,984 Byte workspace；快速路径
降为 0。无需额外 system workspace。若进入 tiled fallback，则仍按旧路径公式分配临时
Tensor，并由 PyTorch current stream 生命周期管理。

---

## 7. 性能分析

### 7.1 当前路径的重复开销

对 `B=1,K=2048,TH=48,TL=96,NH=NL=43`：

- K2 的 43 个 task 各读一次完整 float bitmap；
- K3 的 43 个 Hit task 各读一次接近 2K 的 candidate row；
- K3 的 43 个 LRU task 各读一次完整 bitmap；
- 每个 task 还读取 metadata，并做前缀/后缀 Scalar scan；
- K3 写 keep workspace，K4 再读并写最终输出；
- 总计 4 次 launch。

这类重复开销随 `NH/NL` 增长，而有效算法工作量本来只需要每行 O(K)。

### 7.2 row-fused 快速路径流量

忽略对齐 padding，每行逻辑 GM 流量为：

| 对象 | 读 | 写 |
|---|---:|---:|
| `hit` | `4K` | `4K` |
| `hit_mask` | `K` | 0 |
| `lru` | `8K` | 0 |
| `new_lru` | 0 | `8K` |
| **合计** | **`13K`** | **`12K`** |

总计 **`25K Byte/row`**；`K=2048` 时为 51,200 Byte/row。没有任何中间 GM 读写。

### 7.3 权衡与验收假设

- 主要收益：4→1 launch、输入只读一次、中间 GM workspace 清零、bitmap int16 化、删除
  metadata scan 和大量短 DMA。
- 主要代价：并行度从 `B*tileCount` 降到 B；`B=1` 时只有一个 AIV 工作。
- 目标场景：整行可驻留 UB，尤其 `K=2048,B=1/16`。
- 核心风险：所有筛选/随机 bitmap 访问仍走 Scalar `GetValue/SetValue`。若 Scalar pipe
  成为绝对瓶颈，单核执行时间可能抵消并行度损失；必须用 profiler 验证。

不预先承诺固定加速比。需要比较端到端延迟、Scalar pipe、MTE stall 与 launch 开销，
而不是只比较活跃 AIV 数。

---

## 8. 同步与并发安全

- 不同 task 只读写不同 batch 行，无需 atomic；
- `Duplicate(bitmapLocal)` 后 Scalar 首次访问使用 `V_S`；
- 三个输入完成 GM→UB 后 Scalar 首次访问使用 `MTE2_S`；
- Scalar 组装完成后，两个 GM 写出前使用 `S_MTE3`；
- 同一 core 进入下一行并复用 UB 前使用 `MTE3_S`；
- `hit` 必须先完整搬入 UB 再原地写回，不能边读边覆盖 GM；
- Phase 4 完整读取原 lru 后才允许把 `lruLocal` 改写成最终 new_lru；
- `hit` 与 `lru`/`hit_mask` 禁止存储重叠；
- 所有 UB 子区起点按 32B 对齐，GM 尾部使用 `DataCopyPad` 的逻辑字节数。

---

## 9. 实施检查清单

### 9.1 Host

- [ ] 新增 row-fused UB 字节计算与 `tilingKey` 选择；
- [ ] 快速路径不分配任何 GM workspace，只创建 `new_lru`；
- [ ] 快速路径 `blockDim=max(1,min(B,C))`；
- [ ] 快速路径只 launch `build_lru_plan_row` 一次；
- [ ] 若保留 fallback，旧四 Kernel 及 workspace 仅在容量不足时启用；
- [ ] 保持 device/dtype/shape/contiguous/overlap/overflow 检查。

### 9.2 Kernel

- [ ] 新增每 task 一整行的 `build_lru_plan_row`；
- [ ] bitmap 改为 UB 内 int16，删除 GM bitmap 和 atomic；
- [ ] 分配 `K` hit、`K` mask、`2K` lru、`K` candidate；
- [ ] Phase 2 只生成前 K 个 candidate；
- [ ] Phase 3 原地修改 hitLocal，并把使用的 candidate 标入 bitmap；
- [ ] Phase 4 复用 candidateLocal 保存 K 个 remaining；
- [ ] 扫描结束后把 `[filled_hit, remaining]` 组装到 lruLocal；
- [ ] hit 与 new_lru 分别只执行一次连续 MTE3；
- [ ] 使用窄事件依赖，不引入无条件 `PIPE_ALL`。

---

## 10. 测试与性能验收

### 10.1 CPU Reference 与性质测试

```python
hit_mask = hit.ne(-1).contiguous()
expected_lru, expected_hit = reference(lru, hit.clone(), hit_mask)
actual_lru = torch.ops.npu.build_lru_plan(lru_npu, hit_npu, hit_mask_npu)

assert torch.equal(actual_lru.cpu(), expected_lru)
assert torch.equal(hit_npu.cpu(), expected_hit)
assert torch.equal(actual_lru[:, :K].cpu(), hit_npu.cpu())
assert torch.equal(hit_mask_npu.cpu(), hit_mask)
```

CPU 模拟必须额外验证：

- Phase 2 只保存前 K 个 candidate 仍覆盖所有 miss；
- Phase 3 后 selected bitmap 恰有 K 个 1；
- Phase 4 恰生成 K 个 remaining；
- 最终每行仍为 `0..2K-1` 的排列。

测试矩阵：

| 维度 | Case |
|---|---|
| K/对齐 | `1,3,7,8,9,31,32,33,127,128,129,257,2048` |
| hit 分布 | 全 hit、全 miss、单 miss、约 50%、约 80%、miss 集中在首/尾 |
| batch | `B=1,2,16,40,41,80` |
| 接口 | dtype/shape/device/非连续 hit/存储重叠 |
| 容量 | row-fused 临界 K、刚超过临界 K 的 fallback 或报错 |

### 10.2 NPU 性能验收

对 `B=1/16,K=2048` 和真实业务 shape，对比当前 tiled 四 Kernel 与 row-fused 单 Kernel：

- 端到端 p50/p90 和异步 stream 平均耗时；
- launch 数应从 4 降为 1；
- profiler 中不再出现中间 bitmap/candidate/count/keep GM 流量；
- workspace 显存与 allocator 调用；
- AIV 活跃数、Scalar pipe 占比、MTE2/MTE3 stall；
- 全 hit、全 miss和典型 hit rate分别测量。

验收标准是结果逐元素正确、原地 alias 语义正确，且目标业务 shape 的端到端性能优于
当前实现。若 `B=1` 因 Scalar 单核瓶颈反而变慢，再评估 row-fused 与 tiled 的按规模
dispatch，而不是直接恢复固定 48/96 元素的小 Tile。

---

## 11. 文件改动范围与下一步

- `csrc/ops/build_lru_plan/op_host/build_lru_plan.cpp`
- `csrc/ops/build_lru_plan/op_kernel/build_lru_plan.cpp`
- `tests/test_build_lru_plan.py`
- `tests/benchmark_build_lru_plan.py`

本文件完成整行 row-fused 设计。下一步使用 `ascendc-operator-code-gen` 实现 Host/Kernel，
再使用 `ascendc-operator-compile-debug` 编译、安装并在 NPU 上完成精度与 profiler 验证。
