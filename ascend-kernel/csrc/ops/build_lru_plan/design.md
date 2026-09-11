# build_lru_plan：A2/A3 全向量化设计

## 1. 结论

当前 row-fused kernel 虽然已经把一行 `hit[K] + lru[2K]` 放进 UB，但四个核心阶段
仍由 `LocalTensor::GetValue/SetValue` 逐元素执行。新路径删除所有业务元素上的 Scalar
访问，使用：

```text
Sort + Gather + Compare       原 hit membership
GatherMask                    稳定压紧非原 hit 的 LRU
CumSum                        miss 的一维前缀 rank
Gather + Select               miss 候选回填
```

目标 SoC `Ascend910_9382` 属于 A2/A3 路线，基础 Scatter ISASI 不支持，因此不构造
随机写 bitmap。采用“排序后的 hit + 向量并行二分”代替 Scatter，所有 GM workspace
仍为 0。

---

## 2. 接口与约束

```cpp
at::Tensor build_lru_plan(
    const at::Tensor &lru,
    at::Tensor &hit,
    const at::Tensor &hit_mask);
```

```cpp
m.def("build_lru_plan(Tensor lru, Tensor(a!) hit, Tensor hit_mask) "
      "-> Tensor new_lru");
```

| 名称 | Shape | dtype | 语义 |
|---|---:|---|---|
| `lru` | `[B,2K]` | int16 | 每行是 `0..2K-1` 的 MRU→LRU 排列 |
| `hit` | `[B,K]` | int16 | `-1` 表示 miss，kernel 原地回填 |
| `hit_mask` | `[B,K]` | bool | 调用方语义输入，满足 `hit_mask == hit.ne(-1)` |
| `new_lru` | `[B,2K]` | int16 | `[filled_hit, remaining]` |

Host 保留 device、dtype、rank、shape、contiguous、storage overlap 和 `K<=16384` 检查。
性能 kernel 直接由 `hit == -1` 生成 bit mask；`hit_mask` 保留在 dispatcher 接口中以兼容
现有调用约定，但不重复搬入 UB。

调用方保证有效 hit 行内唯一，且都来自对应 LRU 行。

---

## 3. 等价变换

令 `H` 为原 hit 的有效 ID 集合，`M` 为 miss 数量，定义稳定压紧序列：

```text
F = [x for x in lru if x not in H]
```

因为 LRU 是 `2K` 个唯一 ID 的排列且 `|H|=K-M`：

```text
len(F) = 2K - (K-M) = K+M
```

原算法从 LRU 尾部向前选择不在 `H` 中的前 M 项，因此：

```text
miss_values[r] = F[K+M-1-r], 0 <= r < M
remaining      = F[:K]
```

这意味着只需对 LRU 做一次 membership 和一次稳定压紧，不必在回填后再次更新 bitmap、
也不必第二次扫描 LRU。

---

## 4. Kernel 数据流

### 4.1 Phase A：Sort 原 hit

```cpp
Duplicate(sortScore, -1.0f, sortLength);
Cast(sortScore, hitLocal, RoundMode::CAST_NONE, K);
CreateVecIndex(sortIndex, 0, sortLength);
Sort<float, true>(sortRecord, sortScore, sortIndexU32,
                  sortTmp, sortLength / 32);
Extract(sortedHit, sortIndexU32, sortRecord, sortLength / 32);
```

其中：

```text
sortLength = max(32, NextPowerOfTwo(K+1))
```

miss 与 padding 均为 `-1.0f`。合法 LRU ID 非负，因此重复 sentinel 不影响 membership。

### 4.2 Phase B：2K 路并行二分

对降序数组用 binary lifting 查找第一个 `<=query` 的位置：

```cpp
base = -1;
for (step = sortLength/2; step > 0; step >>= 1) {
    candidate = base + step;
    probe = Gather(sortedHit, candidate * sizeof(float));
    base = Select(probe > query, candidate, base);
}
probe = Gather(sortedHit, (base + 1) * sizeof(float));
notFound = (probe != query);
```

`sortLength` 是 2 的幂，所以所有 lane 执行固定 `log2(sortLength)` 轮，无 active 分支、
无逐元素 Scalar。LRU 扩到 `vectorLength=AlignUp(2K,128)`；padding 为 -1，必定命中
sentinel，因此不会进入 F。

### 4.3 Phase C：一次 GatherMask 得到 F

int16 的单 repeat 为 128 个元素。Compare 结果的 128 个有效 bit 放进独立 32B pattern
块，随后调用：

```cpp
GatherMaskParams params{1, vectorLength / 128, 8, 1};
GatherMask(nonHitForward, lruLocal, patternU16,
           false, 0, params, nonHitCount);
```

一次整行压紧保证 destination 起点 32B 对齐，也避免分 tile 压紧后 `kept` 非 16 倍导致
下一段 destination 未对齐。

合法输入下 `nonHitCount=K+M`，Scalar 只读取这一个 API 元数据。

### 4.4 Phase D：CumSum、Gather 与 Select 回填

```cpp
missMask = (Cast(hitLocal) == -1.0f);
missFlag = Select(missMask, 1.0f, 0.0f);
prefixFloat = CumSum(missFlag);           // miss lane: 1,2,...,M
prefixInt   = Cast<int32>(prefixFloat);   // 此后全部使用整数索引

candidateIndex = clamp(nonHitCount - prefixInt, 0, nonHitCount - 1);
candidate      = Gather(F, candidateIndex * sizeof(int16_t));
filledHit      = Select(hitLocal != -1, hitLocal, candidate);
```

hit lane 的 candidate 无业务语义，但 index 仍钳到合法范围，避免 speculative 越界。
全 hit 时 `nonHitCount=K`、prefix 全 0，钳位后访问 `F[K-1]`，最终 Select 保留原 hit，
因此不需要 Scalar 特判。

### 4.5 输出

```text
hit      <- filledHit
new_lru  <- concat(filledHit, F[:K])
```

每行读一次 hit/lru，写一次 hit 和两个连续 new_lru 区段。没有中间 GM workspace。

---

## 5. Tiling 与 UB

Block 级：

```text
blockDim = max(1, min(B, AIV core count))
row b    = blockIdx, blockIdx+blockDim, ...
```

每行只由一个 task 写，不需要 atomic。

对齐量：

```text
H = AlignUp(K, 128)
V = AlignUp(2K, 128)
S = max(32, NextPowerOfTwo(K+1))
```

常驻区：

| Buffer | Byte |
|---|---:|
| `hitLocal` | `2H` |
| `sortedHit` | `4S` |
| `lruLocal` | `2V` |
| `nonHitForward` | `2V` |

Sort scratch：

```text
4S score + 4S index + 8S record + sortTmpBytes
= 16S + sortTmpBytes
```

Membership scratch：

```text
4V query + 4V base + 4V candidate + 4V byteOffset
+ 4V probe + binaryMask + gatherPattern
= 20V + masks
```

Fill scratch：

```text
24H + hitMaskBytes + cumSumTmpBytes + 32B lastRow
```

Host 使用 `GetSortTmpSize` 和 `GetCumSumMaxMinTmpSize` 查询高阶 API 临时空间；CumSum
先保证 minimum，再在剩余 UB 内尽量靠近 maximum。最终：

```text
persistent = 2H + 4S + 4V
scratch    = max(sortScratch, membershipScratch, fillScratch)
workBytes  = persistent + scratch
workBytes + 8KB <= device UB
```

超过容量或 `Sort repeat > 255` 时明确报错。本次不引入多 kernel GM-workspace fallback。

---

## 6. 同步与约束

- GM↔UB 均使用 `DataCopyPad`；
- padding 初始化完成后用 `V_MTE2`，输入到达后用 `MTE2_V`；
- `GatherMask::rsvdCnt` 用 `V_S` 后读取，再以 `S_V` 交给 Vector 指令；
- 输出前用 `V_MTE3`，复用 UB 前用 `MTE3_V`；
- Gather offset 为 uint32 字节偏移；
- Compare 输入长度按 256B 对齐；
- CumSum inner 的 float 字节数按 32B 对齐；
- 不使用 `PIPE_ALL`，不使用业务元素 `GetValue/SetValue`。

---

## 7. 测试与验收

CPU 增加独立 `build_lru_plan_vector_model`，逐步模拟 power-of-two Sort、binary lifting、
F、prefix rank 和 reverse-suffix Gather，并与原 reference 逐元素比较。

覆盖：

```text
K: 1,3,7,8,9,31,32,33,127,128,129,257,512,2048
B: 1,3,41
hit rate: random, all miss, all hit
```

NPU DoD：

1. 所有精度 case 与 CPU reference 完全一致；
2. `new_lru[:,:K] == updated hit`，每行仍为 `0..2K-1` 的排列；
3. profiler 中没有业务元素 Scalar load/store；
4. 对目标 `K=2048` 比较旧 Scalar row-fused 与新 vector path 的 p50/p90、Vector/Scalar
   pipe、Sort/GatherMask/CumSum 周期和 UB occupancy；
5. 只有实测收益后才决定是否需要 small-K dispatch。
