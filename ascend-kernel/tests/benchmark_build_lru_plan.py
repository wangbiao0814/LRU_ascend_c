#!/usr/bin/env python3
"""Correctness and performance benchmark for build_lru_plan on NPU."""

import argparse
import glob
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401


def load_operator_library():
    try:
        import ascend_kernel  # noqa: F401

        return
    except ImportError:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pattern = os.path.join(
            project_root, "python", "ascend_kernel", "ascend_kernel", "lib", "*.so"
        )
        libraries = glob.glob(pattern)
        if not libraries:
            raise ImportError(
                "ascend_kernel is not installed and no local operator library was found"
            )
        torch.ops.load_library(libraries[0])


def make_inputs(batch_size, k, hit_rate, seed):
    generator = torch.Generator().manual_seed(seed)
    hit_count = max(0, min(k, int(round(k * hit_rate))))
    lru_rows = []
    hit_rows = []

    for _ in range(batch_size):
        lru_row = torch.randperm(2 * k, generator=generator).to(torch.int16)
        hit_indices = torch.randperm(2 * k, generator=generator)[:hit_count]
        valid_hits = lru_row[hit_indices]
        hit_row = torch.cat(
            [valid_hits, torch.full((k - hit_count,), -1, dtype=torch.int16)]
        )
        hit_row = hit_row[torch.randperm(k, generator=generator)]
        lru_rows.append(lru_row)
        hit_rows.append(hit_row)

    return torch.stack(lru_rows), torch.stack(hit_rows), hit_count


def build_lru_plan_reference(lru, hit, hit_mask):
    output_lru = []
    output_hit = []
    for lru_row, hit_row, mask_row in zip(
        lru.tolist(), hit.tolist(), hit_mask.tolist()
    ):
        hit_set = {value for value, is_hit in zip(hit_row, mask_row) if is_hit}
        miss_values = [value for value in reversed(lru_row) if value not in hit_set]
        miss_iter = iter(miss_values)
        filled = [
            value if is_hit else next(miss_iter)
            for value, is_hit in zip(hit_row, mask_row)
        ]
        selected = set(filled)
        remaining = [value for value in lru_row if value not in selected]
        output_hit.append(filled)
        output_lru.append(filled + remaining)
    return (
        torch.tensor(output_lru, dtype=torch.int16),
        torch.tensor(output_hit, dtype=torch.int16),
    )


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def benchmark(args):
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    load_operator_library()
    device = torch.device(args.device)
    lru_cpu, hit_cpu, hit_count = make_inputs(
        args.batch_size, args.k, args.hit_rate, args.seed
    )
    hit_mask_cpu = hit_cpu.ne(-1)
    expected_lru, expected_hit = build_lru_plan_reference(
        lru_cpu, hit_cpu, hit_mask_cpu
    )
    lru = lru_cpu.to(device)
    hit_mask = hit_mask_cpu.to(device)
    correctness_hit = hit_cpu.to(device)

    actual_lru = torch.ops.npu.build_lru_plan(
        lru, correctness_hit, hit_mask
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual_lru.cpu(), expected_lru, rtol=0, atol=0)
    torch.testing.assert_close(correctness_hit.cpu(), expected_hit, rtol=0, atol=0)

    total_calls = args.warmup + args.iterations + args.latency_samples
    hit_pool = (
        hit_cpu.unsqueeze(0)
        .expand(total_calls, -1, -1)
        .clone()
        .to(device)
    )
    call_index = 0

    for _ in range(args.warmup):
        torch.ops.npu.build_lru_plan(lru, hit_pool[call_index], hit_mask)
        call_index += 1
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(args.iterations):
        actual_lru = torch.ops.npu.build_lru_plan(
            lru, hit_pool[call_index], hit_mask
        )
        call_index += 1
    torch.npu.synchronize()
    total_seconds = time.perf_counter() - start

    synchronized_samples_ms = []
    for _ in range(args.latency_samples):
        sample_start = time.perf_counter()
        actual_lru = torch.ops.npu.build_lru_plan(
            lru, hit_pool[call_index], hit_mask
        )
        call_index += 1
        torch.npu.synchronize()
        synchronized_samples_ms.append((time.perf_counter() - sample_start) * 1000.0)

    average_ms = total_seconds * 1000.0 / args.iterations
    calls_per_second = args.iterations / total_seconds
    rows_per_second = calls_per_second * args.batch_size
    accesses_per_second = rows_per_second * args.k
    actual_hit_rate = hit_count / args.k

    print("build_lru_plan correctness: PASS (exact int16 match)")
    print(
        f"shape: lru=({args.batch_size}, {2 * args.k}), "
        f"hit=({args.batch_size}, {args.k})"
    )
    print(
        f"hit ratio: {actual_hit_rate:.4%} "
        f"({hit_count} hits, {args.k - hit_count} misses per row)"
    )
    print(f"warmup iterations: {args.warmup}")
    print(f"timed iterations: {args.iterations}")
    print(f"stream average latency: {average_ms:.6f} ms/call")
    print(f"throughput: {calls_per_second:.2f} calls/s")
    print(f"row throughput: {rows_per_second:.2f} rows/s")
    print(f"access throughput: {accesses_per_second:.2f} items/s")
    print(
        "synchronized latency: "
        f"mean={statistics.mean(synchronized_samples_ms):.6f} ms, "
        f"p50={percentile(synchronized_samples_ms, 0.50):.6f} ms, "
        f"p90={percentile(synchronized_samples_ms, 0.90):.6f} ms"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--hit-rate", type=float, default=0.80)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--latency-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.k <= 0:
        parser.error("--batch-size and --k must be positive")
    if args.k > 16384:
        parser.error("--k must be <= 16384 because IDs are stored as int16")
    if not 0.0 <= args.hit_rate <= 1.0:
        parser.error("--hit-rate must be in [0, 1]")
    if args.warmup < 0 or args.iterations <= 0 or args.latency_samples <= 0:
        parser.error("iteration counts are invalid")
    return args


if __name__ == "__main__":
    benchmark(parse_args())
