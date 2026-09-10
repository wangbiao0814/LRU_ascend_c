#!/usr/bin/env python3

import glob
import os

import pytest
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None


def build_lru_plan_reference(lru, hit, hit_mask=None):
    if hit_mask is None:
        hit_mask = hit.ne(-1)
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


def make_inputs(batch_size, k, seed, all_miss=False, no_miss=False):
    generator = torch.Generator().manual_seed(seed)
    lru_rows = []
    hit_rows = []
    for _ in range(batch_size):
        lru_row = torch.randperm(2 * k, generator=generator, dtype=torch.int64)
        if all_miss:
            valid_count = 0
        elif no_miss:
            valid_count = k
        else:
            valid_count = int(torch.randint(0, k + 1, (1,), generator=generator))
        valid = lru_row[:valid_count].to(torch.int16)
        hit_row = torch.cat(
            [valid, torch.full((k - valid_count,), -1, dtype=torch.int16)]
        )
        order = torch.randperm(k, generator=generator)
        lru_rows.append(lru_row.to(torch.int16))
        hit_rows.append(hit_row[order])
    return torch.stack(lru_rows), torch.stack(hit_rows)


def build_lru_plan_staged(lru, hit, tile_length):
    """CPU simulation of the eight staged kernels, including tile scans."""
    batch_size, k = hit.shape
    lru_length = 2 * k
    output_lru = []
    output_hit = []

    for batch in range(batch_size):
        lru_row = lru[batch].tolist()
        hit_row = hit[batch].tolist()

        bitmap = [0] * lru_length
        hit_rank = [-1] * k
        hit_counts = []
        for start in range(0, k, tile_length):
            prefix = 0
            for index in range(start, min(start + tile_length, k)):
                value = hit_row[index]
                if value == -1:
                    hit_rank[index] = prefix
                    prefix += 1
                else:
                    bitmap[value] = 1
            hit_counts.append(prefix)

        miss_count = 0
        for tile, count in enumerate(hit_counts):
            start = tile * tile_length
            for index in range(start, min(start + tile_length, k)):
                if hit_rank[index] >= 0:
                    hit_rank[index] += miss_count
            miss_count += count

        candidate_rank = [-1] * lru_length
        candidate_counts = []
        for start in range(0, lru_length, tile_length):
            suffix = 0
            end = min(start + tile_length, lru_length)
            for index in range(end - 1, start - 1, -1):
                if bitmap[lru_row[index]] == 0:
                    candidate_rank[index] = suffix
                    suffix += 1
            candidate_counts.append(suffix)

        candidate_offset = 0
        for tile in range(len(candidate_counts) - 1, -1, -1):
            start = tile * tile_length
            end = min(start + tile_length, lru_length)
            for index in range(start, end):
                if candidate_rank[index] >= 0:
                    candidate_rank[index] += candidate_offset
            candidate_offset += candidate_counts[tile]

        miss_values = [0] * k
        for index, rank in enumerate(candidate_rank):
            if 0 <= rank < miss_count:
                miss_values[rank] = lru_row[index]

        filled = []
        selected_bitmap = [0] * lru_length
        for index, value in enumerate(hit_row):
            output_value = value if value != -1 else miss_values[hit_rank[index]]
            filled.append(output_value)
            selected_bitmap[output_value] = 1

        keep_rank = [-1] * lru_length
        keep_counts = []
        for start in range(0, lru_length, tile_length):
            prefix = 0
            end = min(start + tile_length, lru_length)
            for index in range(start, end):
                if selected_bitmap[lru_row[index]] == 0:
                    keep_rank[index] = prefix
                    prefix += 1
            keep_counts.append(prefix)

        keep_offset = 0
        for tile, count in enumerate(keep_counts):
            start = tile * tile_length
            end = min(start + tile_length, lru_length)
            for index in range(start, end):
                if keep_rank[index] >= 0:
                    keep_rank[index] += keep_offset
            keep_counts[tile] = keep_offset
            keep_offset += count

        new_lru_row = filled + [0] * k
        for index, rank in enumerate(keep_rank):
            if rank >= 0:
                new_lru_row[k + rank] = lru_row[index]

        output_hit.append(filled)
        output_lru.append(new_lru_row)

    return (
        torch.tensor(output_lru, dtype=torch.int16),
        torch.tensor(output_hit, dtype=torch.int16),
    )


def make_parallel_tile_length(logical_length, batch_size, core_num):
    desired_tiles_per_row = (core_num + batch_size - 1) // batch_size
    tile_length = (logical_length // desired_tiles_per_row // 8) * 8
    return min(max(tile_length, 8), 256)


def build_hit_row_local(hit, hit_tile_length):
    """CPU model of the row-owned K1 bitmap/count construction."""
    batch_size, k = hit.shape
    lru_length = 2 * k
    bitmaps = []
    counts = []
    for batch in range(batch_size):
        bitmap = [0] * lru_length
        row_counts = []
        for start in range(0, k, hit_tile_length):
            miss_count = 0
            for value in hit[batch, start : start + hit_tile_length].tolist():
                if value == -1:
                    miss_count += 1
                else:
                    bitmap[value] = 1
            row_counts.append(miss_count)
        bitmaps.append(bitmap)
        counts.append(row_counts)
    return bitmaps, counts


def build_hit_bitmap_sparse(hit, tile_length):
    """Simulate K1's non-atomic 4-byte stores and verify address ownership."""
    batch_size, k = hit.shape
    lru_length = 2 * k
    bitmaps = []
    counts = []
    for batch in range(batch_size):
        bitmap = [0] * lru_length
        written_ids = set()
        row_counts = []
        for start in range(0, k, tile_length):
            miss_count = 0
            for value in hit[batch, start : start + tile_length].tolist():
                if value == -1:
                    miss_count += 1
                else:
                    assert value not in written_ids
                    written_ids.add(value)
                    bitmap[value] = 1
            row_counts.append(miss_count)
        bitmaps.append(bitmap)
        counts.append(row_counts)
    return bitmaps, counts


def build_lru_plan_parallel_staged(
    lru, hit, hit_mask, hit_tile_length, lru_tile_length
):
    """CPU simulation of the fused four-kernel pipeline."""
    batch_size, k = hit.shape
    lru_length = 2 * k
    output_lru = []
    output_hit = []

    for batch in range(batch_size):
        lru_row = lru[batch].tolist()
        hit_row = hit[batch].tolist()
        mask_row = hit_mask[batch].tolist()

        hit_bitmap = [0] * lru_length
        hit_counts = []
        for start in range(0, k, hit_tile_length):
            prefix = 0
            for index in range(start, min(start + hit_tile_length, k)):
                value = hit_row[index]
                if value == -1:
                    prefix += 1
                else:
                    hit_bitmap[value] += 1
            hit_counts.append(prefix)

        miss_count = sum(hit_counts)

        candidate_values = []
        candidate_counts = []
        for start in range(0, lru_length, lru_tile_length):
            end = min(start + lru_tile_length, lru_length)
            values = [
                lru_row[index]
                for index in range(end - 1, start - 1, -1)
                if hit_bitmap[lru_row[index]] == 0
            ]
            candidate_values.append(values)
            candidate_counts.append(len(values))

        filled = []
        for hit_tile, start in enumerate(range(0, k, hit_tile_length)):
            miss_base = sum(hit_counts[:hit_tile])
            local_miss_rank = 0
            for index in range(start, min(start + hit_tile_length, k)):
                output_value = hit_row[index]
                if not mask_row[index]:
                    rank = miss_base + local_miss_rank
                    assert 0 <= rank < miss_count
                    for tile in range(len(candidate_counts) - 1, -1, -1):
                        tile_count = candidate_counts[tile]
                        if rank < tile_count:
                            output_value = candidate_values[tile][rank]
                            break
                        rank -= tile_count
                    local_miss_rank += 1
                filled.append(output_value)

        keep_values = []
        for tile, start in enumerate(range(0, lru_length, lru_tile_length)):
            end = min(start + lru_tile_length, lru_length)
            suffix_offset = sum(candidate_counts[tile + 1 :])
            within_reverse = candidate_counts[tile]
            tile_keep_values = []
            for index in range(start, end):
                value = lru_row[index]
                if hit_bitmap[value] == 0:
                    within_reverse -= 1
                    reverse_rank = suffix_offset + within_reverse
                    if reverse_rank >= miss_count:
                        tile_keep_values.append(value)
            keep_values.append(tile_keep_values)

        output_hit.append(filled)
        output_lru.append(filled + [value for tile in keep_values for value in tile])

    return (
        torch.tensor(output_lru, dtype=torch.int16),
        torch.tensor(output_hit, dtype=torch.int16),
    )


def build_lru_plan_row_fused(lru, hit, hit_mask):
    """CPU model of the single-kernel K/2K row-resident fast path."""
    batch_size, k = hit.shape
    output_lru = []
    output_hit = []

    for batch in range(batch_size):
        lru_local = lru[batch].tolist()
        hit_local = hit[batch].tolist()
        mask_local = hit_mask[batch].tolist()

        bitmap = bytearray(2 * k)
        for index in range(k):
            if mask_local[index]:
                bitmap[hit_local[index]] = 1

        candidates = []
        for value in reversed(lru_local):
            if bitmap[value] == 0:
                candidates.append(value)
                if len(candidates) == k:
                    break
        assert len(candidates) == k

        miss_rank = 0
        for index in range(k):
            if not mask_local[index]:
                value = candidates[miss_rank]
                miss_rank += 1
                hit_local[index] = value
                bitmap[value] = 1

        remaining = [value for value in lru_local if bitmap[value] == 0]
        assert len(remaining) == k
        output_hit.append(hit_local)
        output_lru.append(hit_local + remaining)

    return (
        torch.tensor(output_lru, dtype=torch.int16),
        torch.tensor(output_hit, dtype=torch.int16),
    )


@pytest.mark.parametrize("k", [1, 3, 127, 128, 129, 257])
@pytest.mark.parametrize("tile_length", [128, 256])
def test_staged_algorithm_matches_reference(k, tile_length):
    lru, hit = make_inputs(4, k, seed=1000 + k + tile_length)
    expected = build_lru_plan_reference(lru, hit)
    actual = build_lru_plan_staged(lru, hit, tile_length)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("all_miss,no_miss", [(True, False), (False, True)])
def test_staged_algorithm_miss_extremes(all_miss, no_miss):
    lru, hit = make_inputs(
        3, 129, seed=2048, all_miss=all_miss, no_miss=no_miss
    )
    expected = build_lru_plan_reference(lru, hit)
    actual = build_lru_plan_staged(lru, hit, tile_length=128)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 3, 41])
@pytest.mark.parametrize(
    "k", [1, 3, 7, 8, 9, 31, 32, 33, 127, 128, 129, 257, 2048]
)
def test_row_fused_algorithm_matches_reference(batch_size, k):
    lru, hit = make_inputs(batch_size, k, seed=32768 + batch_size + k)
    hit_mask = hit.ne(-1)
    expected = build_lru_plan_reference(lru, hit, hit_mask)
    actual = build_lru_plan_row_fused(lru, hit, hit_mask)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("all_miss,no_miss", [(True, False), (False, True)])
def test_row_fused_algorithm_miss_extremes(all_miss, no_miss):
    lru, hit = make_inputs(
        3, 129, seed=65536, all_miss=all_miss, no_miss=no_miss
    )
    hit_mask = hit.ne(-1)
    expected = build_lru_plan_reference(lru, hit, hit_mask)
    actual = build_lru_plan_row_fused(lru, hit, hit_mask)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_fully_parallel_target_shape_matches_reference():
    batch_size = 1
    k = 2048
    core_num = 40
    hit_tile_length = make_parallel_tile_length(k, batch_size, core_num)
    lru_tile_length = make_parallel_tile_length(2 * k, batch_size, core_num)
    assert hit_tile_length == 48
    assert lru_tile_length == 96
    assert (k + hit_tile_length - 1) // hit_tile_length >= core_num
    assert (2 * k + lru_tile_length - 1) // lru_tile_length >= core_num

    lru, hit = make_inputs(batch_size, k, seed=4096)
    hit_mask = hit.ne(-1)
    expected = build_lru_plan_reference(lru, hit, hit_mask)
    actual = build_lru_plan_parallel_staged(
        lru, hit, hit_mask, hit_tile_length, lru_tile_length
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 16, 41])
@pytest.mark.parametrize("k", [1, 3, 127, 2048])
def test_row_owned_hit_bitmap_and_counts(batch_size, k):
    _, hit = make_inputs(batch_size, k, seed=12288 + batch_size + k)
    tile_length = make_parallel_tile_length(k, batch_size, core_num=40)
    bitmaps, counts = build_hit_row_local(hit, tile_length)
    for batch in range(batch_size):
        expected_hits = {value for value in hit[batch].tolist() if value != -1}
        actual_hits = {index for index, value in enumerate(bitmaps[batch]) if value}
        assert actual_hits == expected_hits
        assert sum(counts[batch]) == k - len(expected_hits)


@pytest.mark.parametrize("k", [1, 3, 31, 32, 127, 2048])
@pytest.mark.parametrize("tile_length", [16, 32, 48])
def test_sparse_hit_bitmap_has_disjoint_stores(k, tile_length):
    lru, hit = make_inputs(3, k, seed=16384 + k + tile_length)
    bitmaps, counts = build_hit_bitmap_sparse(hit, tile_length)
    for batch in range(hit.size(0)):
        expected_ids = {value for value in hit[batch].tolist() if value != -1}
        actual_ids = {index for index, value in enumerate(bitmaps[batch]) if value}
        assert actual_ids == expected_ids
        assert sum(counts[batch]) == k - len(expected_ids)


@pytest.mark.parametrize("k", [1, 3, 127, 128, 129, 257, 2048])
def test_fully_parallel_pipeline_random_shapes(k):
    batch_size = 3
    core_num = 40
    hit_tile_length = make_parallel_tile_length(k, batch_size, core_num)
    lru_tile_length = make_parallel_tile_length(2 * k, batch_size, core_num)
    lru, hit = make_inputs(batch_size, k, seed=8192 + k)
    hit_mask = hit.ne(-1)
    expected = build_lru_plan_reference(lru, hit, hit_mask)
    actual = build_lru_plan_parallel_staged(
        lru, hit, hit_mask, hit_tile_length, lru_tile_length
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def is_npu_available():
    if torch_npu is None:
        return False
    try:
        return torch.npu.is_available()
    except Exception:
        return False


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
                "ascend_kernel is not installed and no local library was found"
            )
        torch.ops.load_library(libraries[0])


@pytest.fixture(scope="module")
def npu_device():
    if not is_npu_available():
        pytest.skip("NPU is not available")
    load_operator_library()
    return torch.device("npu:0")


@pytest.mark.parametrize(
    "batch_size,k,seed",
    [
        (1, 1, 1),
        (2, 3, 2),
        (3, 127, 3),
        (2, 128, 4),
        (4, 129, 5),
        (2, 257, 6),
        (1, 2048, 7),
    ],
)
def test_random_valid_inputs(npu_device, batch_size, k, seed):
    lru, hit = make_inputs(batch_size, k, seed)
    hit_mask = hit.ne(-1)
    expected_lru, expected_hit = build_lru_plan_reference(lru, hit, hit_mask)
    hit_npu = hit.to(npu_device)
    hit_mask_npu = hit_mask.to(npu_device)
    hit_mask_before = hit_mask_npu.clone()

    actual_lru = torch.ops.npu.build_lru_plan(
        lru.to(npu_device), hit_npu, hit_mask_npu
    )

    assert actual_lru.dtype == torch.int16
    assert hit_npu.dtype == torch.int16
    torch.testing.assert_close(actual_lru.cpu(), expected_lru, rtol=0, atol=0)
    torch.testing.assert_close(hit_npu.cpu(), expected_hit, rtol=0, atol=0)
    torch.testing.assert_close(actual_lru[:, :k].cpu(), hit_npu.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(hit_mask_npu, hit_mask_before, rtol=0, atol=0)


@pytest.mark.parametrize("all_miss,no_miss", [(True, False), (False, True)])
def test_miss_extremes(npu_device, all_miss, no_miss):
    lru, hit = make_inputs(3, 129, 11, all_miss=all_miss, no_miss=no_miss)
    hit_mask = hit.ne(-1)
    expected_lru, expected_hit = build_lru_plan_reference(lru, hit, hit_mask)
    hit_npu = hit.to(npu_device)
    actual_lru = torch.ops.npu.build_lru_plan(
        lru.to(npu_device), hit_npu, hit_mask.to(npu_device)
    )
    torch.testing.assert_close(actual_lru.cpu(), expected_lru, rtol=0, atol=0)
    torch.testing.assert_close(hit_npu.cpu(), expected_hit, rtol=0, atol=0)


def test_noncontiguous_read_only_inputs(npu_device):
    lru, hit = make_inputs(3, 65, 21)
    hit_mask = hit.ne(-1)
    lru_noncontiguous = lru.to(npu_device).t().contiguous().t()
    hit_mask_noncontiguous = hit_mask.to(npu_device).t().contiguous().t()
    hit_npu = hit.to(npu_device)
    assert not lru_noncontiguous.is_contiguous()
    assert not hit_mask_noncontiguous.is_contiguous()

    expected_lru, expected_hit = build_lru_plan_reference(lru, hit, hit_mask)
    actual_lru = torch.ops.npu.build_lru_plan(
        lru_noncontiguous, hit_npu, hit_mask_noncontiguous
    )
    torch.testing.assert_close(actual_lru.cpu(), expected_lru, rtol=0, atol=0)
    torch.testing.assert_close(hit_npu.cpu(), expected_hit, rtol=0, atol=0)


def test_rejects_noncontiguous_hit(npu_device):
    lru, hit = make_inputs(3, 65, 22)
    hit_mask = hit.ne(-1)
    hit_noncontiguous = hit.to(npu_device).t().contiguous().t()
    assert not hit_noncontiguous.is_contiguous()

    with pytest.raises(RuntimeError, match="hit must be contiguous"):
        torch.ops.npu.build_lru_plan(
            lru.to(npu_device), hit_noncontiguous, hit_mask.to(npu_device)
        )


def test_rejects_wrong_dtype(npu_device):
    lru, hit = make_inputs(1, 3, 31)
    hit_mask = hit.ne(-1)
    with pytest.raises(RuntimeError, match="lru and hit must be int16"):
        torch.ops.npu.build_lru_plan(
            lru.to(device=npu_device, dtype=torch.int64),
            hit.to(npu_device),
            hit_mask.to(npu_device),
        )


def test_rejects_int16_id_overflow(npu_device):
    k = 16385
    lru = torch.zeros((1, 2 * k), dtype=torch.int16, device=npu_device)
    hit = torch.full((1, k), -1, dtype=torch.int16, device=npu_device)
    hit_mask = torch.zeros((1, k), dtype=torch.bool, device=npu_device)
    with pytest.raises(RuntimeError, match="int16 IDs require K <= 16384"):
        torch.ops.npu.build_lru_plan(lru, hit, hit_mask)


def test_rejects_wrong_mask(npu_device):
    lru, hit = make_inputs(2, 9, 32)
    with pytest.raises(RuntimeError, match="hit_mask must be a bool"):
        torch.ops.npu.build_lru_plan(
            lru.to(npu_device), hit.to(npu_device), hit.to(npu_device)
        )

    with pytest.raises(RuntimeError, match="same shape as hit"):
        torch.ops.npu.build_lru_plan(
            lru.to(npu_device),
            hit.to(npu_device),
            hit.ne(-1)[:, :-1].to(npu_device),
        )


def run_simple_test():
    """Run exact CPU-vs-NPU checks without invoking the pytest runner."""
    if not is_npu_available():
        print("NPU is not available; build_lru_plan device test was not run.")
        return False

    try:
        load_operator_library()
        device = torch.device("npu:0")
        cases = [
            (1, 1, 101, False, False),
            (2, 3, 102, False, False),
            (2, 129, 103, False, False),
            (2, 257, 104, False, False),
            (3, 129, 105, True, False),
            (3, 129, 106, False, True),
        ]

        for batch_size, k, seed, all_miss, no_miss in cases:
            lru, hit = make_inputs(
                batch_size,
                k,
                seed,
                all_miss=all_miss,
                no_miss=no_miss,
            )
            hit_mask = hit.ne(-1)
            expected_lru, expected_hit = build_lru_plan_reference(
                lru, hit, hit_mask
            )
            hit_npu = hit.to(device)
            actual_lru = torch.ops.npu.build_lru_plan(
                lru.to(device), hit_npu, hit_mask.to(device)
            )
            torch.testing.assert_close(
                actual_lru.cpu(), expected_lru, rtol=0, atol=0
            )
            torch.testing.assert_close(
                hit_npu.cpu(), expected_hit, rtol=0, atol=0
            )
            print(f"PASS: B={batch_size}, K={k}, all_miss={all_miss}, no_miss={no_miss}")

        print("build_lru_plan correctness test PASSED")
        return True
    except Exception as error:
        print(f"build_lru_plan correctness test FAILED: {error}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    raise SystemExit(0 if run_simple_test() else 1)
