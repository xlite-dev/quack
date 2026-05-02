import argparse
import math
import time

import torch
from triton.testing import do_bench

from quack.hadamard import hadamard_transform, hadamard_transform_ref

try:
    from fast_hadamard_transform import hadamard_transform as fast_hadamard_transform
except ImportError:
    fast_hadamard_transform = None


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

TOLERANCES = {
    torch.bfloat16: (1e-1, 2e-2),
    torch.float16: (3e-2, 2e-2),
    torch.float32: (1e-4, 1e-4),
}


def _effective_bandwidth_gbps(x: torch.Tensor, latency_ms: float) -> float:
    bytes_moved = 2 * x.numel() * x.element_size()
    return bytes_moved / (latency_ms / 1000.0) / 1e9


def _bench(name, fn, x, warmup, rep):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    time.sleep(0.2)
    latency_ms = do_bench(fn, warmup=warmup, rep=rep)
    print(
        f"{name:>24}: {latency_ms:.4f} ms, "
        f"{_effective_bandwidth_gbps(x, latency_ms):.1f} effective GB/s"
    )
    return latency_ms


def run_hadamard(M, N, dtype, scale, warmup, rep, include_torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark")
    if N > 32768:
        raise ValueError("QuACK Hadamard currently supports N <= 32768")

    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"dtype: {dtype}, scale: {scale}")

    out = hadamard_transform(x, scale=scale)
    if fast_hadamard_transform is not None:
        out_ref = fast_hadamard_transform(x, scale)
        ref_name = "fast-hadamard-transform"
    else:
        out_ref = hadamard_transform_ref(x, scale=scale)
        ref_name = "torch reference"
    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    print(f"Correctness: compared QuACK against {ref_name}")

    _bench("QuACK CuTe-DSL", lambda: hadamard_transform(x, scale=scale), x, warmup, rep)

    if fast_hadamard_transform is not None:
        _bench(
            "fast-hadamard-transform",
            lambda: fast_hadamard_transform(x, scale),
            x,
            warmup,
            rep,
        )
    else:
        print("fast-hadamard-transform: not installed")

    _bench("torch.clone lower bound", lambda: torch.clone(x), x, warmup, rep)

    if include_torch:
        _bench("torch FWHT reference", lambda: hadamard_transform_ref(x, scale=scale), x, 3, 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Hadamard transform")
    parser.add_argument("--M", default=8192, type=int)
    parser.add_argument("--N", default=4096, type=int)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="bfloat16")
    parser.add_argument("--scale", default=None, type=float)
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--include-torch", action="store_true")
    args = parser.parse_args()

    dtype = DTYPES[args.dtype]
    scale = args.scale
    if scale is None:
        scale = 1.0 / math.sqrt(1 << (args.N - 1).bit_length())

    run_hadamard(
        args.M,
        args.N,
        dtype,
        scale,
        args.warmup_iterations,
        args.iterations,
        args.include_torch,
    )
