import argparse
import torch
from forge_cute_py.ops import reduce_sum
from forge_cute_py.util.bench import do_bench, estimate_bandwidth, summarize_times

def main():
    parser = argparse.ArgumentParser(description="Benchmark reduce_sum")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=-1, choices=[-1, 0, 1])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--compile-ref", action="store_true")
    parser.add_argument("--variant", choices=["basic", "shfl"], default="shfl")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmarking")
    
    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.m, args.n, device="cuda", dtype=dtype)
    
    def fn():
        return reduce_sum(x, dim=args.dim, variant=args.variant)
    
    times = do_bench(fn, warmup=args.warmup, rep=args.iterations)
    stats = summarize_times(times)
    
    input_bytes = x.numel() * x.element_size()
    output_size = args.m if args.dim in (-1, 1) else args.n
    output_bytes = output_size * x.element_size()
    bytes_moved = input_bytes + output_bytes
    bw = estimate_bandwidth(bytes_moved, stats["p50_ms"])
    
    print(f"reduce_sum p50: {stats['p50_ms']:.4f} ms, BW: {bw:.2f} GB/s")
    
    ref = lambda: x.sum(dim=args.dim)
    if args.compile_ref and hasattr(torch, "compile"):
        ref = torch.compile(ref, fullgraph=True)
    
    ref_times = do_bench(ref, warmup=args.warmup, rep=args.iterations)
    ref_stats = summarize_times(ref_times)
    ref_bw = estimate_bandwidth(bytes_moved, ref_stats["p50_ms"])
    
    print(f"reference p50: {ref_stats['p50_ms']:.4f} ms, BW: {ref_bw:.2f} GB/s")

if __name__ == "__main__":
    main()