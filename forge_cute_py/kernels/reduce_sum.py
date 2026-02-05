"""
Parallel reduction kernel using CuTe DSL.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import const_expr


class ReduceSum:
    """Parallel reduction sum operation using CuTe DSL."""

    def __init__(self, dtype: type, block_size: int = 256):
        """
        Initialize reduce sum kernel.

        Args:
            dtype: CUTLASS numeric type (Float16, Float32, BFloat16)
            block_size: Number of threads per block (default: 256)
        """
        self.dtype = dtype
        self.block_size = block_size

    @cute.jit
    def __call__(
        self,
        input: cute.Tensor,
        output: cute.Tensor,
        dim: int,
        stream: cuda.CUstream = None,
    ):
        """
        Execute reduction sum.

        Args:
            input: Input tensor of shape (M, N)
            output: Output tensor of shape (M,) if dim=1 or (N,) if dim=0
            dim: Dimension to reduce (0 or 1)
            stream: CUDA stream
        """
        M, N = input.shape
        
        if dim == 1 or dim == -1:
            # Reduce along columns: (M, N) -> (M,)
            grid_size = cute.ceil_div(M, 1)
            self.kernel_reduce_cols(input, output).launch(
                grid=[grid_size, 1, 1],
                block=[self.block_size, 1, 1],
                stream=stream,
            )
        elif dim == 0:
            # Reduce along rows: (M, N) -> (N,)
            grid_size = cute.ceil_div(N, 1)
            self.kernel_reduce_rows(input, output).launch(
                grid=[grid_size, 1, 1],
                block=[self.block_size, 1, 1],
                stream=stream,
            )
        else:
            raise ValueError("dim must be 0, 1, or -1")

    @cute.kernel
    def kernel_reduce_cols(
        self,
        input: cute.Tensor,
        output: cute.Tensor,
    ):
        """
        Reduce columns: sum along axis 1.
        Each block handles one row, threads cooperatively reduce.
        
        Strategy:
        1. Each thread loads elements from its row
        2. Sequential addressing reduction in shared memory
        3. Warp-level reduction for final step
        4. Thread 0 writes result
        """
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        
        M, N = input.shape
        block_size = const_expr(self.block_size)
        
        if bidx >= M:
            return
        
        # Allocate shared memory for reduction
        smem = cutlass.utils.SmemAllocator()
        sdata = smem.allocate_array(self.dtype, block_size, byte_alignment=16)
        
        # Initialize accumulator
        sum_val = self.dtype(0)
        
        # Each thread accumulates multiple elements
        col = tidx
        while col < N:
            sum_val = sum_val + input[bidx, col]
            col = col + block_size
        
        # Store thread's partial sum to shared memory
        sdata[tidx] = sum_val
        cute.arch.sync_threads()
        
        # Parallel reduction in shared memory (sequential addressing)
        stride = block_size // 2
        while stride > 32:
            if tidx < stride:
                sdata[tidx] = sdata[tidx] + sdata[tidx + stride]
            cute.arch.sync_threads()
            stride = stride // 2
        
        # Final warp reduction (no sync needed within warp)
        if tidx < 32:
            # Manually unroll last 6 iterations for warp
            #if block_size >= 64:
            #    sdata[tidx] = sdata[tidx] + sdata[tidx + 32]
            if tidx < 16:
                sdata[tidx] = sdata[tidx] + sdata[tidx + 16]
            if tidx < 8:
                sdata[tidx] = sdata[tidx] + sdata[tidx + 8]
            if tidx < 4:
                sdata[tidx] = sdata[tidx] + sdata[tidx + 4]
            if tidx < 2:
                sdata[tidx] = sdata[tidx] + sdata[tidx + 2]
            if tidx < 1:
                sdata[tidx] = sdata[tidx] + sdata[tidx + 1]
        
        # Thread 0 writes final result
        if tidx == 0:
            output[bidx] = sdata[0]

    @cute.kernel
    def kernel_reduce_rows(
        self,
        input: cute.Tensor,
        output: cute.Tensor,
    ):
        """
        Reduce rows: sum along axis 0.
        Each block handles one column, threads cooperatively reduce.
        """
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        
        M, N = input.shape
        block_size = const_expr(self.block_size)
        
        if bidx >= N:
            return
        
        # Allocate shared memory for reduction
        smem = cutlass.utils.SmemAllocator()
        sdata = smem.allocate_array(self.dtype, block_size, byte_alignment=16)
        
        # Initialize accumulator
        sum_val = self.dtype(0)
        
        # Each thread accumulates multiple elements
        row = tidx
        while row < M:
            sum_val = sum_val + input[row, bidx]
            row = row + block_size
        
        # Store thread's partial sum to shared memory
        sdata[tidx] = sum_val
        cute.arch.sync_threads()
        
        # Parallel reduction in shared memory (sequential addressing)
        stride = block_size // 2
        while stride > 32:
            if tidx < stride:
                sdata[tidx] = sdata[tidx] + sdata[tidx + stride]
            cute.arch.sync_threads()
            stride = stride // 2
        
        # Final warp reduction (no sync needed within warp)
        if tidx < 32:
            # Manually unroll last 6 iterations for warp
            #if block_size >= 64:
            #    sdata[tidx] = sdata[tidx] + sdata[tidx + 32]
            if tidx < 16:
                sdata[tidx] = sdata[tidx] + sdata[tidx + 16]
            if tidx < 8:
                sdata[tidx] = sdata[tidx] + sdata[tidx + 8]
            if tidx < 4:
                sdata[tidx] = sdata[tidx] + sdata[tidx + 4]
            if tidx < 2:
                sdata[tidx] = sdata[tidx] + sdata[tidx + 2]
            if tidx < 1:
                sdata[tidx] = sdata[tidx] + sdata[tidx + 1]
        
        # Thread 0 writes final result
        if tidx == 0:
            output[bidx] = sdata[0]