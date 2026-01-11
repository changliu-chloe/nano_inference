import torch
import triton

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version (PyTorch): {torch.version.cuda}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"Triton Version: {triton.__version__}")
print("GPU count:", torch.cuda.device_count())

# 如果输出的 CUDA Version 显示 12.8，且 is_available 为 True，则配置成功。

# import torch
# import triton
import triton.language as tl

def check_environment():
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # 1. 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("❌ Error: CUDA is not available via PyTorch.")
        return
    
    print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version (Torch): {torch.version.cuda}")
    
    # 2. 检查 Triton 编译是否正常
    # 这里的目的是测试 ptxas 编译器是否能正常工作
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    try:
        x = torch.ones(10, device='cuda')
        y = torch.ones(10, device='cuda')
        output = torch.zeros(10, device='cuda')
        grid = lambda meta: (triton.cdiv(10, meta['BLOCK_SIZE']),)
        
        # 触发编译
        add_kernel[grid](x, y, output, 10, BLOCK_SIZE=16)
        
        if torch.allclose(output, x + y):
            print("✅ Triton Compilation: SUCCESS (Kernel ran successfully)")
        else:
            print("❌ Triton Calculation Error.")
    except Exception as e:
        print(f"❌ Triton Compilation FAILED.\nError: {e}")
        
import sys
if __name__ == "__main__":
    check_environment()