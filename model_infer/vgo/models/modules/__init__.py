import os

import torch
import torch.nn as nn

from vgo.utils.accel import is_npu

from .distributed_ops import AllReduceGradBackward

# apply_rope_v2 的正确性经过了 tests/test_rope_ops.py 的检验
# 如果你放心的话，可以将下面的注释打开，去掉原本的 triton_apply_rope
# 可以带来一定性能增长，在 Head Number 较多时比较明显
# from .experiments.rope_ops_v2 import apply_rope_v2 as triton_apply_rope
from .rope_ops import apply_rope as triton_apply_rope


class Function(nn.Module):
    def __init__(self, implementation: str = "basic", **implementation_params):
        super().__init__()
        self.implementation = implementation
        self.implementation_params = implementation_params
        self._setup_forward()

    def _setup_forward(self):
        if self.implementation == "basic":
            self.forward = self._forward
        elif self.implementation == "compiled":
            if os.environ.get("VGO_DISABLE_TORCH_COMPILE", "0") == "1":
                self.forward = self._forward
                return
            compiled_fn = torch.compile(
                self._forward,
                mode=self.implementation_params.pop("mode", "max-autotune-no-cudagraphs"),
                dynamic=self.implementation_params.pop("dynamic", True),
                **self.implementation_params,
            )
            self.forward = compiled_fn
        else:
            create_fn = getattr(self, f"{self.implementation}_forward", None)
            assert create_fn is not None, f"Unknown implementation: {self.implementation}"
            self.forward = create_fn(**self.implementation_params)

    def _forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement _forward")


class ScaleAndResidualFunction(Function):
    def _forward(self, x: torch.Tensor, scale: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x * scale + residual


scale_add_residual = ScaleAndResidualFunction(implementation="compiled")


class LayerNormAndScaleShiftFunction(Function):
    def _forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, (x.size(-1),)) * (scale + 1) + shift


layernorm_and_scale_shift = LayerNormAndScaleShiftFunction(implementation="compiled")


class ApplyRopeFunction(Function):
    def _forward(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
        # 将 num_heads 和 seq_len 的维度交换回原函数的处理顺序
        xq = xq.transpose(1, 2)  # BSHD -> [batch, num_heads, seq_len, head_dim]
        xk = xk.transpose(1, 2)

        # 将 head_dim 拆分为复数部分(实部和虚部)
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)

        # 应用旋转位置编码(复数乘法)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

        # 恢复张量形状并转置回目标维度顺序
        xq_out = xq_out.reshape(*xq.shape).type_as(xq).transpose(1, 2)
        xk_out = xk_out.reshape(*xk.shape).type_as(xk).transpose(1, 2)

        return xq_out, xk_out

    def triton_forward(self):
        from vgo.models.modules.apply_rope_op import LigerRopePaperFunction

        def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, inplace=True):
            # 假设xq, xk输入是BSHD
            cos = freqs_cis[:, :, 0, :, 0, 0]
            sin = freqs_cis[:, :, 0, :, 1, 0]
            if not inplace:
                xq = xq.clone()
                xk = xk.clone()
            return LigerRopePaperFunction.apply(xq, xk, cos, sin)

        return apply_rope


apply_rope = ApplyRopeFunction(implementation="basic")


def _test_rope():
    from einops import repeat

    from vgo.models.transformers.layers import EmbedND

    pe_embedder = EmbedND(dim=3072 // 24, theta=10_000, axes_dim=[16, 56, 56])

    h, w, b = 1024 // 8, 1024 // 8, 2

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=b)
    txt_ids = torch.zeros(b, 256, 3)
    ids = torch.cat((img_ids, txt_ids), dim=1)
    pe = pe_embedder(ids)
    print(pe.shape)
    # pe = pe.to("cuda")

    shape = [2, 4352, 24, 128]
    xq, xk = torch.randn(*shape, device="cpu", dtype=torch.bfloat16)

    ApplyRopeFunction(implementation="basic")(xq, xk, pe)


class RMSNormFunction(Function):
    def _forward(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        return (x * rrms).to(dtype=dtype) * weight

    def liger_kernel_forward(self):
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction

        def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
            return LigerRMSNormFunction.apply(x, weight, eps, 0.0, "llama", True)

        return rmsnorm

    def optimus_forward(self):
        import optimus  # noqa: F401

        def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
            return torch.ops.Optimus.rms_norm(x.contiguous(), weight, eps)

        return rmsnorm

    def npu_forward(self):
        import torch_npu

        def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
            return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]

        return rmsnorm


class LayerNormAutoCast(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias
        if self.weight is not None:
            if self.weight.dtype != torch.float32:
                weight = self.weight.to(torch.float32)
                if self.bias:
                    bias = self.bias.to(torch.float32)

        if input.dtype != torch.float32:
            input = input.to(torch.float32)

        return nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        weight_name="scale",
        implementation: str = "liger_kernel",
        **implementation_params,
    ):
        super().__init__()
        if elementwise_affine:
            self.weight_name = weight_name
            self.register_parameter(weight_name, nn.Parameter(torch.ones(dim)))

        assert elementwise_affine

        self.eps = eps
        if implementation == "liger_kernel":
            if is_npu():
                implementation = "npu"
            else:
                try:
                    import liger_kernel  # noqa: F401
                except Exception:
                    implementation = "basic"
        self.rmsnorm = RMSNormFunction(implementation=implementation, **implementation_params)

        self.is_shard = False

    def set_device_mesh(self, device_mesh):
        self.is_shard = True
        self.device_mesh = device_mesh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = getattr(self, self.weight_name)

        if weight.dtype != torch.float32:
            weight = weight.to(torch.float32)

        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)

        if not self.is_shard:
            out = self.rmsnorm(x, weight, self.eps)
        else:
            out = self.rmsnorm(x, AllReduceGradBackward.apply(weight, self.device_mesh), self.eps)

        return out


def _test():  # noqa: C901
    import time

    test_cases = [
        {
            "name": "ScaleAndResidual",
            "cls": ScaleAndResidualFunction,
            "correctness_shapes": [(4, 4), (2, 3, 4)],
            "performance_shapes": [(4096, 4096), (128, 128)],
            "input_generator": lambda shape: (
                torch.randn(shape),
                torch.tensor(0.5),
                torch.randn(shape),
            ),
        },
        {
            "name": "LayerNormAndScaleShift",
            "cls": LayerNormAndScaleShiftFunction,
            "correctness_shapes": [(4, 4), (2, 3, 4), (1, 5)],
            "performance_shapes": [(32, 64, 256), (2, 4096, 3072)],
            "input_generator": lambda shape: (
                torch.randn(shape),
                torch.randn(shape[-1:]),  # scale as vector
                torch.randn(shape[-1:]),  # shift as vector
            ),
            "extra_correctness_generators": [
                # 添加标量测试用例
                lambda shape: (
                    torch.randn(shape),
                    torch.tensor(0.5),  # scalar scale
                    torch.tensor(0.1),  # scalar shift
                )
            ],
        },
    ]

    # 正确性验证
    print("=== Correctness Tests ===")
    for case in test_cases:
        print(f"\nTesting {case['name']}...")
        # 初始化基础实现和编译实现
        basic = case["cls"](impl="basic")
        compiled = case["cls"](impl="compiled")

        # 标准正确性测试
        for shape in case["correctness_shapes"]:
            # 使用主输入生成器
            inputs = case["input_generator"](shape)
            # 转换为float32以确保精度
            inputs = [i.float() for i in inputs]

            # 基础实现结果
            torch.manual_seed(42)
            out_basic = basic(*inputs)

            # 编译实现结果
            torch.manual_seed(42)
            out_compiled = compiled(*inputs)

            # 验证一致性
            assert torch.allclose(out_basic, out_compiled, atol=1e-5), f"Correctness failed for shape {shape}"
            print(f"Shape {shape}: Vector test passed")

        # 额外测试用例（如LayerNorm的标量情况）
        if "extra_correctness_generators" in case:
            for gen in case["extra_correctness_generators"]:
                for shape in case["correctness_shapes"]:
                    inputs = gen(shape)
                    inputs = [i.float() for i in inputs]

                    torch.manual_seed(42)
                    out_basic = basic(*inputs)

                    torch.manual_seed(42)
                    out_compiled = compiled(*inputs)

                    assert torch.allclose(out_basic, out_compiled, atol=1e-5), (
                        f"Extra correctness failed for shape {shape}"
                    )
                    print(f"Shape {shape}: Scalar test passed")

    # 性能测试
    print("\n=== Performance Tests ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device.upper()}")

    def measure_performance(func, inputs, device, warmup=5, runs=20):
        # 确保输入在正确设备上
        inputs = [i.to(device) for i in inputs]
        # 预热
        for _ in range(warmup):
            func(*inputs)
        # 计时
        torch.cuda.synchronize() if "cuda" in device else None
        start = time.time()
        for _ in range(runs):
            func(*inputs)
        torch.cuda.synchronize() if "cuda" in device else None
        return (time.time() - start) / runs

    for case in test_cases:
        print(f"\nTesting {case['name']}:")
        basic = case["cls"](impl="basic").to(device)
        compiled = case["cls"](impl="compiled").to(device)

        for shape in case["performance_shapes"]:
            # 生成输入数据
            inputs = case["input_generator"](shape)
            inputs = [i.to(device) for i in inputs]

            # 基础版性能
            basic_time = measure_performance(basic, inputs, device)
            # 编译版性能（稳定状态）
            compiled_time = measure_performance(compiled, inputs, device)

            print(f"Shape {shape}:")
            print(f"  Basic: {basic_time:.5f}s/iter")
            print(f"  Compiled: {compiled_time:.5f}s/iter")
            print(f"  Speedup: {basic_time / compiled_time:.1f}x")

    rmsnorm_default = RMSNormFunction(implementation="basic")
    rmsnorm_liger = RMSNormFunction(implementation="liger_kernel")

    # 配置测试参数
    test_cases = [
        # LLaMA风格（batch_size, seq_len, hidden_size）
        {"name": "LLaMA-7B", "shape": (32, 2048, 4096)},  # 标准7B模型参数
        {"name": "LLaMA-70B", "shape": (16, 2048, 8192)},  # 大模型参数
        {"name": "LLaMA-长序列", "shape": (8, 16384, 4096)},  # 长上下文场景
        # DiT风格（batch_size, num_patches, embed_dim）
        {"name": "DiT-XL", "shape": (64, 256, 1152)},  # 标准DiT-XL参数
        {"name": "DiT-HD", "shape": (32, 1024, 1024)},  # 高分辨率场景
        {"name": "DiT-3D", "shape": (16, 512, 2048)},  # 三维数据场景
        # 混合场景
        {"name": "高通道", "shape": (128, 128, 8192)},  # 极高通道数场景
        {"name": "小批次", "shape": (1, 4096, 5120)},  # 小批次推理场景
    ]

    # 定义CUDA事件计时器
    def measure_time(func, x, weight, eps, num_repeats=100):
        # 预热
        for _ in range(10):
            _ = func(x, weight, eps)

        # 创建CUDA事件
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # 确保初始同步
        torch.cuda.synchronize()

        # 记录开始时间
        start_event.record()
        for _ in range(num_repeats):
            _ = func(x, weight, eps)
        end_event.record()

        # 等待事件完成
        torch.cuda.synchronize()

        # 计算单次运行时间（毫秒）
        return start_event.elapsed_time(end_event) / num_repeats

    for case in test_cases:
        print(f"\n{'=' * 40}\n测试用例：{case['name']} {case['shape']}\n{'=' * 40}")

        # 生成测试数据
        x = torch.randn(*case["shape"], device="cuda")
        weight = torch.ones(case["shape"][-1], device="cuda")
        eps = 1e-6

        # 正确性验证
        with torch.no_grad():
            out_default = rmsnorm_default(x, weight, eps)
            out_liger = rmsnorm_liger(x, weight, eps)

        # 使用相对误差验证，适应不同规模的数据
        diff = (out_default - out_liger).abs().max()
        assert diff < 1e-4, f"正确性验证失败，最大差异：{diff:.2e}"

        # 预热（避免冷启动误差）
        for _ in range(10):
            _ = rmsnorm_default(x, weight, eps)
            _ = rmsnorm_liger(x, weight, eps)

        # 默认实现
        time_default = measure_time(rmsnorm_default, x, weight, eps)
        time_liger = measure_time(rmsnorm_liger, x, weight, eps)

        # 打印结果
        print(f"默认实现：{time_default:.4f} ms")
        print(f"Liger实现：{time_liger:.4f} ms")
        print(f"加速比：{time_default / time_liger:.2f}x")

        # 显存清理（针对大尺寸测试用例）
        del x, weight, out_default, out_liger
        torch.cuda.empty_cache()


__all__ = [
    "LayerNormAutoCast",
    "RMSNorm",
    "apply_rope",
    "layernorm_and_scale_shift",
    "scale_add_residual",
    "triton_apply_rope",
]

if __name__ == "__main__":
    _test_rope()
