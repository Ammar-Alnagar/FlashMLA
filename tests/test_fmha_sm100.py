import random

import torch
from torch.utils.checkpoint import checkpoint
import triton

from flash_mla import flash_attn_varlen_func
from kernelkit import check_is_allclose

def get_free_gpu_memory():
    try:
        return torch.cuda.mem_get_info()[0]
    except Exception:
        return None


def estimate_dense_memory_bytes(b, s_q, s_k, h, d, dv, has_bwd):
    """Peak working set of one dense-attention testcase.

    `sdpa()` above materialises the [b * h, s_q, s_k] score matrix in fp32, then the softmax result
    in fp32 on top of it, and the backward pass keeps both alive. For the upstream 4096/8192 shapes
    that is tens of gigabytes, which no consumer GPU has; such cases are skipped rather than
    reported as failures.
    """
    scores = b * h * s_q * s_k * 4
    qkv = b * h * (d + 2 * dv) * 2 * 4          # bf16 inputs plus their fp32 copies
    total = scores * 2 + qkv
    if has_bwd:
        total += scores * 2                     # saved activations for the backward reference
    return total


def fits_in_gpu_memory(b, s_q, s_k, h, d, dv, has_bwd, fraction=0.5):
    free = get_free_gpu_memory()
    if free is None:
        return True
    return estimate_dense_memory_bytes(b, s_q, s_k, h, d, dv, has_bwd) <= free * fraction


def get_window_size(causal, window):
    if window > 0:
        window_size = (window - 1, 0) if causal else (window - 1, window - 1)
    else:
        window_size = (-1, -1)
    return window_size


def get_attn_bias(s_q, s_k, causal, window):
    attn_bias = torch.zeros(s_q, s_k, dtype=torch.float32)
    if causal:
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if window > 0:
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q - window)
        attn_bias.masked_fill_(temp_mask, float("-inf"))
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q + window - 1)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    return attn_bias


def sdpa(query, key, value, attn_bias, softmax_scale=None):
    query = query.float().transpose(-3, -2)
    key = key.float().transpose(-3, -2)
    value = value.float().transpose(-3, -2)
    key = key.repeat_interleave(h // h_k, dim=-3)
    value = value.repeat_interleave(h // h_k, dim=-3)
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** (-0.5)
    attn_weight = (query @ key.transpose(-2, -1)) * softmax_scale
    attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight.to(query.dtype) @ value, lse


def sdpa_checkpoint(*args, **kwargs):
    return checkpoint(sdpa, *args, use_reentrant=False, **kwargs)


def test_flash_attention(b, mean_sq, mean_sk, varlen, h, h_k, d, dv, causal, window, has_bwd, check_correctness: bool = True):
    print(f"{b=}, {mean_sq=}, {mean_sk=}, {varlen=}, {h=}, {h_k=}, {d=}, {dv=}, {causal=}, {has_bwd=}, {check_correctness=}")
    torch.manual_seed(0)
    random.seed(0)

    seqlens_q = torch.full((b,), mean_sq, dtype=torch.int32)
    seqlens_k = torch.full((b,), mean_sk, dtype=torch.int32)

    if varlen:
        for i in range(b):
            seqlens_q[i] = max(random.normalvariate(mean_sq, mean_sq / 2), 1)
        for i in range(b):
            seqlens_k[i] = max(random.normalvariate(mean_sk, mean_sk / 2), seqlens_q[i].item())
    cu_seqlens_q = torch.cumsum(torch.nn.functional.pad(seqlens_q, (1, 0)), 0, dtype=torch.int32)
    cu_seqlens_k = torch.cumsum(torch.nn.functional.pad(seqlens_k, (1, 0)), 0, dtype=torch.int32)
    total_q = seqlens_q.sum().item()
    total_k = seqlens_k.sum().item()
    max_seqlen_q = seqlens_q.max().item()
    max_seqlen_k = seqlens_k.max().item()
    total_attn_compute = sum([(get_attn_bias(seqlens_q[i].item(), seqlens_k[i].item(),
                             causal, window) == 0).sum().item() for i in range(b)])
    # print(f"{total_q=}, {max_seqlen_q=}, {total_k=}, {max_seqlen_k=}, {total_attn_compute=}, {cu_seqlens_q.tolist()}, {cu_seqlens_k.tolist()}")

    q = torch.randn(total_q, h, d) / 10
    k = torch.randn(total_k, h_k, d) / 10
    v = torch.randn(total_k, h_k, dv) / 10
    grad_out = torch.randn(total_q, h, dv) / 10
    softmax_scale = (d + 100) ** (-0.5)

    q1 = q.clone().requires_grad_()
    k1 = k.clone().requires_grad_()
    v1 = v.clone().requires_grad_()

    if check_correctness:
        q2 = q.clone().requires_grad_()
        k2 = k.clone().requires_grad_()
        v2 = v.clone().requires_grad_()

    def flash_attn():
        q1.grad = k1.grad = v1.grad = None
        kwargs = {}
        if causal:
            kwargs["causal"] = causal
        if window != 0:
            kwargs["window_size"] = get_window_size(causal, window)
        return flash_attn_varlen_func(q1, k1, v1, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                                      max_seqlen_k, softmax_scale=softmax_scale, is_varlen=varlen, **kwargs)

    def torch_attn():
        q2.grad = k2.grad = v2.grad = None
        out = []
        lse = []
        for i in range(b):
            OUT, LSE = sdpa_checkpoint(
                q2[cu_seqlens_q[i].item(): cu_seqlens_q[i + 1].item()],
                k2[cu_seqlens_k[i].item(): cu_seqlens_k[i + 1].item()],
                v2[cu_seqlens_k[i].item(): cu_seqlens_k[i + 1].item()],
                attn_bias=get_attn_bias(seqlens_q[i].item(), seqlens_k[i].item(), causal, window),
                softmax_scale=softmax_scale,
            )
            out.append(OUT.transpose(-3, -2))
            lse.append(LSE.transpose(-2, -1))
        out = torch.cat(out)
        lse = torch.cat(lse)
        return out, lse

    out_flash, lse_flash = flash_attn()
    if has_bwd:
        out_flash.backward(grad_out, retain_graph=True)
        _dq1 = q1.grad.clone()
        dk1 = k1.grad.clone()
        dv1 = v1.grad.clone()

    if check_correctness:
        out_torch, lse_torch = torch_attn()
        assert check_is_allclose("out", out_flash.float(), out_torch, abs_tol=1e-3, rel_tol=8.01 / 128, cos_diff_tol=7e-6)
        assert check_is_allclose("lse", lse_flash.float(), lse_torch, abs_tol=1e-6, rel_tol=2.01 / 65536)

        if has_bwd:
            out_torch.backward(grad_out, retain_graph=True)
            assert check_is_allclose("dq", q1.grad, q2.grad, abs_tol=1e-3, rel_tol=8.01 / 128, cos_diff_tol=7e-6)
            assert check_is_allclose("dk", k1.grad, k2.grad, abs_tol=1e-3, rel_tol=8.01 / 128, cos_diff_tol=7e-6)
            assert check_is_allclose("dv", v1.grad, v2.grad, abs_tol=1e-3, rel_tol=8.01 / 128, cos_diff_tol=7e-6)

    def forward():
        return flash_attn()

    def backward():
        q1.grad = k1.grad = v1.grad = None
        out_flash.backward(grad_out, retain_graph=True)

    for _ in range(5):
        out, lse = forward()
        assert torch.equal(out, out_flash), "out deterministic check failed!"
        assert torch.equal(lse, lse_flash), "lse deterministic check failed!"
        if has_bwd:
            backward()
            # assert torch.equal(q1.grad, dq1), "dq deterministic check failed!"
            assert torch.equal(k1.grad, dk1), "dk deterministic check failed!"
            assert torch.equal(v1.grad, dv1), "dv deterministic check failed!"

    def timer(func, name):
        t = triton.testing.do_bench(func, warmup=2, rep=3)
        FLOPS = total_attn_compute * h * 2 * ((d + dv) if name == "fwd" else ((d * 3 + dv * 2)))
        print(f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOP/s, name: {name}")
        return t

    timer(forward, "fwd")
    if has_bwd:
        timer(backward, "bwd")


if __name__ == "__main__":
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    b = 2
    window = 0
    has_bwd = False

    # The small shapes are not in the upstream list; they are here so that the dense kernels can
    # still be exercised on a GPU whose memory cannot hold the fp32 reference for 4096/8192.
    skipped = 0
    for (mean_sq, mean_sk) in [(512, 512), (1024, 1024), (4096, 4096), (8192, 8192)]:
        for varlen in [False, True]:
            for (h, h_k) in [(128, 128), (32, 4)]:
                if h != h_k:
                    has_bwd = False
                else:
                    has_bwd = True
                for (d, dv) in [(128, 128), (192, 128)]:
                    for causal in [False, True]:
                        skip_correctness_check = mean_sq == 8192 and mean_sk == 8192 and h == 128
                        if not skip_correctness_check and not fits_in_gpu_memory(b, mean_sq, mean_sk, h, d, dv, has_bwd):
                            skipped += 1
                            print(f"skipping {b=} {mean_sq=} {mean_sk=} {h=}: not enough GPU memory for the fp32 reference")
                            continue
                        test_flash_attention(b, mean_sq, mean_sk, varlen, h, h_k, d, dv, causal, window, has_bwd, not skip_correctness_check)
    if skipped:
        print(f"{skipped} case(s) skipped: not enough GPU memory for the fp32 reference")
