# DeepGEMM fallback reference for architectures it does not support.
#
# `tests/test-fused-norm-rope-attn-rope-cast.py` builds its reference with DeepGEMM's FP8 GEMM.
# DeepGEMM targets Hopper/Blackwell (it needs TMA + warp-specialised WGMMA), so it cannot be
# installed or run on SM89. This module reimplements the three DeepGEMM entry points that the
# test uses in plain PyTorch: the maths is the same (dequantise the per-token scales, then do the
# GEMM in fp32), only the kernels are plain rather than tensor-core specialised. It exists purely
# so the fused kernel can be validated numerically on SM89; on SM100 the real DeepGEMM is used.

import torch


def _ue8m0_exp(sf):
    """Convert positive power-of-two scale factors to their ue8m0 exponent (bias 127)."""
    return torch.log2(sf.float()).round().to(torch.int32) + 127


def _dequant(data, sf, data_shape):
    """Expand a packed (data, sf) pair to a dense float tensor.

    `sf` is either per-token packed (`int32`, four consecutive exponents per element) or plain
    per-block `float32`; both come out of `tile_kernels.quant.per_token_cast`. Every leading axis of
    `data_shape` is scaled per token, so `data_shape` may be 2D or 3D.
    """
    K = data_shape[-1]
    M = 1
    for d in data_shape[:-1]:
        M *= d
    if sf.shape[-1] == M and sf.shape[0] != M:
        sf = sf.t()
    if sf.dtype == torch.int32:
        gran = K // (sf.shape[-1] * 4)
        b = sf.to(torch.int64) & 0xFFFFFFFF
        shifts = torch.arange(4, device=sf.device, dtype=torch.int64)
        e = (b.unsqueeze(-1) >> (8 * shifts)) & 0xFF
        scale = torch.exp2(e.reshape(M, K // gran).to(torch.int32).sub_(127).float())
    else:
        gran = K // sf.shape[-1]
        scale = sf.float().reshape(M, K // gran)
    # Each scale factor covers `gran` consecutive elements of K.
    dense = data.float().view(M, K // gran, gran) * scale.reshape(M, K // gran, 1)
    return dense.view(*data_shape)


def _quant_to_dense(pair, shape):
    return _dequant(pair[0], pair[1], shape)


# Deep-SEA sized weights are dequantised in row blocks so that peak memory stays bounded.
_BLOCK_ELEMS = 1 << 23


def _slice_rows(pair, lo, hi):
    data, sf = pair[0], pair[1]
    n = data.shape[0]
    data = data[lo:hi]
    if sf.dim() > 1:
        axis = 0 if sf.shape[0] == n else sf.dim() - 1
        sf = sf[lo:hi] if axis == 0 else sf.narrow(axis, lo, hi - lo)
    return data, sf


def _row_block(n, k):
    return max(1, _BLOCK_ELEMS // max(k, 1))


def transform_sf_into_required_layout(sf, batch_size, hidden_dim, num_groups, recipe, is_sfa):
    """Pack per-block float32 scale factors into DeepGEMM's TMA-aligned packed-ue8m0 layout.

    Mirrors `allocate_scale_factor()` in csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp: the result
    has shape [num_groups, batch_size, hidden_dim / (gran * 4)] with `stride(1) == 1`, and the four
    ue8m0 bytes of one int32 are stored LSB-first.
    """
    assert not is_sfa
    gran = recipe[2]
    g, b, k = sf.shape
    assert (g, b, k) == (num_groups, batch_size, hidden_dim // gran)
    h4 = hidden_dim // (gran * 4)
    e = _ue8m0_exp(sf.reshape(num_groups, batch_size, h4, 4)).to(torch.uint8)

    aligned_b = (batch_size + 3) // 4 * 4
    shifts = torch.arange(4, device=sf.device, dtype=torch.int32) * 8
    packed = (e.to(torch.int32) << shifts).sum(-1).to(torch.int32)  # [G, B, H4]

    out = torch.zeros((num_groups, h4, aligned_b), dtype=torch.int32, device=sf.device)
    out[:, :, :batch_size] = packed.permute(0, 2, 1)
    return out.transpose(1, 2)[:, :batch_size]


def fp8_gemm_nt(a, b, out, recipe_a, recipe_b):
    """`out[m, n] = sum_k dequant(a)[m, k] * dequant(b)[n, k]`."""
    m, k = a[0].shape
    n = b[0].shape[0]
    a_dense = _quant_to_dense(a, (m, k)).to(torch.bfloat16)
    step = _row_block(n, k)
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        b_dense = _quant_to_dense(_slice_rows(b, lo, hi), (hi - lo, k)).to(torch.bfloat16)
        out[:, lo:hi].copy_(a_dense @ b_dense.t())
        del b_dense


def fp8_einsum(eq, a, b, out, recipe):
    """Single-GPU einsum on dequantised operands. Only `"bhr,hdr->bhd"` is needed here."""
    assert eq == "bhr,hdr->bhd"
    a_dense = _quant_to_dense(a, tuple(a[0].shape)).to(torch.bfloat16)
    d = b[0].shape[1]
    r = b[0].shape[2]
    step = _row_block(d, r)
    for lo in range(0, d, step):
        hi = min(lo + step, d)
        b_dense = _quant_to_dense(_slice_rows(b, lo, hi), tuple(b[0][lo:hi].shape)).to(torch.bfloat16)
        out[..., lo:hi].copy_(torch.einsum("bhr,hdr->bhd", a_dense, b_dense).to(out.dtype))
        del b_dense