from typing import Tuple

import torch


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_complex=True
) -> torch.Tensor:
    """
    (From LLama)
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """

    if not use_complex:
        return _mps_precompute_freqs_cis(dim, end, theta)

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""

    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    if "mps" in [xq.device.type, xk.device.type, freqs_cis.device.type]:
        return _mps_apply_rotary_emb(xq, xk, freqs_cis)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def reshape_for_broadcast(
    freqs_cis: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """

    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def _mps_precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    return torch.stack([torch.cos(freqs), torch.sin(freqs)], -1)


def _mps_apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cos, freqs_sin = freqs[..., 0], freqs[..., 1]
    freqs_cos, freqs_sin = _mps_reshape_for_broadcast(freqs_cos, freqs_sin, xq)

    xq = xq.view(*xq.shape[:-1], -1, 2)
    xk = xk.view(*xk.shape[:-1], -1, 2)
    xq_r, xq_i = xq[..., 0], xq[..., 1]
    xk_r, xk_i = xk[..., 0], xk[..., 1]

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1).flatten(-2)
    xk_out = torch.stack((xk_out_r, xk_out_i), dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def _mps_reshape_for_broadcast(
    freqs_sin: torch.Tensor, freqs_cos: torch.Tensor, x: torch.Tensor
):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cos.shape == freqs_sin.shape == (x.shape[1], x.shape[-1] // 2)
    shape = [1] * ndim
    shape[1] = x.shape[1]
    shape[-1] = x.shape[-1] // 2

    return freqs_sin.view(*shape), freqs_cos.view(*shape)
