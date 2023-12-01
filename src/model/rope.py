from typing import Tuple

import torch

device = "cuda"

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = freqs.to(device)
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    if "mps" in [freqs_cis.device.type, x.device.type]:
        ndim = x.ndim - 1
        x_shape = x.shape[:-1]
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x_shape[1], x_shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x_shape)]
        return freqs_cis.view(*shape)

    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def _emulate_complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Emulates complex number multiplication a tensor with the last dim size being 2.

    If you already have a complex tensor, use `torch.view_as_real` to get a tensor
    compatible with this function.

    This is useful for backends that do not support complex numbers yet (MPS for example).

    NOTE: currently hardcoded for tensors with 4 (+1) dimensions!
    """
    acad = a[:, :, :, :, 0:1].repeat(1, 1, 1, 1, 2) * b
    bdbc = (
        a[:, :, :, :, 1:2].repeat(1, 1, 1, 1, 2)
        * b[:, :, :, :, [1, 0]]
        * torch.tensor([-1, 1]).to(device)
    )

    return acad + bdbc


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if "mps" in [xq.device.type, xk.device.type, freqs_cis.device.type]:
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
        freqs_cis = torch.view_as_real(reshape_for_broadcast(freqs_cis, xq_)).to(device)
        xq_out = _emulate_complex_mul(xq_, freqs_cis).flatten(3)
        xk_out = _emulate_complex_mul(xk_, freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


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