from __future__ import annotations

import torch
from depth_anything_3.utils.geometry import affine_inverse


def normalize_extrinsics(ex_t: torch.Tensor | None) -> torch.Tensor | None:
    if ex_t is None:
        return None
    transform = affine_inverse(ex_t[:, :1])
    ex_t_norm = ex_t @ transform
    c2ws = affine_inverse(ex_t_norm)
    translations = c2ws[..., :3, 3]
    dists = translations.norm(dim=-1)
    median_dist = torch.median(dists)
    median_dist = torch.clamp(median_dist, min=1e-1)
    ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
    return ex_t_norm

