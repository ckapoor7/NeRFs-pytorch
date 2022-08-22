import torch
from torch import nn


def inverse_sampling(
    bins: torch.Tensor, weights: torch.Tensor, num_samples: int, perturb: bool = False
) -> torch.Tensor:

    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True)

    # PDF -> CDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # midpoint or random sampling
    if perturb:
        position = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
    else:
        position = torch.linspace(0, 1, num_samples, device=cdf.device)
        position = position.expand(list(cdf.shape[:-1]) + [num_samples])

    # find index positions for placing
    position = position.contiguous()
    indices = torch.searchsorted(cdf, position, right=True)

    # remove out of bound indices
    bottom = torch.clamp(indices - 1, min=0)
    upper = torch.clamp(indices, max=cdf.shape[-1] - 1)
    good_indices = torch.stack([bottom, upper], dim=-1)

    # sample from CDF and the corresponding bin centers
    matched_shape = list(good_indices.shape[:-1]) + [cdf.shape[-1]]
    good_cdf = torch.gather(
        cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=good_indices
    )
    good_bins = torch.gather(
        cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=good_indices
    )

    # sampled ray length
    denominator = good_cdf[..., 1] - good_cdf[..., 0]
    denominator = torch.where(
        denominator < 1e-5, torch.ones_like(denominator), denominator
    )
    t_vector = (position - good_cdf[..., 0]) / denominator
    all_samples = good_bins[..., 0]

    return all_samples


def stratified_sampling(
    rays_origin: torch.Tensor,
    rays_direction: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: bool = True,
):
    """
    sample along the light ray in equally spaced bins
    """

    # create equally spaced bins
    bins = torch.linspace(0.0, 1.0, n_samples, device=rays_origin.device)

    # interpolate values to near and far points
    pts_on_ray = near * (1.0 - bins) + far * (bins)

    # sample points randomly from each bin
    if perturb:
        mid_pts = 0.5 * (pts_on_ray[1:] + pts_on_ray[:-1])
        upper_half = torch.concat([mid_pts, pts_on_ray[-1:]])  # include far point
        lower_half = torch.concat([pts_on_ray[:1], mid_pts])  # include near point
        random_val = torch.rand([n_samples], device=pts_on_ray.device)
        pts_on_ray = lower_half + (upper_half - lower_half) * random_val

    pts_on_ray = pts_on_ray.expand(list(rays_origin.shape[:-1]) + [n_samples])

    pts = (
        rays_origin[..., None, :]
        + rays_direction[..., None, :] * pts_on_ray[..., :, None]
    )
    return pts, pts_on_ray


def hierarchical_sampling(
    rays_origin: torch.Tensor,
    rays_direction: torch.Tensor,
    z_vals: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    perturb: bool = False,
):

    mid_z_vals = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_vals = inverse_sampling(
        mid_z_vals, weights[..., 1:-1], num_samples, perturb=perturb
    )
    new_z_vals = new_z_vals.detach()  # remove from computational graph

    # resample points based on their PDF
    combined_z_vals, _ = torch.sort(torch.cat([z_vals, new_z_vals], dim=-1), dim=-1)
    points = (
        rays_origin[..., None, :]
        + rays_direction[..., None, :] * combined_z_vals[..., :, None]
    )

    return points, combined_z_vals, new_z_vals
