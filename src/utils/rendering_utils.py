import torch
from torch import nn

"""
Convert raw NeRF outputs to images (apply
volume integration).
We find the expected color of each pixel weighted
by its alpha value which determines the degree
of occlusion.
"""


def cumprod(tensor: torch.Tensor) -> torch.Tensor:
    """
    cumulative product (tensorflow like implmentation)
    """
    prod = torch.cumprod(tensor, -1)
    prod = torch.roll(tensor, 1, -1)
    prod[..., 0] = 1

    return prod


def nerf2pixel(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_direction: torch.Tensor,
    noise: float = 0.0,
    background: bool = False,
):

    """
    convert raw NeRF output to a 4-channel pixel
    color value
    """
    # pixel distances
    distances = z_vals[..., 1:] - z_vals[..., :-1]
    distances = torch.cat(
        [distances, 1e10 * torch.ones_like(distances[..., :1])], dim=-1
    )

    # real-world distances
    distances = distances * torch.norm(rays_direction[..., None, :], dim=-1)

    # add noise for regularization
    init_noise = 0
    if noise > 0:
        init_noise = torch.randn(raw[..., 3].shape) * noise

    # predict opacity of a particular pixel
    alpha = 1 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * distances)

    # predict RGB weight for each pixel using alpha values
    weights = alpha * cumprod(1 - alpha + 1e10)

    # compute RGB pixel values in a coordinate mesh
    rgb_pred = torch.sigmoid(raw[..., :3])
    rgb_mesh = torch.sum(weights[..., None] * rgb_pred, dim=-2)

    # estimate object depth
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # sum weights
    sum_weights = torch.sum(weights, dim=-1)

    if background:
        rgb_mesh += 1 - sum_weights[..., None]

    return rgb_mesh, depth_map, sum_weights, weights
