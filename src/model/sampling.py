import torch
from torch import nn


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


def plot_bins():
    rays_o = ray_origin.view([-1, 3])
    rays_d = ray_direction.view([-1, 3])
    n_samples = 8
    perturb = True
    with torch.no_grad():
        pts, z_vals = stratified_sampling(
            rays_o, rays_d, near, far, n_samples, perturb=perturb
        )

    print("Input Points")
    print(pts.shape)
    print("")
    print("Distances Along Ray")
    print(z_vals.shape)

    y_vals = torch.zeros_like(z_vals)

    _, z_vals_unperturbed = stratified_sampling(
        rays_o, rays_d, near, far, n_samples, perturb=False
    )
    plt.plot(z_vals_unperturbed[0].cpu().numpy(), 1 + y_vals[0].cpu().numpy(), "b-o")
    plt.plot(z_vals[0].cpu().numpy(), y_vals[0].cpu().numpy(), "r-o")
    plt.ylim([-1, 2])
    plt.title("Stratified Sampling (blue) with Perturbation (red)")
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    plt.grid(True)
