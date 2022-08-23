import torch
from torch import nn


def get_chunks(inputs: torch.Tensor, chunksize: int = 2**15):
    """
    forward pass for potential memory limitations
    """
    return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def prepare_chunks(points: torch.Tensor, encoding_function, chunksize: int = 2**15):
    """
    chunkify + encode points
    """
    points = points.reshape((-1, 3))
    points = encoding_function(points)  # encode to higher frequency
    points = get_chunks(inputs=points, chunksize=chunksize)


def prepare_viewdir_chunks(
    points: torch.Tensor,
    rays_direction: torch.Tensor,
    encoding_function,
    chunksize: int = 2**15,
):
    viewdirs = rays_direction / torch.norm(
        rays_direction, dim=-1, keepdim=True
    )  # unit direction vector
    # analogous to prev. function
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    viewdirs = encoding_function(viewdirs)
    viewdirs = get_chunks(inputs=viewdirs, chunksize=chunksize)
    return viewdirs
