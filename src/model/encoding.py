import torch
from torch import nn


class PositionalEncoder(nn.Module):
    """
    fourier encoding of input points
    """

    def __init__(self, input_dims: int, num_freq: int, log_space: bool = False):
        super().__init__()
        self.input_dims = input_dims
        self.num_freq = num_freq
        self.log_space = log_space
        self.output_dims = input_dims * (1 + 2 * self.num_freq)
        self.embedding_func = [lambda x: x]

        # frequency band in log and linear space
        if self.log_space:
            freq_band = 2.0 ** torch.linspace(0, self.num_freq - 1, self.num_freq)
        else:
            freq_band = torch.linspace(0, 2 ** (self.num_freq - 1), self.num_freq)

        # construct fourier embedding matrix
        for freq in freq_band:
            self.embedding_func.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embedding_func.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x):
        return torch.concat([embedding(x) for embedding in self.embedding_func], dim=-1)


def test_class():
    encoder = PositionalEncoder(3, 30)

    viewdirs_encoder = PositionalEncoder(3, 4)

    # flattened points + view directions
    pts_flattened = pts.reshape(-1, 3)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    flattened_viewdirs = viewdirs[:, None, ...].expand(pts.shape).reshape((-1, 3))

    # encode inputs
    encoded_points = encoder(pts_flattened)
    encoded_viewdirs = viewdirs_encoder(flattened_viewdirs)

    print("Encoded points")
    print(encoded_points.shape)
    print(
        torch.min(encoded_points), torch.max(encoded_points), torch.mean(encoded_points)
    )

    print("Encoded viewdirs")
    print(encoded_viewdirs.shape)
    print(
        torch.min(encoded_viewdirs),
        torch.max(encoded_viewdirs),
        torch.mean(encoded_viewdirs),
    )
