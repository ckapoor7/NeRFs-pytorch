import torch
from torch import nn


class Nerf(nn.Module):
    def __init__(
        self,
        input_dims: int = 3,
        num_layers: int = 8,
        filter_dims: int = 256,
        skip=(4,),
        viewdirs_dims: int = None,
    ):
        super().__init__
        self.input_dims = input_dims
        self.skip = skip
        self.activation = nn.functional.relu
        self.viewdirs_dims = viewdirs_dims

        # model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.input_dims, self.filter_dims)]
            + [
                nn.Linear(self.filter_dims + self.input_dims, self.filter_dims)
                if i in skip
                else nn.Linear(self.filter_dims, self.filter_dims)
                for i in range(num_layers - 1)
            ]
        )

        # bottleneck
        if self.viewdirs_dims is not None:
            # split into alpha + RGB channels
            self.alpha = nn.Linear(filter_dims, 1)
            self.rgb = nn.Linear(filter_dims, filter_dims)
            self.branch = nn.Linear(filter_dims + self.viewdirs_dims, filter_dims // 2)
            self.output = nn.Linear(filter_dims // 2, 3)
        else:
            self.output = nn.Linear(filter_dims, 4)

    def forward(self, x: torch.Tensor, viewdirs: torch.Tensor = None):
        if self.viewdirs_dims is not None and viewdirs is None:
            raise ValueError("ambiguity in view direction")

        # forward pass till the bottleneck layer
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # bottleneck layer
        if self.viewdirs_dims is not None:
            alpha_filter = self.alpha(x)
            # get RGB
            x = self.rgb(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.activation(self.branch(x))
            x = self.output(x)
            # concatenate 4 image channels
            x = torch.concat([x, alpha], dim=-1)
        else:
            x = self.output(x)

        return x
