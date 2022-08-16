import torch
from torch import nn


class Nerf(nn.Module):
    def __init__(
        self,
        discrete_input: int = 3,
        num_layers: int = 8,
        discrete_filter: int = 256,
        skip=(4,),
        discrete_viewdirs: int = None,
    ):
        super().__init__
        self.discrete_input = discrete_input
        self.skip = skip
        self.activation = nn.functional.relu
        self.discrete_viewdirs = discrete_viewdirs

        # model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.discrete_input, self.discrete_filter)]
            + [
                nn.Linear(
                    self.discrete_filter + self.discrete_input, self.discrete_filter
                )
                if i in skip
                else nn.Linear(self.discrete_filter, self.discrete_filter)
                for i in range(num_layers - 1)
            ]
        )

        # bottleneck
        if self.discrete_viewdirs is not None:
            # split into alpha + RGB channels
            self.alpha = nn.Linear(discrete_filter, 1)
            self.rgb = nn.Linear(discrete_filter, discrete_filter)
            self.branch = nn.Linear(
                discrete_filter + self.discrete_viewdirs, discrete_filter // 2
            )
            self.output = nn.Linear(discrete_filter // 2, 3)
        else:
            self.output = nn.Linear(discrete_filter, 4)

    def forward(self, x: torch.Tensor, viewdirs: torch.Tensor = None):
        if self.discrete_viewdirs is not None and viewdirs is None:
            raise ValueError("ambiguity in view direction")

        # forward pass till the bottleneck layer
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # bottleneck layer
        if self.discrete_viewdirs is not None:
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
