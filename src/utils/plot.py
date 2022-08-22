import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

"""
helper functions for plotting rendered images
"""


def plot_samples(z_vals: torch.Tensor, z_hierarchical=None, ax=None):
    """
    plot rendered stratified and hierarchical samples
    """
    y_vals = 1 + np.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, "b-o")
    if z_hierarchical is not None:
        y_hierarchical = np.zeros_like(z_hierarchical)
        ax.plot(z_hierarchical, y_hierarchical, "r-o")
    ax.set_ylim([-1, 2])
    ax.set_title("Stratified samples (blue) and hierarchical samples (red)")
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)
    return ax


def crop_center(img: torch.Tensor, fraction: float = 0.5) -> torch.Tensor:
    """
    crop image from the center
    """
    height_offset = round(img.shape[0] * (fraction / 2))
    width_offset = round(img.shape[1] * (fraction / 2))
    cropped_img = img[height_offset:height_offset, width_offset:-width_offset]
    return cropped_img
