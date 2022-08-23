import os
import torch
import numpy as np
from torch import nn
from skimage import imsave
from ..dataset import Dataset
from yacs.config import CfgNode
from .nerf_forward import nerf_forward
from ..utils.rendering_utils import getRays
from ..utils.plot import crop_center, plot_samples


def create_kwargs(cfg: CfgNode):
    """
    dictionary arguments for stratified and
    hierarchical sampling
    """

    kwargs_sample_stratified = {
        "n_samples": cfg.STRAT_SAMPLING.NUM_SAMPLES,
        "perturb": cfg.STRAT_SAMPLING.PERTURB,
        "inverse_depth": cfg.STRAT_SAMPLING.INVERSE_DEPTH,
    }

    kwargs_sample_hierarchical = {"perturb": cfg.H_SAMPLING.PERTURB_HIERARCHICAL}

    return kwargs_sample_stratified, kwargs_sample_hierarchical


def train(testimg, testpose, cfg: CfgNode, save_img: bool = False):
    # get kwargs for nerf forward function
    kwargs_sample_stratified, kwargs_sample_hierarchical = create_kwargs(cfg=cfg)
    # scramble rays across all images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dset = Dataset(fpath=cfg.TRAINING.DATASET)
    images, poses, focal = dset.readFile()
    if not cfg.TRAINING.BATCHING_DISABLE:
        height, width = images.shape[1:3]
        all_rays = torch.stack(
            [
                torch.stack(
                    getRays(height=height, width=width, focal_length=focal, pose=poses)
                )
            ]
        )
        rgb_rays = torch.cat([all_rays, images[:, None]], 1)
        rgb_rays = torch.permute(rgb_rays, [0, 2, 3, 1, 4])
        rgb_rays = rgb_rays.reshape([-1, 3, 3])
        rgb_rays = rgb_rays.type(torch.float32)
        rgb_rays = rgb_rays[torch.ranperm(rgb_rays.shape[0])]
        batch_index = 0

    train_psnrs = []
    val_psnrs = []
    num_iters = []

    for i in range(cfg.TRAINING.NUM_ITERS):
        model.train()

        if cfg.TRAINING.BATCHING_DISABLE:
            target_img_idx = np.random.randint(images.shape[0])
            target_img = images[target_img_idx].to(device)
            if cfg.CENTER_CROP and i < cfg.CENTER_CROP_ITERS:
                target_img = crop_center(target_img)
            height, width = target_img.shape[:2]
            target_pose = poses[target_img_idx].to(device)
            rays_origin, rays_direction = getRays(
                height=height, width=width, focal_length=focal, pose=target_pose
            )
            rays_origin = rays_origin.reshape([-1, 3])
            rays_direction = rays_direction.reshape([-1, 3])

    # random rays all over the image
    else:
        batch = rgb_rays[batch_index : batch_index + cfg.TRAINING.BATCH_SIZE]
        batch = torch.transpose(batch, 0, 1)
        rays_origin, rays_direction, target_img = batch
        height, width = target_img.shape[:2]
        batch_index += cfg.TRAINING.BATCH_SIZE
        # shuffle after every iteration
        if batch_index >= rgb_rays.shape[0]:
            rgb_rays = rgb_rays[torch.randperm(rgb_rays.shape[0])]
            batch_index = 0
    target_img = target_img.reshape([-1, 3])

    # run a single iteration and get rendered image
    outputs = nerf_forward(
        rays_origin=rays_origin,
        rays_direction=rays_direction,
        near=2,
        far=6,
        encoding_function=encode,
        coarse_model=model,
        kwargs_sample_stratified=kwargs_sample_stratified,
        num_samples_hierarchical=cfg.H_SAMPLING.NUM_HIERARCHICAL_SAMPLES,
        kwargs_sample_hierarchical=kwargs_sample_hierarchical,
        fine_model=fine_model,
        viewdirs_encoding_function=encode_viewdirs,
        chunksize=cfg.TRANING.CHUNK_SIZE,
        cfg=cfg,
    )

    # backpropagtion
    rgb_predicted = outputs["rgb_mesh"]
    loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # compute MSE error between images
    psnr = -10 * torch.log10(loss)
    train_psnrs.append(psnr.item())

    # evaluate at frequency = display_rate
    if i % cfg.TRAINING.DISPLAY_RATE == 0:
        model.eval()
        height, width = testimg.shape[:2]
        rays_origin, rays_direction = getRays(
            height=height, width=width, focal=focal, pose=testpose
        )
        rays_origin = rays_origin.reshape([-1, 3])
        rays_direction = rays_direction.reshape([-1, 3])
        outputs = nerf_forward(
            rays_origin=rays_origin,
            rays_direction=rays_direction,
            near=2,
            far=6,
            encoding_function=encode,
            coarse_model=model,
            kwargs_sample_stratified=kwargs_sample_stratified,
            num_samples_hierarchical=cfg.H_SAMPLING.NUM_HIERARCHICAL_SAMPLES,
            kwargs_sample_hierarchical=kwargs_sample_hierarchical,
            fine_model=fine_model,
            viewdirs_encoding_function=encode_viewdirs,
            chunksize=cfg.TRANING.CHUNK_SIZE,
            cfg=cfg,
        )

        rgb_predicted = outputs["rgb_mesh"]
        # save for gif
        if save_img:
            res_img = rgb_predicted.reshape([height, width, 3].cpu().detach().numpy())
            if not os.path.exists(cfg.TRAINING.SAVE_PATH):
                print(f"Creating directory for saving images: {cfg.TRANING.SAVE_PATH}")
                os.makedir(cfg.TRANING.SAVE_PATH)
            if os.getcwd() != cfg.TRAINING.SAVE_PATH:
                os.chdir(cfg.TRANINING.SAVE_PATH)
            imsave(f"{i}.png", res_img.astype("float64"))

        loss = torch.nn.functional.mse_loss(rgb_predicted, testimg.reshape(-1, 3))
        print(f"Loss: {loss.item()}")
        val_psnr = -10 * torch.log10(loss)
        val_psnrs.append(val_psnr.item())
        num_iters.append(i)

        # plot outputs
        fig, ax = plt.subplots(
            1, 4, figsize=(24, 4), gridspec_kw={"width_ratios": [1, 1, 1, 3]}
        )
        # rendered image
        ax[0].imshow(rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy())
        ax[0].set_title(f"Iteration: {i}")
        # target image
        ax[1].imshow(testimg.detach().cpu().numpy())
        ax[1].set_title(f"Target")
        # plot psnr
        ax[2].plot(range(0, i + 1), train_psnrs, "r")
        ax[2].plot(num_iters, val_psnrs, "b")
        ax[2].set_title(f"PSNR (train=red, test=bluee)")
        z_vals_stratified = outputs["z_vals_stratified"].view(
            (-1, cfg.STRAT_SAMPLING.NUM_SAMPLES)
        )
        z_samples_stratified = (
            z_vals_stratified[z_vals_stratified.shape[0] // 2].detach().cpu().numpy()
        )
        if "z_vals_hierarchical" in outputs:
            z_vals_hierarchical = outputs["z_vals_hierarchical"].view(
                (-1, cfg.H_SAMPLING.NUM_HIERARCHICAL_SAMPLES)
            )
            z_samples_hierarchical = (
                z_vals_hierarchical[z_vals_hierarchical.shape[0] // 2]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            z_samples_hierarchical = None
        _ = plot_samples(z_samples_stratified, z_samples_hierarchical, ax=ax[3])
        ax[3].margins(0)
        plt.show()

        return True, train_psnrs, val_psnrs
