import torch
import json
from torch import nn
from .dataset import Dataset
from model.nerfs import Nerf
from train.train import train
from yacs.config import CfgNode
from config.utils import load_cfg
from model.encoding import PositionalEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(cfg: CfgNode):
    """
    initalize model with default configuration
    setting
    """
    # encoder
    encoder = PositionalEncoder(
        input_dims=cfg.ENCODERS.DIM_INPUT,
        num_freq=cfg.ENCODERS.NUM_FREQS,
        log_space=cfg.ENCODERS.LOG_SPACE,
    )
    encode = lambda x: encoder(x)

    # view direction encoders
    if cfg.ENCODERS.USE_VIEWDIRS:
        encoder_viewdirs = PositionalEncoder(
            input_dims=cfg.ENCODERS.DIM_INPUT,
            num_freq=cfg.ENCODERS.NUM_FREQ_DIRS,
            log_space=cfg.ENCODERS.LOG_SPACE,
        )
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        dim_viewdirs = encode_viewdirs.output_dims
    else:
        encode_viewdirs = None
        dim_viewdirs = None

    # models
    model = Nerf(
        input_dims=encoder.output_dims,
        num_layers=cfg.MODEL.NUM_LAYERS,
        filter_dims=cfg.MODEL.DIM_FILTER,
        skip=cfg.MODEL.SKIP,
        viewdirs_dims=dim_viewdirs,
    )
    model.to(device)
    model_params = list(model.parameters())

    if cfg.MODEL.USE_FINE_MODEL:
        fine_model = Nerf(
            input_dims=encoder.output_dims,
            num_layers=cfg.MODEL.NUM_LAYERS,
            filter_dims=cfg.MODEL.DIM_FILTER,
            skip=cfg.MODEL.SKIP,
            viewdirs_dims=dim_viewdirs,
        )
        fine_model.to(device)
        model_params = model_params + fine_model.parameters()
    else:
        fine_model = None

    # optimizer
    optimizer = torch.optim.Adam(model_params, lr=cfg.OPTIMIZER.LR)

    return model, fine_model, encode, encode_viewdirs, optimizer


def main():
    cfg = load_cfg
    pretty_cgf = json.dumps(cfg, indent=4)
    print(f"Current configuration of model:\n {pretty_cfg}")

    # get dataset
    dset = Dataset(fpath=cfg.TRAINING.DATASET)
    images, poses, focal = dset.readFile()
    testIdx = 101
    testImg, testPose = images[testIdx], poses[testIdx]

    # run everything
    model, fine_model, encode, encode_viewdirs, optimizer = init_model(cfg=cfg)
    print(f"Starting training...")
    train_status, train_psnrs, val_psnrs = train(
        testimg=testImg, testpose=testPose, cfg=cfg, save_img=True
    )
    if train_status:
        print(f"Training over")
        break
