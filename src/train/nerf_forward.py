import torch
from yacs.config import CfgNode
from torch import nn
from typing import Optional, Callable
from .utils import prepare_chunks, prepare_viewdir_chunks
from ..utils.sampling_utils import stratified_sampling, hierarchical_sampling
from ..utils.rendering_utils import nerf2pixel

def nerf_forward(rays_origin: torch.Tensor,
                 rays_direction: torch.Tensor,
                 near: float,
                 far: float,
                 encoding_function,
                 coarse_model: nn.Module,
                 kwargs_sample_stratified: dict=None,
                 kwargs_sample_hierarchical: dict=None,
                 fine_model=None,
                 viewdirs_encoding_function: Optional[Callable[[torch.Tensor], torch.Tensor]]=None,
                 chunksize: int=2**15,
                 cfg: CfgNode):

    '''
    forward pass for the nerf (optimizers)
    '''
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}

    # sample points along the ray
    query_points, z_vals = stratified_sampling(rays_origin=rays_origin,
                                               rays_direction=rays_direction,
                                               near=near,
                                               far=far,
                                               **kwargs_sample_stratified)
    # batches + chunkify
    batches = prepare_chunks(points=query_points, encoding_function=encoding_function,
                             chunksize=chunksize)
    if viewdirs_encoding_function is not None:
        batches_viewdirs = prepare_viewdir_chunks(points=query_points,
                                                  rays_direction=rays_direction,
                                                  encoding_function=viewdirs_encoding_function,
                                                  chunksize=chunksize)
    else:
        batches_viewdirs = [None] * len(batches)

    # coarse model
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batches_viewdirs))
    raw_result = torch.cat(predictions, dim=0)
    raw_result = raw_result.reshape(list(query_points.shape[:2] + raw_result.shape[-1]))

    rgb_mesh, depth_map, sum_weights, weights = nerf2pixel(raw=raw_result,
                                                           z_vals=z_vals,
                                                           rays_direction=rays_direction)
    outputs = {'z_vals_stratified': z_vals}

    # fine model
    if cfg.H_SAMPLING.NUM_HIERARCHICAL_SAMPLES > 0:
        rgb_mesh_0, depth_map_0, sum_weights_0 = rgb_mesh, depth_map, sum_weights

        query_points, z_vals_combined, z_hierarchical = hierarchical_sampling(rays_origin=rays_origin,
                                                                              rays_direction=rays_direction,
                                                                              z_vals=z_vals,
                                                                              weights=weights,
                                                                              num_samples=cfg.H_SAMPLING.NUM_HIERARCHICAL_SAMPLES,
                                                                              **kwargs_sample_hierarchical)
        # batches + chunkify
        batches = prepare_chunks(points=query_points, encoding_function=encoding_function,
                                 chunksize=chunksize)
        if viewdirs_encoding_function is not None:
            batches_viewdirs = prepare_viewdir_chunks(points=query_points,
                                                      rays_direction=rays_direction,
                                                      encoding_function=viewdirs_encoding_function,
                                                      chunksize=chunksize)
        else:
            batch_viewdirs = [None] * len(batches)

        # perform a forward pass on the fine model (if specified)
        fine_model = fine_model if fine_model is not None else coarse_model
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
        raw_result = torch.cat(predictions, dim=0)
        raw_result = raw_result.reshape(list(query_points.shape[:2] + raw_result.shape[-1]))

        # volumetric rendering using NeRFs raw output
        rgb_mesh, depth_map, sum_weights, weights = nerf2pixel(raw=raw_result,
                                                               z_vals=z_vals_combined,
                                                               rays_direction=rays_direction)

        # store outputs in dictionary
        outputs['z_vals_hierarchical'] = z_hierarchical
        outputs['rgb_mesh_0'] = rgb_mesh_0
        outputs['depth_map_0'] = depth_map_0
        outputs['sum_weights_0'] = sum_weights_0

    # final outputs (coarse + fine model)
    outputs['rgb_mesh'] = rgb_mesh
    outputs['depth_map'] = depth_map
    outputs['sum_weights'] = sum_weights
    outputs['weights'] = weights

    return outputs
