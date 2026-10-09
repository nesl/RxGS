#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Python bindings for complex-valued Gaussian tracer CUDA kernels

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


# ---- entry point: dispatch to autograd Function ----

def tracer_complex_gaussians(means_3d,
                        cov3d_precomp,
                        signal_precomp,
                        attenuation,
                        raster_settings,
                        ):
    
    return _TracerComplexGaussians.apply(means_3d,
                                     cov3d_precomp,
                                     signal_precomp,
                                     attenuation,
                                     raster_settings,
                                     )


# ---- autograd Function: forward and backward through CUDA kernels ----

class _TracerComplexGaussians(torch.autograd.Function):
    

    @staticmethod
    def forward(ctx,
                means_3d,
                cov3d_precomp,
                signal_precomp,
                attenuation,
                raster_settings
                ):
        # filter out Gaussians inside the receiver sphere (1.5x radius margin)
        sphere_radius = raster_settings.sphere_radius
        position = raster_settings.sphere_center

        scale_dis = 1.5
        sphere_radius_filter = sphere_radius * scale_dis
        distances        = torch.norm(means_3d - position, dim=1)

        indices_to_keep = (distances > sphere_radius_filter)

        # Restructure arguments the way that the C++ lib expects them
        args = (means_3d[indices_to_keep],
                cov3d_precomp[indices_to_keep],
                signal_precomp[indices_to_keep],
                attenuation[indices_to_keep],
                raster_settings.gaus_radii[indices_to_keep],
                raster_settings.height,
                raster_settings.width,
                raster_settings.fle_degree_active,
                raster_settings.spectrum_3d_fine,
                raster_settings.sphere_center,
                raster_settings.sphere_radius,
                raster_settings.debug
                )
        
        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:

            cpu_args = cpu_deep_copy_tuple(args)

            try:
                num_rendered, color, geomBuffer, binningBuffer, imgBuffer = _C.tracer_complex_gaussians(*args)

            except Exception as ex:

                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")

                raise ex
            
        else:

            num_rendered, color, geomBuffer, binningBuffer, imgBuffer = _C.tracer_complex_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.indices_to_keep = indices_to_keep
        ctx.save_for_backward(means_3d, cov3d_precomp, signal_precomp, attenuation,\
                              geomBuffer, binningBuffer, imgBuffer)
        
        return color


    @staticmethod
    def backward(ctx, 
                 grad_out_color
                 ):
        

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        indices_to_keep = ctx.indices_to_keep

        means_3d, cov3d_precomp, signal_precomp, attenuation,\
            geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (grad_out_color,
                means_3d[indices_to_keep],
                cov3d_precomp[indices_to_keep],
                signal_precomp[indices_to_keep],
                attenuation[indices_to_keep],
                raster_settings.gaus_radii[indices_to_keep],
                num_rendered,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                raster_settings.height,
                raster_settings.width,
                raster_settings.fle_degree_active,
                raster_settings.spectrum_3d_fine,
                raster_settings.sphere_center,
                raster_settings.sphere_radius,
                raster_settings.debug
                )
        
        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means_3d, grad_cov3d_precomp, \
                    grad_signal_precomp, grad_attenuation = _C.tracer_complex_gaussians_backward(*args)
            
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
            
        else:
            grad_means_3d, grad_cov3d_precomp, \
                grad_signal_precomp, grad_attenuation = _C.tracer_complex_gaussians_backward(*args)
            
        # scatter gradients back to full tensor (filtered Gaussians get zero grad)
        final_grad_means_3d = torch.zeros_like(means_3d)
        final_grad_means_3d[indices_to_keep] = grad_means_3d

        final_grad_cov3d_precomp = torch.zeros_like(cov3d_precomp)
        final_grad_cov3d_precomp[indices_to_keep] = grad_cov3d_precomp

        final_grad_signal_precomp = torch.zeros_like(signal_precomp)
        final_grad_signal_precomp[indices_to_keep] = grad_signal_precomp

        final_grad_attenuation = torch.zeros_like(attenuation)
        final_grad_attenuation[indices_to_keep] = grad_attenuation

        grads = (final_grad_means_3d,
                 final_grad_cov3d_precomp,
                 final_grad_signal_precomp,
                 final_grad_attenuation,
                 None
                 )

        return grads


# ---- rasterizer settings and module ----

class ComplexGaussianTracerSettings(NamedTuple):
    height             : int
    width              : int
    fle_degree_active   : int
    spectrum_3d_fine   : torch.Tensor
    sphere_center      : torch.Tensor
    sphere_radius      : float
    debug              : bool
    gaus_radii         : torch.Tensor


class ComplexGaussianTracer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings


    def forward(self, means_3d, cov3d_precomp, signal_precomp, attenuation):

        raster_settings = self.raster_settings

        # Invoke C++/CUDA rasterization routine
        return tracer_complex_gaussians(means_3d,
                                   cov3d_precomp,
                                   signal_precomp,
                                   attenuation,
                                   raster_settings
                                   )


# ---- multi-RX batched forward + backward (autograd-aware) ----

def tracer_complex_gaussians_multirx(means_3d,
                                      cov3d_precomp,
                                      signal_precomp,        # (N_rx, P, 2)
                                      attenuation,
                                      raster_settings,
                                      ):
    """Batched forward over N_rx receivers sharing the same TX/Gaussian state.

    Shape contract:
        means_3d        : (P, 3)
        cov3d_precomp   : (P, 6)
        signal_precomp  : (N_rx, P, 2)   -- per-RX FLE-evaluated complex amps
        attenuation     : (P, 1)
        raster_settings.gaus_radii : (P,)

    Returns:
        out_color       : (N_rx, 2, H, W)  -- one rendered (re, im) plane per RX

    Autograd is supported: gradients flow through means_3d, cov3d_precomp,
    signal_precomp and attenuation.  Gaussian-level gradients are accumulated
    across all N_rx outputs; signal_precomp gradient is per-RX.
    """
    assert signal_precomp.dim() == 3, \
        f"signal_precomp must be (N_rx, P, 2), got {tuple(signal_precomp.shape)}"
    assert signal_precomp.shape[1] == means_3d.shape[0], \
        "signal_precomp.shape[1] must equal P"
    assert signal_precomp.shape[2] == 2, \
        "signal_precomp.shape[2] must equal 2 (re, im)"

    return _TracerComplexGaussiansMultirx.apply(
        means_3d, cov3d_precomp, signal_precomp, attenuation, raster_settings,
    )


class _TracerComplexGaussiansMultirx(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                means_3d,
                cov3d_precomp,
                signal_precomp,        # (N_rx, P, 2)
                attenuation,
                raster_settings,
                ):
        # Filter Gaussians inside the receiver sphere, as in the single-RX path.
        sphere_radius = raster_settings.sphere_radius
        position      = raster_settings.sphere_center
        sphere_radius_filter = sphere_radius * 1.5
        distances        = torch.norm(means_3d - position, dim=1)
        indices_to_keep  = (distances > sphere_radius_filter)

        args = (means_3d[indices_to_keep].contiguous(),
                cov3d_precomp[indices_to_keep].contiguous(),
                signal_precomp[:, indices_to_keep, :].contiguous(),
                attenuation[indices_to_keep].contiguous(),
                raster_settings.gaus_radii[indices_to_keep].contiguous(),
                raster_settings.height,
                raster_settings.width,
                raster_settings.fle_degree_active,
                raster_settings.spectrum_3d_fine,
                raster_settings.sphere_center,
                raster_settings.sphere_radius,
                raster_settings.debug,
                )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, geomBuffer, binningBuffer, imgBuffer = \
                    _C.tracer_complex_gaussians_multirx(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw_multirx.dump")
                print("\nAn error occured in multirx forward. "
                      "Saved snapshot_fw_multirx.dump for debugging.")
                raise ex
        else:
            num_rendered, color, geomBuffer, binningBuffer, imgBuffer = \
                _C.tracer_complex_gaussians_multirx(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.indices_to_keep = indices_to_keep
        ctx.save_for_backward(means_3d, cov3d_precomp, signal_precomp, attenuation,
                              geomBuffer, binningBuffer, imgBuffer)

        return color

    @staticmethod
    def backward(ctx, grad_out_color):
        num_rendered    = ctx.num_rendered
        raster_settings = ctx.raster_settings
        indices_to_keep = ctx.indices_to_keep

        (means_3d, cov3d_precomp, signal_precomp, attenuation,
            geomBuffer, binningBuffer, imgBuffer) = ctx.saved_tensors

        args = (grad_out_color.contiguous(),
                means_3d[indices_to_keep].contiguous(),
                cov3d_precomp[indices_to_keep].contiguous(),
                signal_precomp[:, indices_to_keep, :].contiguous(),
                attenuation[indices_to_keep].contiguous(),
                raster_settings.gaus_radii[indices_to_keep].contiguous(),
                num_rendered,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                raster_settings.height,
                raster_settings.width,
                raster_settings.fle_degree_active,
                raster_settings.spectrum_3d_fine,
                raster_settings.sphere_center,
                raster_settings.sphere_radius,
                raster_settings.debug,
                )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                grad_means_3d, grad_cov3d_precomp, grad_signal_precomp, grad_attenuation = \
                    _C.tracer_complex_gaussians_multirx_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw_multirx.dump")
                print("\nAn error occured in multirx backward. "
                      "Saved snapshot_bw_multirx.dump for debugging.")
                raise ex
        else:
            grad_means_3d, grad_cov3d_precomp, grad_signal_precomp, grad_attenuation = \
                _C.tracer_complex_gaussians_multirx_backward(*args)

        # Scatter gradients back to the unfiltered tensors.
        N_rx = signal_precomp.shape[0]

        final_grad_means_3d = torch.zeros_like(means_3d)
        final_grad_means_3d[indices_to_keep] = grad_means_3d

        final_grad_cov3d_precomp = torch.zeros_like(cov3d_precomp)
        final_grad_cov3d_precomp[indices_to_keep] = grad_cov3d_precomp

        final_grad_attenuation = torch.zeros_like(attenuation)
        final_grad_attenuation[indices_to_keep] = grad_attenuation

        final_grad_signal_precomp = torch.zeros_like(signal_precomp)
        final_grad_signal_precomp[:, indices_to_keep, :] = grad_signal_precomp

        return (final_grad_means_3d,
                final_grad_cov3d_precomp,
                final_grad_signal_precomp,
                final_grad_attenuation,
                None)
    
    


