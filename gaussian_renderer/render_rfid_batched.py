from typing import List, Sequence

import torch

from utils.fle_utils import eval_fle
from . import calculate_gaussian_radii, create_sphere_rays
from complex_gaussian_tracer_multirx import (
    tracer_complex_gaussians_multirx as _multirx_forward,
    ComplexGaussianTracerSettings as _MultirxSettings,
)


def compute_tx_context(viewpoint_any_rx, pc, pipe):
    """TX-side and Gaussian-side state shared across all RXs."""
    scaling_modifier = 1.0
    radii_scale = 3.0

    means_3d = pc.get_xyz
    base_fle = pc.get_features          # (N, K, 2)
    attenuation = pc.get_attenuation
    cov3d_precomp, actual_cov3d = pc.get_covariance(scaling_modifier)

    tvec_tx = viewpoint_any_rx.T_tx.to(means_3d.device, dtype=means_3d.dtype)
    n_az = int(viewpoint_any_rx.width)
    n_el = int(viewpoint_any_rx.height)

    r_d_fine_ori = create_sphere_rays(n_azimuth=n_az, n_elevation=n_el,
                                      radius=pipe.radius_rx).to(
        means_3d.device, dtype=means_3d.dtype)
    r_d_fine_t = r_d_fine_ori + tvec_tx[:, None]
    r_d_w_fine = r_d_fine_t.permute(1, 0)

    gaus_radii = calculate_gaussian_radii(actual_cov3d, scale=radii_scale)

    return dict(
        means_3d=means_3d,
        base_fle=base_fle,
        attenuation=attenuation,
        cov3d_precomp=cov3d_precomp,
        tvec_tx=tvec_tx,
        n_az=n_az, n_el=n_el,
        r_d_w_fine=r_d_w_fine,
        gaus_radii=gaus_radii,
        radius_rx=pipe.radius_rx,
        debug=pipe.debug,
        active_fle_degree=pc.active_fle_degree,
    )


def _render_via_batched_cuda(viewpoints, ctx, rx_cond_module):
    """Rasterize all N_rx in one CUDA call (preprocess + sort run once)."""
    means_3d = ctx['means_3d']
    base_fle = ctx['base_fle']
    tvec_tx  = ctx['tvec_tx']

    rx_pos_batch = torch.stack([vp.T_rx for vp in viewpoints], dim=0).to(
        means_3d.device, dtype=means_3d.dtype)            # (N_rx, 3)
    cfs = [rx_cond_module(base_fle, rx_pos_batch[i], gaussian_pos=means_3d)
           for i in range(rx_pos_batch.shape[0])]
    cf_batch = torch.stack(cfs, dim=0)

    # eval_fle's basis only depends on dirs (TX-only) → compute once,
    # reduce against (N_rx, P, 2, K) coefficients via broadcasting.
    fle_view = cf_batch.transpose(2, 3).contiguous()   # (N_rx, P, 2, K)
    dir_pp = means_3d - tvec_tx.repeat(means_3d.shape[0], 1)
    dir_pp_n = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    fle_re, fle_im = eval_fle(ctx['active_fle_degree'], fle_view, dir_pp_n)
    signal_batch = torch.stack((fle_re, fle_im), dim=2).contiguous()   # (N_rx, P, 2)

    rs = _MultirxSettings(
        height=ctx['n_el'], width=ctx['n_az'],
        fle_degree_active=ctx['active_fle_degree'],
        spectrum_3d_fine=ctx['r_d_w_fine'],
        sphere_center=ctx['tvec_tx'],
        sphere_radius=ctx['radius_rx'],
        debug=ctx['debug'],
        gaus_radii=ctx['gaus_radii'],
    )
    out_batch = _multirx_forward(
        ctx['means_3d'], ctx['cov3d_precomp'], signal_batch,
        ctx['attenuation'], rs,
    )
    real = out_batch[:, 0]
    imag = out_batch[:, 1]
    mag  = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

    visibility = ctx['gaus_radii'] > 0.0
    return [
        {"render": mag[i], "visibility_filter": visibility, "radii": ctx['gaus_radii']}
        for i in range(len(viewpoints))
    ]


def render_rfid_rxcond_multirx(
    viewpoints: Sequence,
    pc,
    pipe,
    rx_cond_module,
) -> List[dict]:
    """Render N_rx spectrums for a single TX × N_rx RXs. All viewpoints must share T_tx, height, width."""
    ctx = compute_tx_context(viewpoints[0], pc, pipe)
    return _render_via_batched_cuda(viewpoints, ctx, rx_cond_module)
