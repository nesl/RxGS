/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */


#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TracerComplexGaussiansCUDA(const torch::Tensor& means_3d,
					   const torch::Tensor& cov3d_precomp,
					   const torch::Tensor& signal_precomp,
					   const torch::Tensor& attenuation,
					   const torch::Tensor& gaus_radii,
					   const int height,
					   const int width,
					   const int fle_degree_active,
					   const torch::Tensor& spectrum_3d_fine,
					   const torch::Tensor& sphere_center,
					   const float sphere_radius,
					   const bool debug
					   );


// Multi-RX batched forward: signal_precomp shape (N_rx, P, 2),
// out_color shape (N_rx, 2, H, W).
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TracerComplexGaussiansMultirxCUDA(const torch::Tensor& means_3d,
								  const torch::Tensor& cov3d_precomp,
								  const torch::Tensor& signal_precomp,
								  const torch::Tensor& attenuation,
								  const torch::Tensor& gaus_radii,
								  const int height,
								  const int width,
								  const int fle_degree_active,
								  const torch::Tensor& spectrum_3d_fine,
								  const torch::Tensor& sphere_center,
								  const float sphere_radius,
								  const bool debug
								  );


// Multi-RX batched backward.  dL_dout_color: (N_rx, 2, H, W);
// signal_precomp: (N_rx, P, 2).  Returns:
//   grad_means_3d:  (P, 3)         — accumulated across all N_rx
//   grad_cov3d:     (P, 6)         — accumulated across all N_rx
//   grad_signal:    (N_rx, P, 2)   — per-RX
//   grad_atten:     (P, 1)         — accumulated across all N_rx
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TracerComplexGaussiansMultirxBackwardCUDA(const torch::Tensor& dL_dout_color,
										  const torch::Tensor& means_3d,
										  const torch::Tensor& cov3d_precomp,
										  const torch::Tensor& signal_precomp,
										  const torch::Tensor& attenuation,
										  const torch::Tensor& gaus_radii,
										  const int num_rendered,
										  const torch::Tensor& geomBuffer,
										  const torch::Tensor& binningBuffer,
										  const torch::Tensor& imageBuffer,
										  const int height,
										  const int width,
										  const int fle_degree_active,
										  const torch::Tensor& spectrum_3d_fine,
										  const torch::Tensor& sphere_center,
										  const float sphere_radius,
										  const bool debug
										  );


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TracerComplexGaussiansBackwardCUDA(const torch::Tensor& dL_dout_color,
							   const torch::Tensor& means_3d,
							   const torch::Tensor& cov3d_precomp,
							   const torch::Tensor& signal_precomp,
							   const torch::Tensor& attenuation,
							   const torch::Tensor& gaus_radii,
							   const int num_rendered,
							   const torch::Tensor& geomBuffer,
							   const torch::Tensor& binningBuffer,
							   const torch::Tensor& imageBuffer,
							   const int height,
							   const int width,
							   const int fle_degree_active,
							   const torch::Tensor& spectrum_3d_fine,
							   const torch::Tensor& sphere_center,
							   const float sphere_radius,
							   const bool debug
							   );
