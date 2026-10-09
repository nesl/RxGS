import os
import random
from typing import NamedTuple

import numpy as np
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud
import torch
import pandas as pd
import yaml

# ---- data structures ----
class SpectrumInfo(NamedTuple):
    T_rx: np.array
    T_tx: np.array
    spectrum: np.array
    spectrum_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    nerf_normalization: dict


def split_train_test(data_dir, num_samples, ratio_train, seed):
    random.seed(seed)
    np.random.seed(seed)
    all_indices = np.arange(num_samples)
    np.random.shuffle(all_indices)
    num_train = int(num_samples * ratio_train)
    train_set = set(all_indices[:num_train].tolist())
    test_set = set(all_indices[num_train:].tolist())
    np.savetxt(os.path.join(data_dir, 'train_index.txt'), np.sort(list(train_set)), fmt='%s')
    np.savetxt(os.path.join(data_dir, 'test_index.txt'), np.sort(list(test_set)), fmt='%s')
    print(f"\n  [Random split] Train TX: {len(train_set)}  Test TX: {len(test_set)}  (seed={seed})\n")
    return train_set, test_set


# ---- scene info readers (per dataset) ----
def readBLESceneInfo(args_model):

    path = args_model.source_path
    camera_scale = args_model.camera_scale
    voxel_size_scale = args_model.voxel_size_scale

    tx_pos = pd.read_csv(os.path.join(path, 'tx_pos.csv')).values.astype(np.float32)

    with open(os.path.join(path, 'gateway_position.yml')) as f:
        gw_dict = yaml.safe_load(f)
    gateway_names = list(gw_dict.keys())
    gateway_positions = np.array([gw_dict[name] for name in gateway_names], dtype=np.float32)

    gatewa_pos = torch.tensor(gateway_positions[0], dtype=torch.float32)

    cam_centers = [torch.tensor(tx_pos[i], dtype=torch.float32) for i in range(len(tx_pos))]

    gatewa_pos_t = gatewa_pos.unsqueeze(1)
    cam_center = torch.stack(cam_centers, dim=1)
    dists = torch.norm(cam_center - gatewa_pos_t, dim=0)
    radius = torch.max(dists) * camera_scale

    deviations = cam_center - gatewa_pos_t
    positive_deviations = deviations.clone()
    negative_deviations = deviations.clone()
    positive_deviations[positive_deviations < 0] = 0
    negative_deviations[negative_deviations > 0] = 0
    max_positive = positive_deviations.max(dim=1).values
    max_negative = negative_deviations.min(dim=1).values.abs()
    epsilon = 1e-6
    max_positive[max_positive < epsilon] = 1.0
    max_negative[max_negative < epsilon] = 1.0

    nerf_normalization = {
        "radius": radius.item(),
        "extent": {"max_positive": max_positive * camera_scale,
                   "max_negative": max_negative * camera_scale},
    }

    ply_path = os.path.join(path, getattr(args_model, 'init_ply_name', "points3D.ply"))
    if (not os.path.exists(ply_path)) or args_model.gene_init_point:
        receiver_pos = gatewa_pos.numpy()
        frequency = float(getattr(args_model, 'frequency', 2.4e9))
        cube_size = round((3.00e8 / frequency) * voxel_size_scale, 2)
        max_points = getattr(args_model, 'max_init_points', 10000)
        num_pos = init_ply(ply_path, receiver_pos, nerf_normalization["extent"], cube_size, max_points)
        print(f"\nInitialized point cloud: cube_size={cube_size}m, num_points={num_pos}\n")

    pcd = fetch_init_ply(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           nerf_normalization=nerf_normalization)
    return scene_info


def readSpectrumMultiRXSceneInfo(args_model):
    """Scene reader for multi-RX spectrum datasets (rx_positions.yml, spectrum/{idx}_{rx}.png).

    Only initializes point cloud and normalization — spectrum data is loaded
    separately by load_spectrum_per_rx / load_spectrum_all_rx.
    """
    path = args_model.source_path
    camera_scale = args_model.camera_scale
    voxel_size_scale = args_model.voxel_size_scale

    tx_pos = pd.read_csv(os.path.join(path, 'tx_pos.csv')).values.astype(np.float32)

    with open(os.path.join(path, 'rx_positions.yml')) as f:
        rx_dict = yaml.safe_load(f)['rx_positions']
    rx_names = sorted(rx_dict.keys())
    rx_positions = np.array([rx_dict[name] for name in rx_names], dtype=np.float32)

    # use first RX for normalization center
    gatewa_pos = torch.tensor(rx_positions[0], dtype=torch.float32)
    cam_centers = [torch.tensor(tx_pos[i], dtype=torch.float32)
                   for i in range(len(tx_pos))]

    gatewa_pos_t = gatewa_pos.unsqueeze(1)
    cam_center = torch.stack(cam_centers, dim=1)
    dists = torch.norm(cam_center - gatewa_pos_t, dim=0)
    radius = torch.max(dists) * camera_scale

    deviations = cam_center - gatewa_pos_t
    positive_deviations = deviations.clone()
    negative_deviations = deviations.clone()
    positive_deviations[positive_deviations < 0] = 0
    negative_deviations[negative_deviations > 0] = 0
    max_positive = positive_deviations.max(dim=1).values
    max_negative = negative_deviations.min(dim=1).values.abs()
    epsilon = 1e-6
    max_positive[max_positive < epsilon] = 1.0
    max_negative[max_negative < epsilon] = 1.0

    nerf_normalization = {
        "radius": radius.item(),
        "extent": {"max_positive": max_positive * camera_scale,
                   "max_negative": max_negative * camera_scale},
    }

    ply_path = os.path.join(path, "points3D.ply")
    if (not os.path.exists(ply_path)) or args_model.gene_init_point:
        receiver_pos = gatewa_pos.numpy()
        frequency = float(getattr(args_model, 'frequency', 2.4e9))
        cube_size = round((3.00e8 / frequency) * voxel_size_scale, 2)
        max_points = getattr(args_model, 'max_init_points', 10000)
        num_pos = init_ply(ply_path, receiver_pos, nerf_normalization["extent"],
                              cube_size, max_points)
        print(f"\nInitialized point cloud: cube_size={cube_size}m, num_points={num_pos}\n")

    pcd = fetch_init_ply(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           nerf_normalization=nerf_normalization)
    return scene_info


# ---- point cloud initialization ----
def fetch_init_ply(path):

    plydata = PlyData.read(path)

    vertices = plydata['vertex']

    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    return BasicPointCloud(points=positions)


# generate initial 3D grid centered at receiver, downsample if exceeding max_points
def init_ply(
    ply_path,
    receiver_pos,
    camera_extent,
    cube_size,
    max_points=10000,
):

    dtype = [('x', 'f4'),  ('y', 'f4'),  ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')
            ]
    xyz = generate_cube_coordinates(receiver_pos, camera_extent, cube_size)

    # randomly subsample if grid exceeds max_points
    if max_points and xyz.shape[0] > max_points:
        indices = np.random.choice(xyz.shape[0], max_points, replace=False)
        indices.sort()
        xyz = xyz[indices]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)

    attributes = np.concatenate((xyz, normals), axis=1)

    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')

    ply_data = PlyData([vertex_element])

    ply_data.write(ply_path)

    return xyz.shape[0]


def generate_cube_coordinates(receiver_pos, camera_extent, cube_size):
    x_min = receiver_pos[0] - camera_extent["max_negative"][0].item()
    x_max = receiver_pos[0] + camera_extent["max_positive"][0].item()

    y_min = receiver_pos[1] - camera_extent["max_negative"][1].item()
    y_max = receiver_pos[1] + camera_extent["max_positive"][1].item()

    z_min = receiver_pos[2] - camera_extent["max_negative"][2].item()
    z_max = receiver_pos[2] + camera_extent["max_positive"][2].item()

    num_cubes_x = int(np.ceil((x_max - x_min) / cube_size))
    num_cubes_y = int(np.ceil((y_max - y_min) / cube_size))
    num_cubes_z = int(np.ceil((z_max - z_min) / cube_size))

    x_coords = np.linspace(x_min, x_max, num_cubes_x) if num_cubes_x > 1 else np.array([(x_min + x_max) / 2])
    y_coords = np.linspace(y_min, y_max, num_cubes_y) if num_cubes_y > 1 else np.array([(y_min + y_max) / 2])
    z_coords = np.linspace(z_min, z_max, num_cubes_z) if num_cubes_z > 1 else np.array([(z_min + z_max) / 2])

    x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    cube_points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

    return cube_points


def readCSISceneInfo(args_model):
    """Scene reader for KU Leuven distributed CSI dataset (multi-RX)."""

    path = args_model.source_path
    camera_scale = args_model.camera_scale
    voxel_size_scale = args_model.voxel_size_scale

    tx_pos = pd.read_csv(os.path.join(path, 'tx_pos.csv')).values.astype(np.float32)

    with open(os.path.join(path, 'rx_positions.yml')) as f:
        rx_cfg = yaml.safe_load(f)
    rx_names = [k for k in rx_cfg.keys() if k.startswith('rx')]
    rx_names = sorted(rx_names)
    rx_positions = np.array([rx_cfg[name] for name in rx_names], dtype=np.float32)

    # Use first RX as reference for scene extent
    rx_pos = torch.tensor(rx_positions[0], dtype=torch.float32)

    cam_centers = [torch.tensor(tx_pos[i], dtype=torch.float32) for i in range(len(tx_pos))]
    cam_center = torch.stack(cam_centers, dim=1)
    rx_pos_t = rx_pos.unsqueeze(1)
    dists = torch.norm(cam_center - rx_pos_t, dim=0)
    radius = torch.max(dists) * camera_scale

    deviations = cam_center - rx_pos_t
    positive_deviations = deviations.clone()
    negative_deviations = deviations.clone()
    positive_deviations[positive_deviations < 0] = 0
    negative_deviations[negative_deviations > 0] = 0
    max_positive = positive_deviations.max(dim=1).values
    max_negative = negative_deviations.min(dim=1).values.abs()
    epsilon = 1e-6
    max_positive[max_positive < epsilon] = 1.0
    max_negative[max_negative < epsilon] = 1.0

    nerf_normalization = {
        "radius": radius.item(),
        "extent": {"max_positive": max_positive * camera_scale,
                   "max_negative": max_negative * camera_scale},
    }

    ply_path = os.path.join(path, "points3D.ply")
    if (not os.path.exists(ply_path)) or args_model.gene_init_point:
        receiver_pos = rx_pos.numpy()
        frequency = float(getattr(args_model, 'frequency', 2.61e9))
        cube_size = round((3.00e8 / frequency) * voxel_size_scale, 2)
        max_points = getattr(args_model, 'max_init_points', 10000)
        num_pos = init_ply(ply_path, receiver_pos, nerf_normalization["extent"], cube_size, max_points)
        print(f"\nInitialized point cloud: cube_size={cube_size}m, num_points={num_pos}\n")

    pcd = fetch_init_ply(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           nerf_normalization=nerf_normalization)
    return scene_info
