import os

from scene.gaussian_model import GaussianModel
from arguments import ModelParams

from scene.dataset_readers import readBLESceneInfo, readCSISceneInfo, readSpectrumMultiRXSceneInfo

class Scene:

    gaussians : GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
    ):

        self.model_path  = args.model_path
        self.gaussians   = gaussians

        # select dataset reader based on type
        dataset = args.dataset
        if dataset == 'ble_rssi':
            scene_info = readBLESceneInfo(args)
        elif dataset == 'csi':
            scene_info = readCSISceneInfo(args)
        elif dataset.startswith('spectrum_multirx'):
            scene_info = readSpectrumMultiRXSceneInfo(args)
        else:
            raise ValueError(f"Unknown dataset '{dataset}'")

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        self.gaussians.load_from_pcd(scene_info.point_cloud,
                                       self.cameras_extent)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
