from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import re
import pdb
import open3d as o3d
import torch
import json

robot_info = {
    "ur5e":[0.85, 1.03],
    "xarm":[0.7, 0.33],
    "flexiv":[0.78, 1.03],
    "panda":[0.85, 1.03]
}

def img_normalize(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


class PointAfforDataset(Dataset):
    def __init__(self, run_type, cfg):
        self.run_type = run_type
        self.data = open(f"data_split/kitchen_{run_type}.txt","r").readlines()

        self.floor_max = 10701
        self.global_max = 16647

        self.robot_info_root = cfg.info_root
        self.rgbd_root = cfg.rgbd_root


    def read_data(self, item_name):
        detail_name = item_name.split("/")[0][5:]
        robot_path = os.path.join(self.robot_info_root, detail_name, "trial_info.json")

        with open(robot_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # robot name
        robot_name = data["robot_name"]

        robot_h_w = torch.tensor(np.array(robot_info[robot_name]), dtype=torch.float32)

        path = os.path.join(self.rgbd_root, item_name)

        # load global point cloud
        global_pc_path = os.path.join(path, "point_cloud_transformed/world_pcd_transformed.ply")
        global_pc = o3d.io.read_point_cloud(global_pc_path)
        global_pc_point = np.asarray(global_pc.points)

        # load floor point cloud
        floor_pc_path = os.path.join(path, "point_cloud_transformed/filtered_pcd_z_less_than_0_02.ply")
        floor_pc = o3d.io.read_point_cloud(floor_pc_path)
        floor_pc_point = np.asarray(floor_pc.points)

        global_mask = np.ones(self.global_max)
        floor_mask = np.ones(self.floor_max)
        floor_gt_mask = np.ones(self.floor_max)

        # pad floor pc
        if floor_pc_point.shape[0] < self.floor_max:
            floor_mask[floor_pc_point.shape[0]:] = 0
            floor_points_padded = np.zeros((self.floor_max, 3))
            floor_points_padded[:floor_pc_point.shape[0], :] = floor_pc_point
        else:
            floor_points_padded = floor_pc_point[:self.floor_max, :]

        # Load ground truth
        floor_gt_path = os.path.join(path, "filtered_transformed_results_with_gaussian_success_rate.txt")
        temp = np.genfromtxt(floor_gt_path, delimiter=',')
        floor_ground_truth = np.genfromtxt(floor_gt_path, delimiter=',')[:, 3]

        if floor_ground_truth.shape[0] < self.floor_max:
            floor_gt_mask[floor_pc_point.shape[0]:] = 0
            floor_ground_truth_padded = np.zeros((self.floor_max))
            floor_ground_truth_padded[:floor_ground_truth.shape[0]] = floor_ground_truth
        else:
            floor_ground_truth_padded = floor_ground_truth[:self.floor_max]
        
        # add target information
        target_pc_path = os.path.join(path, "point_cloud_transformed/target_pcd_transformed.ply")
        target_pc = o3d.io.read_point_cloud(target_pc_path)
        target_pc_point = np.asarray(target_pc.points)

        # add barrier information
        barrier_pc_path_0 = os.path.join(path, "point_cloud_transformed/barrier_0_pcd_transformed.ply")
        if os.path.exists(barrier_pc_path_0):
            barrier0_pc = o3d.io.read_point_cloud(barrier_pc_path_0)
            barrier0_pc_point = np.asarray(barrier0_pc.points)
            barrier_pc_path_1 = os.path.join(path, "point_cloud_transformed/barrier_1_pcd_transformed.ply")
            if os.path.exists(barrier_pc_path_1):
                barrier1_pc = o3d.io.read_point_cloud(barrier_pc_path_1)
                barrier1_pc_point = np.asarray(barrier1_pc.points)


        # init global feature_dimension
        feature_dimension = np.zeros((global_pc_point.shape[0], 1))
        # add target area
        distances_tar = np.linalg.norm(
            global_pc_point[:, np.newaxis, :] - target_pc_point[np.newaxis, :, :],
            axis=2
        )
        min_distances = np.min(distances_tar, axis=1)
        threshold = 0.031  # match threshold
        # a = min_distances < threshold
        # print(np.sum(a))
        # import pdb; pdb.set_trace()
        
        # target area to 1
        feature_dimension[min_distances < threshold, 0] = 1  

        if os.path.exists(barrier_pc_path_0):
            distances_b1 = np.linalg.norm(
                global_pc_point[:, np.newaxis, :] - barrier0_pc_point[np.newaxis, :, :],
                axis=2
            )
            min_distances = np.min(distances_b1, axis=1)
            # threshold = 0.031  
            # a = min_distances < threshold
            # print(np.sum(a))
            # import pdb; pdb.set_trace()
            feature_dimension[min_distances < threshold, 0] = -1  
            if os.path.exists(barrier_pc_path_1):
                distances_b2 = np.linalg.norm(
                    global_pc_point[:, np.newaxis, :] - barrier1_pc_point[np.newaxis, :, :],
                    axis=2
                )
                min_distances = np.min(distances_b2, axis=1)
                # threshold = 0.031  
                # a = min_distances < threshold
                # print(np.sum(a))
                # import pdb; pdb.set_trace()
                feature_dimension[min_distances < threshold, 0] = -1  

        # normalize
        global_nor, _, _ = pc_normalize(global_pc_point)
        floor_nor, _, _ = pc_normalize(floor_points_padded)

        points_with_feature = np.hstack((global_nor, feature_dimension))

        # pad global pc
        if global_pc_point.shape[0] < self.global_max:
            global_mask[points_with_feature.shape[0]:] = 0
            global_points_padded = np.zeros((self.global_max, 4))
            global_points_padded[:points_with_feature.shape[0], :] = points_with_feature
        else:
            global_points_padded = points_with_feature[:self.global_max, :]

        # Convert to tensors
        global_points_padded = torch.tensor(global_points_padded, dtype=torch.float32)
        floor_points_padded = torch.tensor(floor_points_padded, dtype=torch.float32)

        # global_mask = torch.tensor(global_mask, dtype=torch.float32)
        floor_mask = torch.tensor(floor_mask, dtype=torch.float32)

        floor_ground_truth_padded = torch.tensor(floor_ground_truth_padded, dtype=torch.float32)

        return global_points_padded, floor_points_padded, floor_mask, floor_ground_truth_padded, robot_h_w


    def __len__(self):                
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].strip()

        global_pc, floor_pc, floor_mask, floot_gt, robot_h_w = self.read_data(item)

        # transpose
        global_pc = global_pc.transpose(0, 1)
        floor_pc = floor_pc.transpose(0, 1)

        return global_pc, floor_pc, floor_mask, floot_gt, robot_h_w


if __name__ == "__main__":
    dataset = PointAfforDataset('train')
    dataset.__getitem__(0)
    # print(dataset[0][0].shape)
    # print(dataset[40][0].shape)
    pdb.set_trace()