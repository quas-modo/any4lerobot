from curses import erasechar
from pathlib import Path

import numpy as np
import h5py
from h5py import File
import numpy as np
from scipy.spatial.transform import Rotation as R
EPSILON = 5e-4
import glob
from tqdm import tqdm
import os
from PIL import Image


def read_realfranka_hdf5_data(file_path):
    data = {}
    with h5py.File(file_path, 'r') as f:
        # 读取各个数据组的数据
        data["actions_pose"] = f["actions_pose"][:]
        data["actions_pose_quat"] = f["actions_pose_quat"][:]
        data["action_gripper_width"] = f["action_gripper_width"][:]
        data["current_joints"] = f["current_joints"][:]
        data["current_pose"] = f["current_pose"][:]
        data["current_pose_quat"] = f["current_pose_quat"][:]
        data["current_gripper_width"] = f["current_gripper_width"][:]
        data["time_since_skill_started"] = f["time_since_skill_started"][:]
        data["instruct_keypoint"] = f["instruct_keypoint"][:]
        data["camera_1_image"] = f["camera_1_image"][:]
        data["camera_3_image"] = f["camera_3_image"][:]
        data["camera_3_image_a"] = f["camera_3_image_a"][:]
        # 检查是否存在并读取深度图
        if "camera_1_depth" in f:
            data["camera_1_depth"] = f["camera_1_depth"][:]
            
        print("actions_pose:", data["actions_pose"].shape)
        print("camera_1_image shape:", data["camera_1_image"].shape)  
        print("camera_3_image shape:", data["camera_3_image"].shape)
        print("camera_3_image_a shape:", data["camera_3_image_a"].shape)
    return data

# 6D (xyz + rpy) => 4x4
def _6d_to_pose(pose6d, degrees=False):
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()
    return pose

# 将 4x4 矩阵转回 6D (xyz + rpy)
def pose_to_6d(pose, degrees=False):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d

def compute_xyz_rpy_delta_9col(action_processed_data, state_processed_data_0):
    """
    processed_data: (N, 7)
      Columns: [x, y, z, roll, pitch, yaw, gripper]

    Returns:
      delta_data: (N, 7)
      Columns: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
    """
    N = action_processed_data.shape[0]
    delta_data = np.zeros((N, 7))
    delta_data[:, -1] = action_processed_data[:, -1]

    for i in range(N):
        if i == 0:
            last6 = state_processed_data_0[0:6]
            last2world = _6d_to_pose(last6)
        else:
            last6 = action_processed_data[i-1, 0:6]
            last2world = _6d_to_pose(last6)
            
        cur6 = action_processed_data[i, 0:6]
        cur2world = _6d_to_pose(cur6)

        cur2last = np.linalg.inv(last2world) @ cur2world
        delta6 = pose_to_6d(cur2last)
        delta_data[i, 0:6] = delta6    # [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]

    return delta_data

def load_local_episodes(input_h5: Path):
    data = read_realfranka_hdf5_data(input_h5)
    instr = np.asarray(data['instruct_keypoint'], dtype=int)
    
    # Find indices where indicator equals 1
    split_idxs = np.nonzero(instr == 1)[0]

    # Define the start/end of each segment
    # We want segments [0:split_idxs[0]+1], [split_idxs[0]+1:split_idxs[1]+1], ...
    boundaries = [0] + (split_idxs + 1).tolist()
    # Ensure we include the final segment to the end
    lengths = len(instr)
    boundaries.append(lengths)

    # Build the list of subtask dicts
    data_subtasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        sub = {}
        for k, v in data.items():
            arr = np.asarray(v)
            sub[k] = arr[start:end].tolist() if isinstance(v, list) else arr[start:end]
        data_subtasks.append(sub)

    for task_index, sub_task in enumerate(data_subtasks):
            state_pose = sub_task["current_pose"]  # Shape: (N, 4, 4)
            state_xyz = state_pose[:, :3, 3]           # Extract translation (N, 3)
            state_rotation_matrices = state_pose[:, :3, :3]  # Extract rotation matrices (N, 3, 3)
            state_rot = R.from_matrix(state_rotation_matrices)
            state_rpy = state_rot.as_euler('xyz', degrees=False)  # Convert to RPY (N, 3)
            state_quat = state_rot.as_quat(scalar_first=False)  # Convert to quaternion (N, 4)
            state_gripper_raw = sub_task['current_gripper_width']  # Gripper state (N, 1)
            state_gripper = (state_gripper_raw >= 0.04).astype(np.float32).reshape(-1, 1)  # Binary gripper state (N, 1)
            # Combine into (N, 8) processed_data
            state_processed_data = np.hstack((state_xyz, state_quat, state_gripper))  # Shape: (N, 8)
            ee_state_processed_data = np.hstack((state_xyz, state_rpy))  # Shape: (N, 6)
            joint_state_processed_data = sub_task['current_joints'].astype(np.float64)  # Shape: (N, 7)

            # Step 1: Prepare processed_data
            action_pose = sub_task["actions_pose"]  # Shape: (N, 4, 4)
            action_xyz = action_pose[:, :3, 3]           # Extract translation (N, 3)
            action_rotation_matrices = action_pose[:, :3, :3]  # Extract rotation matrices (N, 3, 3)
            action_rot = R.from_matrix(action_rotation_matrices)
            action_rpy = action_rot.as_euler('xyz', degrees=False)  # Convert to RPY (N, 3)
            action_gripper_raw = sub_task['action_gripper_width']  # Gripper state (N, 1)
            action_gripper = (action_gripper_raw >= 0.04).astype(np.float32).reshape(-1, 1)  # Binary gripper state (N, 1)
            # Combine into (N, 8) processed_data
            action_processed_data = np.hstack((action_xyz, action_rpy, action_gripper))  # Shape: (N, 7)
            # Step 2: Compute Delta Data
            action_delta_data = compute_xyz_rpy_delta_9col(action_processed_data, state_processed_data[0])  # Shape: (N, 7)

            # Step 3: Extract Non-Zero Delta Actions
            non_zero_delta_indices = np.any(np.abs(action_delta_data[:, :3]) > EPSILON, axis=1)   # Non-zero delta xyz/rpy
            previous_seventh_dim = 1.0
            seventh_dim_change = np.array([
                action_delta_data[i, 6] != (previous_seventh_dim if i == 0 else action_delta_data[i - 1, 6])
                for i in range(action_delta_data.shape[0])
            ])
            non_zero_delta_indices = non_zero_delta_indices | seventh_dim_change


            # non_zero_action_delta_data = action_delta_data[non_zero_delta_indices]  # Filtered sub_task
            non_zero_state_processed_data = state_processed_data[non_zero_delta_indices]  # Absolute actions already in processed_data
            non_zero_ee_state_processed_data = ee_state_processed_data[non_zero_delta_indices]
            non_zero_joint_state_processed_data = joint_state_processed_data[non_zero_delta_indices]
            non_zero_action_processed_data = action_processed_data[non_zero_delta_indices]

            # Step 5: Align Camera Data
            non_zero_wrist_images = sub_task['camera_1_image'][non_zero_delta_indices]  # Remove the extra frame
            non_zero_images_0 = sub_task['camera_3_image'][non_zero_delta_indices]
            non_zero_images_1 = sub_task['camera_3_image_a'][non_zero_delta_indices]
            assert non_zero_wrist_images.shape[0] == non_zero_state_processed_data.shape[0], \
                "Camera images and actions are misaligned in number of frames!"
            assert non_zero_images_0.shape[0] == non_zero_state_processed_data.shape[0], \
                "Camera images and actions are misaligned in number of frames!"
            assert non_zero_images_1.shape[0] == non_zero_state_processed_data.shape[0], \
                "Camera images and actions are misaligned in number of frames!"

            # third view image: image_0, image_1
            # action: absolute action
            # gripper(0: close, 1: open)
            episode = {
                "observation.images.image_0": non_zero_images_0,
                "observation.images.image_1": non_zero_images_1,
                "observation.images.wrist_image": non_zero_wrist_images,
                "observation.state": non_zero_state_processed_data,
                "observation.states.ee_state": non_zero_ee_state_processed_data,
                "observation.states.joint_state": non_zero_joint_state_processed_data,
                "action": non_zero_action_processed_data,
            }
            yield [{**{k: v[i] for k, v in episode.items()}} for i in range(len(non_zero_action_processed_data))]