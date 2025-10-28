import os
import argparse
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import json
from multiprocessing import Pool
from tqdm import tqdm
import shutil
import numpy as np
import torch

def to_tensor_column(column, dtype=torch.float32):
    return torch.stack([
        torch.as_tensor(x, dtype=dtype) for x in column
    ])

def check_value(value, feature_info):
    # ---- 1. Convert list → numpy ----
    if isinstance(value, list):
        value = np.array(value)

    # ---- 2. Convert torch.Tensor → numpy ----
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

        
    # ---- 3. Handle non-numeric like strings ----
    if isinstance(value, str):
        return value

    if "dtype" in feature_info:
        if feature_info["dtype"] not in ["image", "video"]:
            target_dtype = np.dtype(feature_info["dtype"])
            if value.dtype != target_dtype:
                value = value.astype(target_dtype)

    # ---- 5. Fix shape ----
    if "shape" in feature_info:
        expected_shape = feature_info["shape"]

        # Handle image
        if feature_info.get("dtype") in ["image", "video"]:
            # Convert (3,H,W) → (H,W,3)
            if value.ndim == 3 and value.shape[0] == 3:
                value = value.transpose(1, 2, 0)
            
        # Handle 1D vectors (like state/actions)
        elif len(expected_shape) == 1:
            # print(value)
            value = value.reshape(-1)
            # value = torch.tensor(value)
            # if len(value) > expected_shape[0]:
            #     value = value[:expected_shape[0]]  # truncate
            # elif len(value) < expected_shape[0]:
            #     pad = np.zeros(expected_shape[0] - len(value), dtype=value.dtype)
            #     value = np.concatenate([value, pad])
               
    return value


def parse_step(item, rename_map, merge_map, features, ignore_fields=["index", "episode_index", "frame_index", "task_index", "timestamp"]):
    """
    Parse a single step of data:
      1. Rename specific fields using rename_map
      2. Merge multiple fields into one using merge_map

    Args:
        item (dict): Input data dictionary for one step
        rename_map (dict): Mapping of old field names to new field names
        merge_map (dict): Mapping of new field names to a list of old field names to merge

    Returns:
        dict: Parsed data dictionary
    """
    result = {}
    merge_source_fields = {field for fields in merge_map.values() for field in fields}

    # Step 1: Rename fields
    for key, value in item.items():
        if key in merge_source_fields or key in ignore_fields:
            continue
        # Replace key if it is in rename_map, otherwise keep original
        new_key = rename_map.get(key, key)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        if new_key not in ["task", "prompt"]:
            value = check_value(value, features[new_key])

        result[new_key] = value

    # Step 2: Merge fields
    for merged_key, source_keys in merge_map.items():
        merged = []
        for src_key in source_keys:
            if src_key not in item:
                raise KeyError(f"Missing field '{src_key}' for merging into '{merged_key}'")
            src_value = item[src_key]
            if isinstance(src_value, torch.Tensor):
                src_value = src_value.detach().cpu().numpy()
            merged.append(src_value)

        merged = [np.array(x) for x in merged]
        merged_value = np.concatenate(merged, axis=0)
        
        if merged_key in features:
            merged_value = check_value(merged_value, features[merged_key])

        # Add merged field to result
        result[merged_key] = merged_value

    return result

def update_features(src_features, rename_map, merge_map, update_info, ignore_fields):
    """
    Update dataset feature definitions after renaming and merging fields.

    This function is used to transform the original feature schema (`src_features`)
    into a new schema that matches the transformed data after applying:
      1. Field renaming   (defined in `rename_map`)
      2. Field merging    (defined in `merge_map`)
      3. Schema updates   (defined in `update_info` for merged fields)

    Returns:
        dict: Updated feature schema after applying field renaming and merging.
    """
    dst_features = {}
    merge_source_fields = {field for fields in merge_map.values() for field in fields}
    print(f"merge source fields: {merge_source_fields}")
    for key, value in src_features.items():
        if key in merge_source_fields:
            continue
        if key in ignore_fields:
            continue
        new_key = rename_map.get(key,key)
        dst_features[new_key] = value

    for key, value in update_info.items():
        dst_features[key] = value
    return dst_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="/mnt/petrelfs/share/efm_p/wangfangjing/datasets/3L_data_lerobot/sandwitch_task_merge")
    parser.add_argument("--dst", type=str, default="/mnt/petrelfs/share/efm_p/chenxinyi/3L_data_lerobot/sandwich_task_merge")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output folder if exists")
    args = parser.parse_args()

    if args.overwrite:
        if os.path.exists(args.dst):
            print(f"{args.dst} already exist, delete it")
            shutil.rmtree(args.dst)

    rename_map = {
        # old -> new
        "video.base_view": "image",
        "video.base_view_2": "image_2",
        "video.ego_view": "wrist_image",
        "task": "prompt",
        "task": "task"
    }
    merge_map = {
        "state": [
            "state.joints",
            "state.gripper"
        ],
        "actions": [
            "action.eef_position",
            "action.eef_rotation",
            "action.gripper_close"
        ]
    }

    update_info = {
        "state": {
            "dtype": "float64",
            "shape": (9,),
            "names": [
                "state"
            ]
        },
        "actions": {
            "dtype": "float64",
            "shape": (7,),
            "names": [
                "actions"
            ]
        }
    }

    ignore_fields=["index", "episode_index", "frame_index", "task_index", "timestamp", "action.delta_eef_position", "action.delta_eef_rotation", "state.eef_position", "state.eef_rotation"]

    src_dataset = LeRobotDataset(
        repo_id="None",
        root=args.src
    )

    dst_features = update_features(src_dataset.features, rename_map, merge_map, update_info, ignore_fields)
    print(dst_features.keys())

    dst_dataset = LeRobotDataset.create(
        repo_id="None",
        fps=src_dataset.fps,
        root=args.dst,
        robot_type="franka",
        features=dst_features
    )

    current_episode_index = 0
    total = len(src_dataset)
    for data_item in tqdm(src_dataset, total=total, desc="Processing frames"):
        if data_item["episode_index"] != current_episode_index:
            dst_dataset.save_episode()
            current_episode_index = data_item["episode_index"]
        cur_frame_data = parse_step(data_item, rename_map, merge_map, dst_features, ignore_fields)
        dst_dataset.add_frame(
            cur_frame_data
        )
    dst_dataset.save_episode()

if __name__ == "__main__":
    main()
