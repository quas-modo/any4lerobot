REALWORLD_FRANKA_FEATURES = {
    "observation.images.image_0": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.image_1": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float64",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]},
    },
    "observation.states.ee_state": {
        "dtype": "float64",
        "shape": (6,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw"]},
    },
    "observation.states.joint_state": {
        "dtype": "float64",
        "shape": (7,),
        "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]},
    },
    "action": {
        "dtype": "float64",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
    },
}