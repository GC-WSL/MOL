#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json

COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "baseball"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "basketball"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "bridge"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "groundtrackfield"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "harbor"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "ship"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "storagetank"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "tenniscourt"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "vehicle"},
]
'''
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "airport"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "baseballfield"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "basketballcourt"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "bridge"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "chimney"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "dam"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "expressway-service-area"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "expressway-toll-station"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "golffield"},
    {"color": [220, 20, 60], "isthing": 1, "id": 11, "name": "groundtrackfield"},
    {"color": [119, 11, 32], "isthing": 1, "id": 12, "name": "harbor"},
    {"color": [0, 0, 142], "isthing": 1, "id": 13, "name": "overpass"},
    {"color": [0, 0, 230], "isthing": 1, "id": 14, "name": "ship"},
    {"color": [106, 0, 228], "isthing": 1, "id": 15, "name": "stadium"},
    {"color": [0, 60, 100], "isthing": 1, "id": 16, "name": "storagetank"},
    {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "tenniscourt"},
    {"color": [0, 0, 70], "isthing": 1, "id": 18, "name": "trainstation"},
    {"color": [0, 0, 192], "isthing": 1, "id": 19, "name": "vehicle"},
    {"color": [250, 170, 30], "isthing": 1, "id": 20, "name": "windmill"},
]
'''

# fmt: off
COCO_PERSON_KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
# fmt: on

# Pairs of keypoints that should be exchanged under horizontal flipping
COCO_PERSON_KEYPOINT_FLIP_MAP = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)

# rules for pairs of keypoints to draw a line between, and the line color to use.
KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    ("left_hip", "right_hip", (255, 102, 0)),
    ("left_hip", "left_knee", (255, 255, 77)),
    ("right_hip", "right_knee", (153, 255, 204)),
    ("left_knee", "left_ankle", (191, 255, 128)),
    ("right_knee", "right_ankle", (255, 195, 77)),
]


def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 10, len(thing_ids)#NWPU 10 DIOR 20
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_coco_panoptic_separated_meta():
    """
    Returns metadata for "separated" version of the panoptic segmentation dataset.
    """
    stuff_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    assert len(stuff_ids) == 53, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 53], used in models) to ids in the dataset (used for processing results)
    # The id 0 is mapped to an extra category "thing".
    stuff_dataset_id_to_contiguous_id = {k: i + 1 for i, k in enumerate(stuff_ids)}
    # When converting COCO panoptic annotations to semantic annotations
    # We label the "thing" category to 0
    stuff_dataset_id_to_contiguous_id[0] = 0

    # 54 names for COCO stuff categories (including "things")
    stuff_classes = ["things"] + [
        k["name"].replace("-other", "").replace("-merged", "")
        for k in COCO_CATEGORIES
        if k["isthing"] == 0
    ]

    # NOTE: I randomly picked a color for things
    stuff_colors = [[82, 18, 128]] + [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    ret.update(_get_coco_instances_meta())
    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == "coco":
        return _get_coco_instances_meta()
    if dataset_name == "coco_panoptic_separated":
        return _get_coco_panoptic_separated_meta()
    elif dataset_name == "coco_person":
        return {
            "thing_classes": ["person"],
            "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
            "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
            "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
        }
    elif dataset_name == "cityscapes":
        # fmt: off
        CITYSCAPES_THING_CLASSES = [
            "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle",
        ]
        CITYSCAPES_STUFF_CLASSES = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle", "license plate",
        ]
        # fmt: on
        return {
            "thing_classes": CITYSCAPES_THING_CLASSES,
            "stuff_classes": CITYSCAPES_STUFF_CLASSES,
        }
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
