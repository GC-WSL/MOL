#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii Inc. and its affiliates.

from PIL import Image
import torchvision.transforms as tfs

from cvpods.data.datasets import COCODataset
from cvpods.data.registry import DATASETS, PATH_ROUTES
import xml.etree.ElementTree as ET
import numpy as np
_UPDATE_DICT = {
    "coco_NWPU_train":
        ("NWPU/train2017", "NWPU/annotations/NWPU_trainval.json"),
    "coco_NWPU_test":
        ("NWPU/val2017", "NWPU/annotations/NWPU_test.json"),
}


PATH_ROUTES.get("COCO")["coco"].update(_UPDATE_DICT)
PATH_ROUTES.get("COCO")["dataset_type"] = "NWPU"

CLASS_TO_IND = dict(list(zip(("airplane","baseball", "basketball", "bridge", "groundtrackfield", "harbor", "ship","storagetank", "tenniscourt", "vehicle"), list(range(10)))))
        
@DATASETS.register()
class NWPU(COCODataset):
    def __getitem__(self, index):
        dataset_dict = super().__getitem__(index)
        image_name = dataset_dict["file_name"]
        image_level_gt = self.load_label((image_name.split('/')[-1]).split('.')[0])
        dataset_dict["image_level_gt"] = image_level_gt
        image = dataset_dict["image"]
        image = image.permute(1, 2, 0).numpy()

        color_jitter = tfs.ColorJitter(1.2, 1.2, 1.2, 0.2)
        color_jitter_img = tfs.RandomApply([color_jitter], p=1.0)(Image.fromarray(image))
        dataset_dict["image_color"] = tfs.ToTensor()(color_jitter_img) * 255
        
        rerase = tfs.RandomErasing(p=1.0)
        rerase_img = rerase(tfs.ToTensor()(Image.fromarray(image)) * 255)
        dataset_dict["image_erase"] = rerase_img
        return dataset_dict
    def load_label(self, image_name):
        image_level_gt = np.zeros(10, dtype=np.float32)
        path = '../MOL-1/data/NWPU/Annotations/{}.xml'.format(image_name)
        tree = ET.parse(path)
        objs = tree.findall('object')
        for ix, obj in enumerate(objs):
            image_level_gt[CLASS_TO_IND[obj.find('name').text.lower().strip()]]=1
        return image_level_gt
    
    
    
    
    
    
