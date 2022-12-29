#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
#from model.nms_wrapper import nms

from ops.nms import nms
from datasets.factory import get_imdb


from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import MELM_vgg16
from nets.resnet_v1 import resnetv1

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CLASSES = ('__background__',"airplane",
        "baseball",
        "basketball",
        "bridge",
        "groundtrackfield",
        "harbor",
        "ship",
        "storagetank",
        "tenniscourt",
        "vehicle")
CLASSES = ("airplane",
        "baseball",
        "basketball",
        "bridge",
        "groundtrackfield",
        "harbor",
        "ship",
        "storagetank",
        "tenniscourt",
        "vehicle")
class_to_ind = dict(list(zip(CLASSES, list(range(10)))))
NETS = {'vgg16_MELM': ('vgg16_MELM_iter_%d.pth',),'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS= {'NWPU': ('NWPU_test',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        #print('hahaha')
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def compute_colors_for_labels(labels):
    """ Simple function that adds fixed colors depending on the class """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("bool")
    return colors

def vis_results(img, alldets, image_name, thresh=0.5):
    out_path = './vis/melm'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    template = "{}: {:.2f}"
    #import ipdb
    #ipdb.set_trace()
    for box, cls in alldets:
        inds = np.where(box[:,-1] >= thresh)[0]
        box=box[inds]
        #color = compute_colors_for_labels(class_to_ind[cls]).tolist()
        for b in box:
            score=b[-1]
            b=np.array(b).astype(dtype=int)
            top_left, bottom_right = b[:2].tolist(), b[2:-1].tolist()
            #print(top_left, bottom_right)
            img = cv2.rectangle(img, tuple(top_left), tuple(bottom_right),(0, 0, 255), 2)
            x, y = b[:2]
            s = template.format(cls, score)
            cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
    cv2.imwrite(os.path.join(out_path, image_name.split('/')[-1]), img)


def demo(net, image_name, roidb):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes,_,_ = im_detect(net, im, roidb)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.3
    NMS_THRESH = 0.3
    all_dets = []
    for cls_ind, cls in enumerate(CLASSES):
        #cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        _,keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        all_dets.append((dets,cls))
    vis_results(im, all_dets, image_name, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [NWPU pascal_voc_0712]',
                        choices='NWPU', default='NWPU')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    saved_model = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0] %(35160 if dataset == 'NWPU' else 110000))


    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))
    
    if args.dataset == 'NWPU':
        test_name = 'NWPU_test'
    
    imdb = get_imdb(test_name)
    roidb = imdb.roidb
    #imdb.competition_mode(args.comp_mode)
    # load network
    if demonet == 'vgg16_MELM':
        net = MELM_vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(10,
                          tag='default', anchor_scales=[8, 16, 32])

    net.load_state_dict(torch.load(saved_model, map_location=lambda storage, loc: storage))

    net.eval()
    if not torch.cuda.is_available():
        net._device = 'cpu'
    net.to(net._device)

    print('Loaded network {:s}'.format(saved_model))
    
    
    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']
    #for im_name in im_names:
    #    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #    print('Demo for data/demo/{}'.format(im_name))
    #    demo(net, im_name)
    
    #index = np.random.randint(0, len(imdb.image_index), 1)
    
    index = np.arange(0, len(imdb.image_index),1)
    
    for i in index:
        im_path = imdb.image_path_at(i)
        demo(net, im_path, roidb[i])    
    plt.show()
