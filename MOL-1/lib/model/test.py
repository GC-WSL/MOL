# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from ops.nms import nms
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

import torch
from xml.etree import ElementTree as ET
from shutil import copyfile

from model.apmetric import AveragePrecisionMeter

NWPU_CATEGORIES =["airplane",
        "baseball",
        "basketball",
        "bridge",
        "groundtrackfield",
        "harbor",
        "ship",
        "storagetank",
        "tenniscourt",
        "vehicle"]
DIOR_CATEGORIES =["airplane",
        "airport",
        "baseballfield",
        "basketballcourt",
        "bridge",
        "chimney",
        "dam",
        "expressway-service-area",
        "expressway-toll-station",
        "golffield",
        "groundtrackfield",
        "harbor",
        "overpass",
        "ship",
        "stadium",
        "storagetank",
        "tenniscourt",
        "trainstation",
        "vehicle",
        "windmill"]
def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)
  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def get_ss_boxes(roidb_i, im_scales):
  ss_inds = np.where(roidb_i['gt_classes'] == -1)[0] 
  ss_boxes = np.empty((len(ss_inds), 5), dtype=np.float32)
  ss_boxes[:, 1:] = roidb_i['boxes'][ss_inds,:] * im_scales
  ss_boxes[:, 0] = 0
  return ss_boxes

def im_detect(net, im, roidb_i):
  blobs, im_scales = _get_blobs(im)
  im_blob_scales = blobs['data']
  score_list = []
  pred_box_list = []
  det_cls_list = []
  for index, scale in enumerate(im_scales):
      im_blob = im_blob_scales[index]
      ss_boxes = get_ss_boxes(roidb_i, scale)
      im_blob = im_blob[np.newaxis, :,:,:]
      img_info = np.array([im_blob.shape[1], im_blob.shape[2], scale], dtype=np.float32)
      bbox_pred, rois ,det_cls_prob , det_cls_prob_product, refine_prob_1, refine_prob_2, _ = net.test_image(im_blob, img_info, ss_boxes)
      boxes = rois[:, 1:5] / scale
      scores = np.reshape((refine_prob_1+refine_prob_2)/2, [det_cls_prob_product.shape[0], -1])
      bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
      if cfg.TEST.BBOX_REG:
          box_deltas = bbox_pred
          pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
          pred_boxes = _clip_boxes(pred_boxes, im.shape)
      else:
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      score_list.append(scores)
      pred_box_list.append(pred_boxes)
      det_cls_list.append(det_cls_prob)

  scores = np.array(score_list).mean(axis=0)
  pred_boxes = np.array(pred_box_list).mean(axis=0)
  det_cls_prob = np.array(det_cls_list).mean(axis=0)
  target = np.reshape(roidb_i['image_level_labels'], (-1))
  return scores, pred_boxes, det_cls_prob, target

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue
      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1))[0]
      dets = dets[inds,:]
      if dets == []:
        continue
      keep = nms(torch.from_numpy(dets), thresh).numpy()
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes


def generate_pseudo_label(cls_dets, img_path, output_dir):
  image = cv2.imread(img_path)
  copyfile('./empty.xml','./temp.xml')
  temp=ET.parse(r'./temp.xml')
  filename = temp.find('filename')
  filename.text = '{0}.jpg'.format((img_path.split('/')[-1]).split('.')[0])
  width = temp.find('size').find('width')
  height = temp.find('size').find('height')
  depth = temp.find('size').find('depth')
  width.text = '{0}'.format(image.shape[1])
  height.text = '{0}'.format(image.shape[0])
  depth.text = '3'
  root = temp.getroot()
  template = "{}: {:.2f}"
  for label in range(len(cls_dets)):
      boxes = cls_dets[label][0]
      if boxes==[]:
          continue
      for box_id in range(boxes.shape[0]):
          score = boxes[box_id,-1]
          if score<0.5:
              continue
          bbox = boxes[box_id,:-1].astype(int)
          x, y = bbox[:2]
          s = template.format(NWPU_CATEGORIES[label], score)
          top_left, bottom_right = bbox[:2].tolist(), bbox[2:].tolist()
          obj = ET.Element('object')
          ET.SubElement(obj, 'name').text = NWPU_CATEGORIES[label]
          ET.SubElement(obj, 'pose').text = 'Unspecified'
          ET.SubElement(obj, 'truncated').text = '0'
          ET.SubElement(obj, 'difficult').text = '0'
          bndbox = ET.SubElement(obj, 'bndbox')
          ET.SubElement(bndbox, 'xmin').text = '{0}'.format(top_left[0])
          ET.SubElement(bndbox, 'ymin').text = '{0}'.format(top_left[1])
          ET.SubElement(bndbox, 'xmax').text = '{0}'.format(bottom_right[0])
          ET.SubElement(bndbox, 'ymax').text = '{0}'.format(bottom_right[1])
          root.append(obj) 
  if not os.path.exists("{0}/{1}".format(output_dir,'pseudo_label')):
      os.makedirs("{0}/{1}".format(output_dir,'pseudo_label'))    
  temp.write("{0}/{1}/{2}.xml".format(output_dir,'pseudo_label',(img_path.split('/')[-1]).split('.')[0]), encoding='utf-8', xml_declaration=True)
  

def test_net(net, imdb, roidb, weights_filename, max_per_image=100, thresh=0., use_corloc=False):
  np.random.seed(cfg.RNG_SEED)
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]
  output_dir = get_output_dir(imdb, weights_filename)
  
  ap_meter = AveragePrecisionMeter(difficult_examples=True)
  ap_meter.reset()
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  for i in range(num_images):
    pseudo_boxes = [[] for _ in range(imdb.num_classes)]
    im = cv2.imread(imdb.image_path_at(i))
    img_info = imdb.image_path_at(i)
    _t['im_detect'].tic()
    scores, boxes , det_cls_prob, target = im_detect(net, im, roidb[i])
    _t['im_detect'].toc()
    _t['misc'].tic()
    output = np.reshape(det_cls_prob[1:],(1,-1))
    target = np.reshape(target[:], (1, -1))
    ap_meter.add(output, target)
    for j in range(0, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
      if use_corloc:
          keep = np.argmax(cls_dets[:, -1])
          keep = torch.tensor([keep]).reshape(-1,1) if cls_dets.size > 0 else []
          cls_dets = cls_dets[keep, :].reshape(-1,5)
      else:
          _,keep=nms(torch.from_numpy(cls_dets), cfg.TEST.NMS)
          keep = keep.numpy() if cls_dets.size > 0 else []
          cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets
      pseudo_boxes[j].append(cls_dets)
     
    #generate_pseudo_label(pseudo_boxes, imdb.image_path_at(i), output_dir)
    
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time(),
            _t['misc'].average_time()))
  
  
  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir, use_corloc)

