# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from .dis_eval import dis_eval
from model.config import cfg


class DIOR(imdb):
  def __init__(self, image_set, use_diff=False):
    name = 'DIOR_{}'.format(image_set)
    if use_diff:
      name += '_diff'
    imdb.__init__(self, name)
    self._image_set = image_set
    self._devkit_path = self._get_default_path()
    print(self._devkit_path)
    self._data_path = os.path.join(self._devkit_path, 'DIOR')
    self._classes = ("airplane",
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
        "windmill")
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.jpg'
    self._image_index = self._load_image_set_index()
    self._roidb_handler = self.selective_search_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': use_diff,
                   'matlab_eval': False,
                   'rpn_file': None,
                   'min_size': 20}

    assert os.path.exists(self._devkit_path), \
      'VOCdevkit path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, 'JPEGImages',
                              index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return cfg.DATA_DIR

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_pascal_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def selective_search_roidb(self):
    """
    Return the database of selective search regions of interest.
    Ground-truth ROIs are also included.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path,
                              self.name + '_selective_search_roidb.pkl')

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidb = pickle.load(fid)
        print('{} ss roidb loaded from {}'.format(self.name, cache_file))
        return roidb

    if self._image_set != 'test':
        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
    else:
        roidb = self._load_selective_search_roidb(None)
    with open(cache_file, 'wb') as fid:
        pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote ss roidb to {}'.format(cache_file))

    return roidb

  def rpn_roidb(self):
    if self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_selective_search_roidb(self, gt_roidb):
    filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                            'selective_search_data',
                                            self.name + '.mat'))
    assert os.path.exists(filename), \
        'Selective search data not found at: {}'.format(filename)
        
    raw_data = sio.loadmat(filename)['boxes'].ravel()
    raw_data_name = sio.loadmat(filename)['images'].ravel()
    raw_name = raw_data_name.tolist()
    box_list = []
    for i in range(len(self._image_index)):
        ind =raw_name.index(self._image_index[i])
        boxes = raw_data[ind][:, (1, 0, 3, 2)] - 1
        keep = ds_utils.unique_boxes(boxes)
        boxes = boxes[keep, :]
        keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
        boxes = boxes[keep, :]
        
        boxes_index = np.arange(boxes.shape[0])
        box_num = min(2000, boxes.shape[0])
        boxes_index = np.random.choice(boxes_index, size=box_num, replace=False)
        boxes = boxes[boxes_index]
        
        box_list.append(boxes)
       
    return self.create_roidb_from_box_list(box_list, gt_roidb)


  def _load_pascal_annotation(self, index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    if not self.config['use_diff']:
      non_diff_objs = [
        obj for obj in objs]
      objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    
    image_level_labels = -np.ones((1, self.num_classes), dtype=np.int32)
    for ix, obj in enumerate(objs):
      bbox = obj.find('bndbox')
      x1 = float(bbox.find('xmin').text) - 1
      y1 = float(bbox.find('ymin').text) - 1
      x2 = float(bbox.find('xmax').text) - 1
      y2 = float(bbox.find('ymax').text) - 1
      cls = self._class_to_ind[obj.find('name').text.lower().strip()]
      image_level_labels[0][cls] = 1
      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = cls
      overlaps[ix, cls] = 1.0
      seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas,
            'image_level_labels':image_level_labels}

  def _load_pascal_annotation_retrain(self, index):
    gt_folder = '/root/Cloud/G5/MELM-VGG16/output/VOC2007/VGG16/PsuedoGT/Detections/'
    print('Loading ', gt_folder, ': ', index)
    filename = os.path.join(gt_folder, index + '.txt')
    objs = np.loadtxt(filename)
    if objs.ndim == 1:
      num_objs = 1
      objs = objs.reshape(1,-1)
    else:
      num_objs = objs.shape[0]

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    for i in range(num_objs):
      x1 = objs[i][1]
      y1 = objs[i][2]
      x2 = objs[i][3]
      y2 = objs[i][4]
      boxes[i, :] = [x1, y1, x2, y2]
      gt_classes[i] = int(objs[i][0])
      overlaps[i, int(objs[i][0])] = 1.0
      seg_areas[i] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes' : boxes,
            'gt_classes': gt_classes,
            'gt_overlaps' : overlaps,
            'flipped' : False,
            'seg_areas': seg_areas}


  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    filename = self._image_set + '_{:s}.txt'
    path = os.path.join(
      self._devkit_path,
      'results',
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} VOC results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def _do_python_eval(self, output_dir='output',use_corloc=False):
    annopath = os.path.join(
      self._devkit_path,
      'DIOR',
      'Annotations',
      '{:s}.xml')
    imagesetfile = os.path.join(
      self._devkit_path,
      'DIOR',
      'ImageSets',
      'Main',
      self._image_set + '.txt')
    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    aps = []
    CorLocs = []
    use_07_metric = True
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      filename = self._get_voc_results_file_template().format(cls)
      if use_corloc:
        CorLoc = dis_eval(filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
        CorLocs += [CorLoc]
        print(('CorLoc for {} = {:.4f}'.format(cls, CorLoc)))
      else:
        rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
        aps += [ap]
        print(('AP for {} = {:.4f}'.format(cls, ap)))
    
    if use_corloc:
      print(('Mean CorLoc = {:.4f}'.format(np.mean(CorLocs))))
      print('~~~~~~~~')
      print('Results:')
      for CorLoc in CorLocs:
        print(('{:.3f}'.format(CorLoc)))
      print(('{:.3f}'.format(np.mean(CorLocs))))
    else:
      print(('Mean AP = {:.4f}'.format(np.mean(aps))))
      print('~~~~~~~~')
      print('Results:')
      for ap in aps:
        print(('{:.3f}'.format(ap)))
      print(('{:.3f}'.format(np.mean(aps))))
    
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir, use_corloc):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir, use_corloc)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


if __name__ == '__main__':
  from datasets.pascal_voc import pascal_voc

  d = pascal_voc('trainval', '2007')
  res = d.roidb
  from IPython import embed;

  embed()
