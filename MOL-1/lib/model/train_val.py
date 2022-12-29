# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorboardX as tb

from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
import utils.timer
try:
  import cPickle as pickle
except ImportError:
  import pickle

import torch

import torch.optim as optim
import numpy as np
import os
import glob
import time
from tqdm import tqdm
import math
def scale_lr(optimizer, scale):
  """Scale the learning rate of the optimizer"""
  for param_group in optimizer.param_groups:
    param_group['lr'] *= scale

class SolverWrapper(object):
  """
    A wrapper class for the training process
  """

  def __init__(self, network, imdb, roidb, valroidb, output_dir, tbdir, pretrained_model=None, momentum_net=None):
    self.net = network
    if cfg.USE_MOMENTUM:
      self.momentum_net = momentum_net
    self.imdb = imdb
    self.roidb = roidb
    self.valroidb = valroidb
    self.output_dir = output_dir
    self.tbdir = tbdir
    self.tbvaldir = tbdir + '_val'
    if not os.path.exists(self.tbvaldir):
      os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model
    self.img_proposal_history = dict()
    self.kl_history = dict()
    self.prposal_bboxes=dict()

  def snapshot(self, iter):

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pth'
    filename = os.path.join(self.output_dir, filename)
    torch.save(self.net.state_dict(), filename)
    print('Wrote snapshot to: {:s}'.format(filename))
    if cfg.USE_MOMENTUM:
      filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_momentum{:d}'.format(iter) + '.pth'
      filename = os.path.join(self.output_dir, filename)
      torch.save(self.momentum_net.state_dict(), filename)
      print('Wrote snapshot to: {:s}'.format(filename))
    # Also store some meta information, random state, etc.
    nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(self.output_dir, nfilename)
    # current state of numpy random
    st0 = np.random.get_state()
    # current position in the database
    cur = self.data_layer._cur
    # current shuffled indexes of the database
    perm = self.data_layer._perm
    # current position in the validation database
    cur_val = self.data_layer_val._cur
    # current shuffled indexes of the validation database
    perm_val = self.data_layer_val._perm

    # Dump the meta info
    with open(nfilename, 'wb') as fid:
      pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

    return filename, nfilename

  def from_snapshot(self, sfile, nfile):
    print('Restoring model snapshots from {:s}'.format(sfile))
    self.net.load_state_dict(torch.load(str(sfile)))
    print('Restored.')
    with open(nfile, 'rb') as fid:
      st0 = pickle.load(fid)
      cur = pickle.load(fid)
      perm = pickle.load(fid)
      cur_val = pickle.load(fid)
      perm_val = pickle.load(fid)
      last_snapshot_iter = pickle.load(fid)

      np.random.set_state(st0)
      self.data_layer._cur = cur
      self.data_layer._perm = perm
      self.data_layer_val._cur = cur_val
      self.data_layer_val._perm = perm_val

    return last_snapshot_iter

  def construct_graph(self):
    # Set the random seed
    torch.manual_seed(cfg.RNG_SEED)
    # Build the main computation graph
    self.net.create_architecture(self.imdb.num_classes, tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
    if cfg.USE_MOMENTUM:
      self.momentum_net.create_architecture(self.imdb.num_classes, tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
    # Set learning rate and momentum
    lr = cfg.TRAIN.LEARNING_RATE
    params = []
    for key, value in dict(self.net.named_parameters()).items():
      if value.requires_grad:
        if 'refine' in key and 'bias' in key:
          params += [{'params':[value],'lr':10*lr*(cfg.TRAIN.DOUBLE_BIAS + 1), 'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        elif 'refine' in key and 'bias' not in key:
          params += [{'params':[value],'lr':10*lr, 'weight_decay': getattr(value, 'weight_decay', cfg.TRAIN.WEIGHT_DECAY)}]
        elif 'refine' not in key and 'bias' in key:
          params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), 'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else:
          params += [{'params':[value],'lr':lr, 'weight_decay': getattr(value, 'weight_decay', cfg.TRAIN.WEIGHT_DECAY)}]
    self.optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    self.writer = tb.writer.FileWriter(self.tbdir)
    self.valwriter = tb.writer.FileWriter(self.tbvaldir)

    return lr, self.optimizer

  def find_previous(self):
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pth')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in pytorch
    redfiles = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      redfiles.append(os.path.join(self.output_dir, 
                      cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.pth'.format(stepsize+1)))
    sfiles = [ss for ss in sfiles if ss not in redfiles]

    nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)
    redfiles = [redfile.replace('.pth', '.pkl') for redfile in redfiles]
    nfiles = [nn for nn in nfiles if nn not in redfiles]

    lsf = len(sfiles)
    assert len(nfiles) == lsf
    return lsf, nfiles, sfiles

  def initialize(self):
    # Initial file lists are empty
    np_paths = []
    ss_paths = []
    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(self.pretrained_model))
    self.net.load_pretrained_cnn(torch.load(self.pretrained_model))
    if cfg.USE_MOMENTUM:
      self.momentum_net.load_pretrained_cnn(torch.load(self.pretrained_model))
    print('Loaded.')

    last_snapshot_iter = 0
    lr = cfg.TRAIN.LEARNING_RATE
    stepsizes = list(cfg.TRAIN.STEPSIZE)

    return lr, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def restore(self, sfile, nfile):
    # Get the most recent snapshot and restore
    np_paths = [nfile]
    ss_paths = [sfile]
    # Restore model from snapshots
    last_snapshot_iter = self.from_snapshot(sfile, nfile)
    # Set the learning rate
    lr_scale = 1
    stepsizes = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      if last_snapshot_iter > stepsize:
        lr_scale *= cfg.TRAIN.GAMMA
      else:
        stepsizes.append(stepsize)
    scale_lr(self.optimizer, lr_scale)
    lr = cfg.TRAIN.LEARNING_RATE * lr_scale
    return lr, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def remove_snapshot(self, np_paths, ss_paths):
    to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      nfile = np_paths[0]
      os.remove(str(nfile))
      np_paths.remove(nfile)

    to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      sfile = ss_paths[0]
      os.remove(str(sfile))
      ss_paths.remove(sfile)

  def train_model(self, max_iters):
    # Build data layers for both training and validation set
    self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
    self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)

    # Construct the computation graph
    lr, train_op = self.construct_graph()

    # Find previous snapshots if there is any to restore from
    lsf, nfiles, sfiles = self.find_previous()

    # Initialize the variables or restore them from the last snapshot
    if lsf == 0:
      lr, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize()
    else:
      lr, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(str(sfiles[-1]), 
                                                                             str(nfiles[-1]))
    iter = last_snapshot_iter + 1
    last_summary_time = time.time()
    # Make sure the lists are not empty
    stepsizes.append(max_iters)
    stepsizes.reverse()
    next_stepsize = stepsizes.pop()

    self.net.train()
    self.net.to(self.net._device)
    if cfg.USE_MOMENTUM:
      self.momentum_net.train()
      self.momentum_net.to(self.momentum_net._device)

    pbar = tqdm(total = max_iters)
    while iter < max_iters + 1:
      pbar.update(1)
      # Learning rate
      if iter == next_stepsize + 1:
        self.snapshot(iter)
        lr *= cfg.TRAIN.GAMMA
        scale_lr(self.optimizer, cfg.TRAIN.GAMMA)
        next_stepsize = stepsizes.pop()
      utils.timer.timer.tic()
      blobs, db_inds = self.data_layer.forward()
      img_info = self.roidb[db_inds[0]]['image']
      ss_num = len(blobs['ss_boxes'])
      if not img_info in self.img_proposal_history:
        self.img_proposal_history[img_info] = []
        self.prposal_bboxes[img_info] = blobs['ss_boxes']
      now = time.time()
      if iter == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
        cls_det_loss, refine_loss_1, refine_loss_2, total_loss,objs_p  = \
          self.net.train_step_with_summary(blobs, self.optimizer, self.img_proposal_history[img_info])
        last_summary_time = now
      else:
        cls_det_loss, refine_loss_1, refine_loss_2, total_loss, objs_p = \
         self.net.train_step(blobs, self.optimizer, self.img_proposal_history[img_info])
      utils.timer.timer.toc()
      
      if cfg.USE_MOMENTUM:
        with torch.no_grad():
            for param_net, param_net_momentum in zip(self.net.parameters(), self.momentum_net.parameters()):
                param_net_momentum.data.mul_(0.9).add_(0.1*param_net)
            bbox_pred, rois ,det_cls_prob , det_cls_prob_product, refine_prob_1, refine_prob_2, objs_p = self.momentum_net.test_image(blobs['data'], blobs['im_info'], blobs['ss_boxes'])
            self.img_proposal_history[img_info].append(objs_p)
            if len(self.img_proposal_history[img_info])>2:
                self.img_proposal_history[img_info].pop(0)

      if iter % (cfg.TRAIN.DISPLAY) == 0:
        print('iter: %d / %d, total loss: %.6f\n >>> cls_det_loss: %.6f\n '
              '>>> refine_loss_1: %.6f\n >>> refine_loss_2: %.6f\n'
              '>>> lr: %f' % \
              (iter, max_iters, total_loss, cls_det_loss, refine_loss_1, refine_loss_2, lr))
        print('speed: {:.3f}s / iter'.format(utils.timer.timer.average_time()))

      # Snapshotting
      if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = iter
        ss_path, np_path = self.snapshot(iter)
        np_paths.append(np_path)
        ss_paths.append(ss_path)

        # Remove the old snapshots if there are too many
        if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          self.remove_snapshot(np_paths, ss_paths)

      iter += 1

    if last_snapshot_iter != iter - 1:
      self.snapshot(iter - 1)

    self.writer.close()
    self.valwriter.close()
    

def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb


def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""

  def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    #print(entry)
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb


def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000, momentum_network=None):
  """Train a Faster R-CNN network."""
  roidb = filter_roidb(roidb)
  valroidb = filter_roidb(valroidb)

  if not cfg.USE_MOMENTUM:
    sw = SolverWrapper(network, imdb, roidb, valroidb, output_dir, tb_dir,
                     pretrained_model=pretrained_model)
  else:
    sw = SolverWrapper(network, imdb, roidb, valroidb, output_dir, tb_dir,
                     pretrained_model=pretrained_model, momentum_net=momentum_network)

  print('Solving...')
  sw.train_model(max_iters)
  print('done solving')
