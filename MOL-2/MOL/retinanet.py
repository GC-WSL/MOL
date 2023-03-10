# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List
import torch
import cv2
import numpy as np
from cvpods.modeling.losses import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn

from cvpods.layers import ShapeSpec, batched_nms, cat
from cvpods.structures import Boxes, ImageList, Instances, pairwise_iou
from cvpods.utils import log_first_n
from cvpods.modeling.box_regression import Box2BoxTransform
from matcher import Matcher
from cvpods.modeling.postprocessing import detector_postprocess

from cvpods.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls,
                                                  box_delta,
                                                  num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


class RetinaNet(nn.Module):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """
    def __init__(self, cfg, training=True):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features = cfg.MODEL.RETINANET.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        self.pseudo_score_thres = cfg.MODEL.RETINANET.PSEUDO_SCORE_THRES
        # Inference parameters:
        self.score_threshold = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = RetinaNetHead(cfg, feature_shapes, training)
        self.anchor_generator = cfg.build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        
        
        self.use_vflip=True

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training:
            images, images_view1, images_view2 = self.preprocess_image(batched_inputs, self.training)
        else:
            images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10)
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        image_level_gt = [x["image_level_gt"] for x in batched_inputs]
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        if self.training:
            features_view1 = self.backbone(images_view1.tensor)
            features_view2 = self.backbone(images_view2.tensor)
            features_view1 = [features_view1[f] for f in self.in_features]
            features_view2 = [features_view2[f] for f in self.in_features]
            anchors_view1 = self.anchor_generator(features_view1)
            anchors_view2 = self.anchor_generator(features_view2)
            box_cls, box_delta, box_cls_view1, box_delta_view1, box_cls_view2, box_delta_view2 = self.head(features, features_view1, features_view2)

            with torch.no_grad():
                head0_preds = self.pseudo_gt_generate(
                    [cr.clone() for cr in box_cls],
                    [br.clone() for br in box_delta],
                    anchors,
                    images
                )
                head1_preds = self.pseudo_gt_generate(
                    [cr.clone() for cr in box_cls_view1],
                    [br.clone() for br in box_delta_view1],
                    anchors_view1,
                    images_view1
                )
                head2_preds = self.pseudo_gt_generate(
                    [cr.clone() for cr in box_cls_view2],
                    [br.clone() for br in box_delta_view2],
                    anchors_view2,
                    images_view2
                )
                
                gt_instances0 = self.merge_ground_truth(gt_instances, head2_preds, images)
                gt_instances0 = self.merge_ground_truth(gt_instances0, head1_preds, images)

                gt_instances1 = self.merge_ground_truth(gt_instances, head0_preds, images)
                gt_instances1 = self.merge_ground_truth(gt_instances1, head2_preds, images)

                gt_instances2 = self.merge_ground_truth(gt_instances, head0_preds, images)
                gt_instances2 = self.merge_ground_truth(gt_instances2, head1_preds, images)

                gt_classes0, gt_anchors_reg_deltas0 = self.get_ground_truth(anchors, gt_instances0, images)
                gt_classes1, gt_anchors_reg_deltas1 = self.get_ground_truth(anchors_view1, gt_instances1, images)
                gt_classes2, gt_anchors_reg_deltas2 = self.get_ground_truth(anchors_view2, gt_instances2, images)

            head0_losses = self.losses_head0(gt_classes0, gt_anchors_reg_deltas0, box_cls, box_delta, image_level_gt)
            head1_losses = self.losses_head1(gt_classes1, gt_anchors_reg_deltas1, box_cls_view1, box_delta_view1, image_level_gt)
            head2_losses = self.losses_head2(gt_classes2, gt_anchors_reg_deltas2, box_cls_view2, box_delta_view2, image_level_gt)

            return dict(head0_losses, **head1_losses, **head2_losses)
        else:
            box_cls, box_delta = self.head(features, training=self.training)
            results = self.inference(box_cls, box_delta, anchors, images)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    @torch.no_grad()
    def merge_ground_truth(self, targets, predictions, images, iou_thresold=0.4):
        new_targets = []
        for targets_per_image, predictions_per_image, image in zip(targets, predictions, images):
            image_size = image.shape[1:3]
            iou_matrix = pairwise_iou(targets_per_image.gt_boxes,
                                      predictions_per_image.pred_boxes)
            iou_filter = iou_matrix > iou_thresold

            target_class_list = (targets_per_image.gt_classes).reshape(-1, 1)
            pred_class_list = (predictions_per_image.pred_classes).reshape(1, -1)
            class_filter = target_class_list == pred_class_list

            final_filter = iou_filter & class_filter
            unlabel_idxs = torch.sum(final_filter, 0) == 0

            new_target = Instances(image_size)
            new_target.gt_boxes = Boxes.cat([targets_per_image.gt_boxes,
                                             predictions_per_image.pred_boxes[unlabel_idxs]])
            new_target.gt_classes = torch.cat([targets_per_image.gt_classes,
                                               predictions_per_image.pred_classes[unlabel_idxs]])
            
            new_targets.append(new_target)

        return new_targets

    def losses_head0(self, gt_classes, gt_anchors_deltas, pred_class_logits,
               pred_anchor_deltas, image_level_gt):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.
        
        gt_image_label = torch.tensor(image_level_gt).to(gt_classes.device)
        for i in range(gt_image_label.shape[0]):
            class_label = torch.where(gt_image_label[i]==1)[0]
            flag = torch.ones(gt_classes[i].shape)
            for j in range(class_label.shape[0]):
                flag[gt_classes[i]==class_label[j]]=0
            flag[gt_classes[i]==self.num_classes]=0
            gt_classes[i][flag.bool()]=-1
        
        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground)

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, num_foreground)

        return {"loss_box_cls0": loss_cls, "loss_box_reg0": loss_box_reg}

    def losses_head1(self, gt_classes, gt_anchors_deltas, pred_class_logits,
               pred_anchor_deltas, image_level_gt):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.
        
        gt_image_label = torch.tensor(image_level_gt).to(gt_classes.device)
        for i in range(gt_image_label.shape[0]):
            class_label = torch.where(gt_image_label[i]==1)[0]
            flag = torch.ones(gt_classes[i].shape)
            for j in range(class_label.shape[0]):
                flag[gt_classes[i]==class_label[j]]=0
            flag[gt_classes[i]==self.num_classes]=0
            gt_classes[i][flag.bool()]=-1

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground)

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, num_foreground)

        return {"loss_box_cls1": loss_cls, "loss_box_reg1": loss_box_reg}

    def losses_head2(self, gt_classes, gt_anchors_deltas, pred_class_logits,
               pred_anchor_deltas, image_level_gt):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.
        
        gt_image_label = torch.tensor(image_level_gt).to(gt_classes.device)
        for i in range(gt_image_label.shape[0]):
            class_label = torch.where(gt_image_label[i]==1)[0]
            flag = torch.ones(gt_classes[i].shape)
            for j in range(class_label.shape[0]):
                flag[gt_classes[i]==class_label[j]]=0
            flag[gt_classes[i]==self.num_classes]=0
            gt_classes[i][flag.bool()]=-1

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground)

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, num_foreground)

        return {"loss_box_cls2": loss_cls, "loss_box_reg2": loss_box_reg}
    @torch.no_grad()
    def get_ground_truth(self, anchors, targets, images):
        """
        Args:
            anchors (list[list[Boxes]]): a list of N=#image elements. Each is a
                list of #feature level Boxes. The Boxes contains anchors of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, K-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "K". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
        """
        gt_classes = []
        gt_anchors_deltas = []
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        # list[Tensor(R, 4)], one for each image

        for anchors_per_image, targets_per_image, image in zip(anchors, targets, images):
            image_size = image.shape[1:3]
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes,
                                                anchors_per_image)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            has_gt = len(targets_per_image) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                    anchors_per_image.tensor, matched_gt_boxes.tensor
                )

                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(
                    gt_matched_idxs) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(anchors_per_image.tensor)

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)

    def inference(self, box_cls, box_delta, anchors, images):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(anchors) == len(images)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, anchors_per_image in enumerate(anchors):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, anchors_per_image,
                tuple(image_size))
            results.append(results_per_image)
        return results

    def pseudo_gt_generate(self, box_cls, box_delta, anchors, images):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(anchors) == len(images)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, anchors_per_image in enumerate(anchors):
            image_size = images.image_sizes[img_idx]
            
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            results_per_image = self.pseudo_gt_generate_per_image(
                box_cls_per_image, box_reg_per_image, anchors_per_image,
                tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, anchors, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta,
                                                   anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(
                box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all,
                           self.nms_threshold)
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def pseudo_gt_generate_per_image(self, box_cls, box_delta, anchors, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta,
                                                   anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.pseudo_score_thres
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(
                box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all,
                           self.nms_threshold)
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs, training=False):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        if training:
            images_view1 = [x["image_color"].to(self.device) for x in batched_inputs]
            images_view2 = [x["image_erase"].to(self.device) for x in batched_inputs]
            images_view1 = [self.normalizer(x) for x in images_view1]
            images_view2 = [self.normalizer(x) for x in images_view2]
            images_view1 = ImageList.from_tensors(images_view1, self.backbone.size_divisibility)
            images_view2 = ImageList.from_tensors(images_view2, self.backbone.size_divisibility)
            return images, images_view1, images_view2
        else:
            return images
        
    def _inference_for_ms_test(self, batched_inputs):
        """
        function used for multiscale test, will be refactor in the future.
        The same input with `forward` function.
        """
        assert not self.training, "inference mode with training=True"
        assert len(batched_inputs) == 1, "inference image number > 1"
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        anchors = self.anchor_generator(features)

        box_cls, box_delta = self.head(features, training=self.training)
        
        results = self.inference(box_cls, box_delta, anchors, images)
        for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results = detector_postprocess(results_per_image, height, width)
        return processed_results


class RetinaNetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec], training):
        super().__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors = cfg.build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        self.cls_score = nn.Conv2d(in_channels,
                                   num_anchors * num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   num_anchors * 4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
         
        if training:
            self.cls_score1 = nn.Conv2d(in_channels,
                                       num_anchors * num_classes,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
            self.bbox_pred1 = nn.Conv2d(in_channels,
                                       num_anchors * 4,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
            self.cls_score2 = nn.Conv2d(in_channels,
                                       num_anchors * num_classes,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
            self.bbox_pred2 = nn.Conv2d(in_channels,
                                       num_anchors * 4,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

            # Initialization
            for modules in [
                    self.cls_subnet, self.bbox_subnet, 
                    self.cls_score, self.bbox_pred, 
                    self.cls_score1, self.bbox_pred1, 
                    self.cls_score2, self.bbox_pred2
            ]:
                for layer in modules.modules():
                    if isinstance(layer, nn.Conv2d):
                        torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                        torch.nn.init.constant_(layer.bias, 0)
            
            # Use prior in model initialization to improve stability
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.cls_score.bias, bias_value)
            torch.nn.init.constant_(self.cls_score1.bias, bias_value)
            torch.nn.init.constant_(self.cls_score2.bias, bias_value)
            
    def forward(self, features, features_view1=None, features_view2=None, training=True):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        if training:
            logits0 = []
            bbox_reg0 = []
            logits1 = []
            bbox_reg1 = []
            logits2 = []
            bbox_reg2 = []
            for feature, features_view1, features_view2 in zip(features, features_view1, features_view2):
                cls_feat = self.cls_subnet(feature)
                cls_feat_view1 = self.cls_subnet(features_view1)
                cls_feat_view2 = self.cls_subnet(features_view2)
                logits0.append(self.cls_score(cls_feat))
                logits1.append(self.cls_score1(cls_feat_view1))
                logits2.append(self.cls_score2(cls_feat_view2))

                reg_feat = self.bbox_subnet(feature)
                reg_feat_view1 = self.bbox_subnet(features_view1)
                reg_feat_view2 = self.bbox_subnet(features_view2)
                bbox_reg0.append(self.bbox_pred(reg_feat))
                bbox_reg1.append(self.bbox_pred1(reg_feat_view1))
                bbox_reg2.append(self.bbox_pred2(reg_feat_view2))

            return logits0, bbox_reg0, logits1, bbox_reg1, logits2, bbox_reg2
        else:
            logits = []
            bbox_reg = []
            for feature in features:
                cls_feat = self.cls_subnet(feature)
                logits.append(self.cls_score(cls_feat))
                reg_feat = self.bbox_subnet(feature)
                bbox_reg.append(self.bbox_pred(reg_feat))

            return logits, bbox_reg
        
