import logging

from cvpods.layers import ShapeSpec
from retinanet import RetinaNet
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_retinanet_resnet_fpn_backbone
from cvpods.modeling.anchor_generator import DefaultAnchorGenerator


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_retinanet_resnet_fpn_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_anchor_generator(cfg, input_shape):

    return DefaultAnchorGenerator(cfg, input_shape)


def build_model(cfg, training=True):

    cfg.build_backbone = build_backbone
    cfg.build_anchor_generator = build_anchor_generator

    model = RetinaNet(cfg, training)
    total = sum([para.nelement() for para in model.parameters()])
    print('Model Parameters:', total/1e6)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
