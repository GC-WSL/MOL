import os.path as osp

from cvpods.configs.retinanet_config import RetinaNetConfig
from coco import NWPU  # noqa


_config_dict = dict(
    MODEL=dict(
        WEIGHTS="./R-50.pkl",
        RESNETS=dict(DEPTH=50),
        RETINANET=dict(
            NUM_CLASSES=10,
            IOU_THRESHOLDS=[0.4, 0.5],
            IOU_LABELS=[0, -1, 1],
            NMS_THRESH_TEST=0.5,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            SMOOTH_L1_LOSS_BETA=0.1,
            PSEUDO_SCORE_THRES=0.6,
            SCORE_THRESH_TEST=0.001, 
        ),
    
    ),
    DATASETS=dict(
        TRAIN=("coco_NWPU_train",),
        TEST=("coco_NWPU_test",),
    ),

    SOLVER=dict(
        LR_SCHEDULER=dict(
            MAX_ITER=5000,
            STEPS=(3750,),
        ),
        OPTIMIZER=dict(
            BASE_LR=0.01,
        ),
        IMS_PER_BATCH=16,
        IMS_PER_DEVICE=2,
    ),

    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(400, 600, 800, 1000, 1200),
                    max_size=2000, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    TEST=dict(
        AUG=dict(
            ENABLED=False,
            MIN_SIZES=(400, 600, 800, 1000, 1200),
            MAX_SIZE=2000,
            FLIP=True,
            EXTRA_SIZES=(),
            SCALE_FILTER=False,
            SCALE_RANGES=(),
        ),
        
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class CustomRetinaNetConfig(RetinaNetConfig):
    def __init__(self):
        super(CustomRetinaNetConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomRetinaNetConfig()
