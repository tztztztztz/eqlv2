_base_ = ['../end2end/mask_rcnn_r50_8x2_1x.py']

data = dict(train=dict(oversample_thr=0.0))

train_cfg = dict(
    log_cfg=dict(
        print_param=True
    ),
    freeze=dict(
        neck=dict(type="all"),
        rpn=dict(type="all"),
        roi_head=dict(
            bbox_head=dict(type='feat'),
            mask_head=dict(type='all')
        )
    )
)

model = dict(
    backbone=dict(frozen_stages=4),
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(type='GroupSoftmax', version="v1"))
    ),
)

load_from = "r50_1x/model_reset_remove.pth"
work_dir = 'bags_1x'
