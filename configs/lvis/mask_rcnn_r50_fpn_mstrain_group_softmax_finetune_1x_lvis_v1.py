_base_ = ['./mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py']

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
