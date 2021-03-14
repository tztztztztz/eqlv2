_base_ = ['../end2end/mask_rcnn_r50_8x2_1x.py']

data = dict(train=dict(type="CASDataset", oversample_thr=None, max_iter=75000 * 16))

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
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[50000, 70000],
    by_epoch=False,
)

total_epochs = 1

evaluation = dict(interval=1)

load_from = "r50_1x/model_reset_remove.pth"

work_dir = 'crt_1x'
