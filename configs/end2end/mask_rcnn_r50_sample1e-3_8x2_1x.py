cfg_root = '../lvis/'
data_root = 'data/lvis'

_base_ = [cfg_root + 'mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        dataset=dict(
            data_root=data_root,
            ann_file='annotations/lvis_v1_train.json',
            img_prefix='',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        data_root=data_root,
        ann_file='annotations/lvis_v1_val.json',
        img_prefix='',
    ),
    test=dict(
        data_root=data_root,
        ann_file='annotations/lvis_v1_val.json',
        img_prefix='',
    )
)

evaluation = dict(interval=12, metric=['bbox', 'segm'])

test_cfg = dict(rcnn=dict(perclass_nms=True))

work_dir = 'r50_rfs_1x'
