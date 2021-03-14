_base_ = ['../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py']

data_root = 'data/openimage/'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=500)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(samples_per_gpu=2, workers_per_gpu=1,
            train=dict(_delete_=True,
                       type="MaxIterationDataset",
                       max_iter=180000 * 16,
                       dataset=dict(type="OpenimageDataset",
                                    ann_file=data_root + 'annotations/openimages_challenge_2019_train_bbox.json',
                                    img_prefix=data_root + 'images/train/',
                                    pipeline=train_pipeline)),
            val=dict(type="OpenimageDataset", ann_file=data_root + 'annotations/openimages_challenge_2019_val_bbox.json', img_prefix=data_root + 'images/validation/'),
            test=dict(type="OpenimageDataset", ann_file=data_root + 'annotations/openimages_challenge_2019_val_bbox.json', img_prefix=data_root + 'images/validation/'))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[120000, 160000],
    by_epoch=False,
)

log_config = dict(interval=10)

total_epochs = 1

evaluation = dict(interval=2)  # disable evaluation

test_cfg = dict(
    rcnn=dict(
        score_thr=0.0001,
        # LVIS allows up to 300
        max_per_img=300))

work_dir = 'openimage_2x'