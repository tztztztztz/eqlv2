root= '/mnt/lustrenew/zhanggang/data_t1/mmdetection/'
cfg_root = root + 'configs/lvis/'
data_root = root + 'data/lvis'

_base_ = [cfg_root + 'mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        dataset=dict(
            data_root=data_root,
            ann_file='annotations/lvis_v1_train.json',
            img_prefix='',
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

#lr_config = dict(step=[16, 22])

#total_epochs = 24

#resume_from='XXX'
