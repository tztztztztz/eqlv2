_base_ = ['./mask_rcnn_r50_sample1e-3_8x2_1x.py']

data = dict(train=dict(oversample_thr=0.0))

work_dir = 'r50_1x'
