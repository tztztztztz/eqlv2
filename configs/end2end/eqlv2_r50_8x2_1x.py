_base_ = ['./mask_rcnn_r50_8x2_1x.py']

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="EQLv2"))))

work_dir = 'eqlv2_1x'
