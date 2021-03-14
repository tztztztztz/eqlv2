_base_ = ['./faster_rcnn_r50_fpn_8x2_2x.py']

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="EQLv2"))))

work_dir = 'openimage_eqlv2_2x'