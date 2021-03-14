_base_ = ['./faster_rcnn_r50_fpn_8x2_2x.py']

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="EQL", use_sigmoid=True, lambda_=1652, version="openimage"))))

work_dir = 'openimage_eql_2x'
