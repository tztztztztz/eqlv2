_base_ = ['./mask_rcnn_r50_8x2_1x.py']

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="EQL", use_sigmoid=True, lambda_=0.0011, version="v1"))))

work_dir = 'eql_1x'
