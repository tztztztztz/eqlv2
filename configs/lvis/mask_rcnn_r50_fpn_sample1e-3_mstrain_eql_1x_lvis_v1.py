_base_ = ['mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py']


model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(type="EQL", use_sigmoid=True, lambda_=0.0011, version="v1"))))
