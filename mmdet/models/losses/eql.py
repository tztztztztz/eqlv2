import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def get_image_count_frequency(version="v0_5"):
    if version == "v0_5":
        from mmdet.utils.lvis_v0_5_categories import get_image_count_frequency
        return get_image_count_frequency()
    elif version == "v1":
        from mmdet.utils.lvis_v1_0_categories import get_image_count_frequency
        return get_image_count_frequency()
    elif version == "openimage":
        from mmdet.utils.openimage_categories import get_instance_count
        return get_instance_count()
    else:
        raise KeyError(f"version {version} is not supported")


@LOSSES.register_module()
class EQL(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 lambda_=0.00177,
                 version="v0_5"):
        super(EQL, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.lambda_ = lambda_
        self.version = version
        self.freq_info = torch.FloatTensor(get_image_count_frequency(version))

        num_class_included = torch.sum(self.freq_info < self.lambda_)
        print(f"set up EQL (version {version}), {num_class_included} classes included.")

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        target = expand_label(cls_score, label)

        eql_w = 1 - self.exclude_func() * self.threshold_func() * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')

        cls_loss = torch.sum(cls_loss * eql_w) / self.n_i

        return self.loss_weight * cls_loss

    def exclude_func(self):
        # instance-level weight
        bg_ind = self.n_c
        weight = (self.gt_classes != bg_ind).float()
        weight = weight.view(self.n_i, 1).expand(self.n_i, self.n_c)
        return weight

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight