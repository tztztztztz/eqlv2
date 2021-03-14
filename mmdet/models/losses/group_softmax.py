import bisect
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from mmdet.utils import get_root_logger


def get_instance_count(version="v1"):
    if version == "v0_5":
        from mmdet.utils.lvis_v0_5_categories import get_instance_count
        return get_instance_count()
    elif version == "v1":
        from mmdet.utils.lvis_v1_0_categories import get_instance_count
        return get_instance_count()
    else:
        raise KeyError(f"version {version} is not supported")


@LOSSES.register_module()
class GroupSoftmax(nn.Module):
    """
    This uses a different encoding from v1.
    v1: [cls1, cls2, ..., other1_for_group0, other_for_group_1, bg, bg_others]
    this: [group0_others, group0_cls0, ..., group1_others, group1_cls0, ...]
    """
    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 beta=8,
                 bin_split=(10, 100, 1000),
                 version="v1"):
        super(GroupSoftmax, self).__init__()
        self.use_sigmoid = False
        self.group = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.beta = beta
        self.bin_split = bin_split
        self.version = version
        self.cat_instance_count = get_instance_count(version=self.version)
        self.num_classes = len(self.cat_instance_count)
        assert not self.use_sigmoid
        self._assign_group()
        self._prepare_for_label_remapping()
        logger = get_root_logger()
        logger.info(f"set up {self.__class__.__name__}, version {self.version}, beta: {self.beta}, num classes {self.num_classes}")

    def _assign_group(self):
        self.num_group = (len(self.bin_split) + 1) + 1  # add a group for background
        self.group_cls_ids = [[] for _ in range(self.num_group)]
        group_ids = list(map(lambda x: bisect.bisect_right(self.bin_split, x), self.cat_instance_count))
        self.group_ids = group_ids + [self.num_group - 1]
        for cls_id, group_id in enumerate(self.group_ids):
            self.group_cls_ids[group_id].append(cls_id)
        logger = get_root_logger()
        logger.info(f"group number {self.num_group}")
        self.n_cls_group = list(map(lambda x: len(x), self.group_cls_ids))
        logger.info(f"number of classes in group: {self.n_cls_group}")

    def _get_group_pred(self, cls_score, apply_activation_func=False):
        group_pred = []
        start = 0
        for group_id, n_cls in enumerate(self.n_cls_group):
            num_logits = n_cls + 1  # + 1 for "others"
            pred = cls_score.narrow(1, start, num_logits)
            start = start + num_logits
            if apply_activation_func:
                pred = F.softmax(pred, dim=1)
            group_pred.append(pred)
        assert start == self.num_classes + 1 + self.num_group
        return group_pred

    def _prepare_for_label_remapping(self):
        group_label_maps = []
        for group_id, n_cls in enumerate(self.n_cls_group):
            label_map = [0 for _ in range(self.num_classes + 1)]
            group_label_maps.append(label_map)
        # init value is 1 because 0 is set for "others"
        _tmp_group_num = [1 for _ in range(self.num_group)]
        for cls_id, group_id in enumerate(self.group_ids):
            g_p = _tmp_group_num[group_id]  # position in group
            group_label_maps[group_id][cls_id] = g_p
            _tmp_group_num[group_id] += 1
        self.group_label_maps = torch.LongTensor(group_label_maps)

    def _remap_labels(self, labels):
        new_labels = []
        new_weights = []  # use this for sampling others
        new_avg = []
        for group_id in range(self.num_group):
            mapping = self.group_label_maps[group_id]
            new_bin_label = mapping[labels]
            new_bin_label = torch.LongTensor(new_bin_label).to(labels.device)
            if self.is_background_group(group_id):
                weight = torch.ones_like(new_bin_label)
            else:
                weight = self._sample_others(new_bin_label)
            new_labels.append(new_bin_label)
            new_weights.append(weight)

            avg_factor = max(torch.sum(weight).float().item(), 1.)
            new_avg.append(avg_factor)
        return new_labels, new_weights, new_avg

    def _sample_others(self, label):

        # only works for non bg-fg bins

        fg = torch.where(label > 0, torch.ones_like(label),
                         torch.zeros_like(label))
        fg_idx = fg.nonzero(as_tuple=True)[0]
        fg_num = fg_idx.shape[0]
        if fg_num == 0:
            return torch.zeros_like(label)

        bg = 1 - fg
        bg_idx = bg.nonzero(as_tuple=True)[0]
        bg_num = bg_idx.shape[0]

        bg_sample_num = int(fg_num * self.beta)

        if bg_sample_num >= bg_num:
            weight = torch.ones_like(label)
        else:
            sample_idx = np.random.choice(bg_idx.cpu().numpy(),
                                          (bg_sample_num, ), replace=False)
            sample_idx = torch.from_numpy(sample_idx)
            fg[sample_idx] = 1
            weight = fg

        return weight.to(label.device)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        group_preds = self._get_group_pred(cls_score, apply_activation_func=False)
        new_labels, new_weights, new_avg = self._remap_labels(label)

        cls_loss = []
        for group_id in range(self.num_group):
            pred_in_group = group_preds[group_id]
            label_in_group = new_labels[group_id]
            weight_in_group = new_weights[group_id]
            avg_in_group = new_avg[group_id]
            loss_in_group = F.cross_entropy(pred_in_group,
                                            label_in_group,
                                            reduction='none',
                                            ignore_index=-1)
            loss_in_group = torch.sum(loss_in_group * weight_in_group)
            loss_in_group /= avg_in_group
            cls_loss.append(loss_in_group)
        cls_loss = sum(cls_loss)
        return cls_loss * self.loss_weight

    def get_activation(self, cls_score):
        n_i, n_c = cls_score.size()
        group_activation = self._get_group_pred(cls_score, apply_activation_func=True)
        bg_score = group_activation[-1]
        activation = cls_score.new_zeros((n_i, len(self.group_ids)))
        for group_id, cls_ids in enumerate(self.group_cls_ids[:-1]):
            activation[:, cls_ids] = group_activation[group_id][:, 1:]
        activation *= bg_score[:, [0]]
        activation[:, -1] = bg_score[:, 1]

        return activation

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1 + self.num_group
        return num_channel

    def is_background_group(self, group_id):
        return group_id == self.num_group - 1
