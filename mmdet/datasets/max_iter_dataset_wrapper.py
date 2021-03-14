import torch
import numpy as np
from mmdet.utils import get_root_logger
from .builder import DATASETS
from .class_balance_dataset_wrapper import RandomDataStream


@DATASETS.register_module()
class MaxIterationDataset(object):
    def __init__(self, dataset, max_iter):
        self.dataset = dataset
        self.max_iter = max_iter
        self.num_classes = len(dataset.cat_ids)
        self.CLASSES = dataset.CLASSES

        logger = get_root_logger()
        logger.info(f'init max_iteration dataset, num_classes {self.num_classes}')

        indices = []
        flag = []

        g = torch.Generator()
        g.manual_seed(0)
        img_ids = iter(RandomDataStream(list(range(len(dataset))), g))
        for _ in range(max_iter):
            img_idx = next(img_ids)
            indices.append(int(img_idx))
            flag.append(self.dataset.flag[img_idx])

        self.indices = indices
        self.flag = np.asarray(flag, dtype=np.uint8)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ori_index = self.indices[idx]
        return self.dataset[ori_index]