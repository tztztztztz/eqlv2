import torch
import numpy as np
from mmdet.utils import get_root_logger
from .builder import DATASETS


class RandomDataStream:
    def __init__(self, data, generator, shuffle=True, dtype=torch.int32):
        self._size = len(data)
        self.data = torch.Tensor(data).to(dtype=dtype)
        self._shuffle = shuffle
        self.g = generator

    def __iter__(self):
        yield from self._infinite_indices()

    def _infinite_indices(self):
        while True:
            if self._shuffle:
                randperm = torch.randperm(self._size, generator=self.g)
                yield from self.data[randperm]
            else:
                yield self.data


@DATASETS.register_module()
class CASDataset(object):
    def __init__(self, dataset, max_iter):
        self.dataset = dataset
        self.max_iter = max_iter
        self.num_classes = len(dataset.cat_ids)
        self.CLASSES = dataset.CLASSES

        logger = get_root_logger()
        logger.info(f'init CAS dataset, num_classes {self.num_classes}')

        indices = []
        flag = []

        cls_data_inds = [[] for _ in range(self.num_classes)]
        for idx in range(len(dataset)):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            for cat_id in cat_ids:
                label = self.dataset.cat2label[cat_id]
                cls_data_inds[label].append(idx)

        g = torch.Generator()
        g.manual_seed(0)
        cls_ind_stream = iter(RandomDataStream(list(range(self.num_classes)), g))
        cls_data_streams = [None] * self.num_classes
        for i, data_inds in enumerate(cls_data_inds):
            cls_data_streams[i] = iter(RandomDataStream(data_inds, g))

        for _ in range(max_iter):
            cls_idx = next(cls_ind_stream)
            img_idx = next(cls_data_streams[cls_idx])
            indices.append(int(img_idx))
            flag.append(self.dataset.flag[img_idx])

        self.indices = indices
        self.flag = np.asarray(flag, dtype=np.uint8)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ori_index = self.indices[idx]
        return self.dataset[ori_index]
