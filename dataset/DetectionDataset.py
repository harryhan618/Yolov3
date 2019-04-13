import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DetectionDataset(Dataset):
    classes = ()

    def __init__(self, transforms=None):
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Return a dict in following form:
          dummy_sample = {'img':    <np.ndarray, (H, W, 3), np.uint8>
                                    or <torch.tensor, (3, H, W)> if transformed to tensor,
                          'boxes':       numpy array (N, 4),  type numpy.float, [x1, y1, x2, y2]
                          'labels':      numpy array (N, ),   type numpy.int
                          'difficult':   numpy array (N, ),   type numpy.uint8
                          'origin_size': numpy array([W, H]), type numpy.int
                          'img_name':    str, full path to img}
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collate_fn(self, batch):
        if self.transforms is None or 'Resize' not in self.transforms:
            img = [x['img'] for x in batch]
        elif isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([x['img'] for x in batch])
        else:
            img = np.stack([x['img'] for x in batch])
        samples = {'img': img,
                   'boxes': [x['boxes'] for x in batch],
                   'labels': [x['labels'] for x in batch],
                   'difficult': [x['difficult'] for x in batch],
                   'origin_size': np.stack([x['origin_size'] for x in batch]),
                   'img_name': [x['img_name'] for x in batch]}
        return samples

    @property
    def cls_name_to_label(self):
        raise NotImplementedError

    @property
    def label_to_cls_name(self):
        raise NotImplementedError

    # self.img_list
    # self.imgToAnns


class BaseDataLoader(DataLoader):
    pass

# TODO: 先把C的部分查看性能
# TODO: 看看要不要用C还是C++
# TODO: 然后就能确定dataset中bbox的numpy dtype应该是什么
# TODO: 写这个basedataset
