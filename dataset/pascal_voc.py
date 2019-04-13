import os
import pickle
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset


class _VOCDataset_Single(Dataset):
    """
    This is the class for single image set, e.g. "train_2012".
    """
    class_names = ('aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, path, image_set, transforms=None, fetch_img=True, use_difficult=True):
        '''
        Arguments:
        ----------
        path:       root path of VOC folder
        image_set:  single image_set, e.g.: "train_2012"
        for_map:    img file won't be fetched for speed, during mean average percision calculation
        use_difficult: whether to use difficult box when fetch data
        '''
        super(_VOCDataset_Single, self).__init__()
        self.data_dir_path = path
        self.image_set, self.year = image_set.split("_")
        self.year = int(self.year)
        self.transforms = transforms
        self.use_difficult = use_difficult
        self.fetch_img = fetch_img

        self.createIndex()

    def createIndex(self):
        img_list_file_path = os.path.join(self.data_dir_path, 'VOCdevkit', 'VOC{}'.format(self.year), 'ImageSets',
                                          'Main', "{}.txt".format(self.image_set))
        with open(img_list_file_path) as f:
            self.img_list = [s.strip() for s in f.readlines()]

        self.label_to_cls_name = {i: _VOCDataset_Single.class_names[i] for i in range(len(_VOCDataset_Single.class_names))}
        self.cls_name_to_label = {name: label for label, name in self.label_to_cls_name.items()}

        # parse xml and cache it to self.data_dir_path/Annotation_cache
        cache_fname = os.path.join(self.data_dir_path, 'Annotation_cache', "{}{}.pkl".format(self.image_set, self.year))
        if os.path.exists(cache_fname):
            print("Load Pascal VOC {}{} annotation from cache file: {}".format(self.image_set, self.year, cache_fname))
            with open(cache_fname, 'rb') as f:
                self.imgToAnns = pickle.load(f)
        else:
            print("Parsing Pascal VOC {}{} annotation".format(self.image_set, self.year))
            self.imgToAnns = {img_id: self.parse_xml(img_id) for img_id in self.img_list}
            if not os.path.exists(os.path.join(self.data_dir_path, 'Annotation_cache')):
                os.mkdir(os.path.join(self.data_dir_path, 'Annotation_cache'))
            with open(cache_fname, 'wb') as f:
                pickle.dump(self.imgToAnns, f, pickle.HIGHEST_PROTOCOL)
            print("Parsing succeed! Cache to file for further reuse: {}".format(cache_fname))

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError("Index is out of range!")

        img_id = self.img_list[idx]
        img_name = os.path.join(self.data_dir_path, 'VOCdevkit', 'VOC{}'.format(self.year), 'JPEGImages', "{}.jpg".format(img_id))

        boxes, labels, difficult = [], [], []
        for ann_dict in self.imgToAnns[img_id]:
            boxes.append(ann_dict['boxes'])
            labels.append(ann_dict['labels'])
            difficult.append(ann_dict['difficult'])
        boxes = np.array(boxes, dtype=np.float)
        labels = np.array(labels, dtype=np.int)
        difficult = np.array(difficult, dtype=np.bool)

        if self.fetch_img:
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            origin_size = np.array([img.shape[1], img.shape[0]], dtype=np.int)  # (W, H)
        else:
            img = None
            origin_size = None

        if self.transforms is not None:
            img, boxes = self.transforms(img, boxes)

        sample = {'img': img,
                  'boxes': boxes,
                  'labels': labels,
                  'difficult': difficult,
                  'origin_size': origin_size,
                  'img_name': img_name}
        return sample

    def __len__(self):
        return len(self.img_list)

    def collate_fn(self, batch):
        # size may vary if no Resize transform
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

    def parse_xml(self, img_id):
        xml_path = os.path.join(self.data_dir_path, 'VOCdevkit', 'VOC{}'.format(self.year), 'Annotations',
                                '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        objs = tree.findall('object')
        ann = []
        for obj in objs:
            box = obj.find('bndbox')
            x1 = int(box.find('xmin').text) - 1
            y1 = int(box.find('ymin').text) - 1
            x2 = int(box.find('xmax').text) - 1
            y2 = int(box.find('ymax').text) - 1
            d = obj.find('difficult')
            difficult = 0 if d is None else int(d.text)
            label = self.cls_name_to_label[obj.find('name').text.lower().strip()]

            ann.append({'boxes': [x1, y1, x2, y2],
                        'labels': label,
                        'difficult': difficult})

        return ann

    @property
    def num_class(self):
        return len(self.label_to_cls_name)


class VOCDataset(ConcatDataset):
    """
    This is a wrapper for single or multiple VOC datasets.
    """
    class_names = _VOCDataset_Single.class_names

    def __init__(self, path, mix_set, transforms=None, fetch_img=True, use_difficult=True):
        self.data_dir_path = path
        self._base_datasets = []
        if isinstance(mix_set, str):
            mix_set = [mix_set]
        for dataset in mix_set:
            self._base_datasets.append(_VOCDataset_Single(path, dataset, transforms, fetch_img, use_difficult))
        super(VOCDataset, self).__init__(self._base_datasets)

        self.transforms = transforms

        self.img_list = [img_id for d in self._base_datasets for img_id in d.img_list]
        self.label_to_cls_name = self._base_datasets[0].label_to_cls_name
        self.cls_name_to_label = self._base_datasets[0].cls_name_to_label

    def collate_fn(self, batch):
        # size may vary if no Resize transform
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

    def reset_resize_shape(self, size):
        for t in self.transforms.modules():
            if t=='Resize':
                t.reset_size(size)

    @property
    def num_class(self):
        return len(self.class_names)

    @property
    def resize_shape(self):
        for t in self.transforms.modules():
            if t=='Resize':
                return t.size


