# This is a COCO dataset class, code modify from "cocoapi"
# Based on:
# --------------------------------------------------------
# cocodataset/cocoapi AT github.com:
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
# License please refer to:
# https://github.com/cocodataset/cocoapi/blob/master/license.txt
# --------------------------------------------------------


import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import utils.transforms.transforms


class CocoDataset(Dataset):
    def __init__(self, path, image_set, year, img_transforms=None, box_transforms=None):
        self.data_dir_path = path
        if image_set not in ['train', 'val', 'test']:
            raise NameError("image_set name is incorrect: got '{}'".format(image_set))
        self.image_set = image_set
        self.year = year
        self.img_transforms = img_transforms
        self.box_transforms = box_transforms
        if not isinstance(self.img_transforms, utils.transforms.transforms.Compose):
            self.img_transforms = utils.transforms.transforms.Compose(self.img_transforms)

        if image_set != 'test':
            annotation_filename = os.path.join(self.data_dir_path, 'annotations',
                                               'instances_{}{}.json'.format(image_set, year))
            self.dataset = json.load(open(annotation_filename))
            self.createIndex()
        else:
            # note in train/va, imgs_list refers to img_id
            self.imgs_list = os.listdir(os.path.join(self.data_dir_path, 'test{}'.format(year)))

    def createIndex(self):
        imgs_list = set()
        cats, imgs = {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgs_list.add(ann['image_id'])
                imgToAnns[ann['image_id']].append(ann)

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        self.imgs_list = list(imgs_list)
        self.imgToAnns = imgToAnns
        self.cats = cats
        self.imgs = imgs

        # cat_id is of COCO annotation meaning, label is 0,1,2.. encoding
        self.cat_id_to_label = {cat_id: label for label, cat_id in enumerate(cats.keys())}
        self.label_to_cat_id = {label: cat_id for cat_id, label in self.cat_id_to_label.items()}

        self.label_to_cls_name = {label: self._label_to_cat_name(label) for label in self.label_to_cat_id.keys()}
        self.cls_name_to_label = {name: label for label, name in self.label_to_cls_name.items()}

    def __getitem__(self, idx):
        '''
        :return:
           'img': np image, or in tensor form if self.img_transforms
           'boxes': np box, dtype=np.float32, default in TOP_BOTTOM form
           'labels': list of class labels
           'origin_size': np array of [width, height], dtype=np.int32
           'img_name': path of img_name
        '''
        if idx > len(self):
            raise IndexError("Index is out of range!")

        if self.image_set != 'test':
            img_id = self.imgs_list[idx]
            img_name = self.imgs[img_id]['file_name']
            img_name = os.path.join(self.data_dir_path, '{}{}'.format(self.image_set, self.year), img_name)

            boxes, labels = [], []
            for ann_dict in self.imgToAnns[img_id]:
                boxes.append(ann_dict['bbox'])
                labels.append(self.cat_id_to_label[ann_dict['category_id']])
            boxes = np.array(boxes, dtype=np.float)
            boxes[:, 2:] += boxes[:, :2]
        else:
            img_name = os.path.join(self.data_dir_path, 'test{}'.format(self.year), self.imgs_list[idx])
            boxes = None
            labels = None

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origin_size = np.array([img.shape[1], img.shape[0]], dtype=np.int32)

        if self.img_transforms is not None:
            img = self.img_transforms(img)
            # Check rescale
            rescale_w, rescale_h = None, None
            for transform in self.img_transforms:
                if transform == 'Resize':
                    rescale_w, rescale_h = transform.size
            if rescale_w is not None:
                boxes /= (origin_size[0], origin_size[1], origin_size[0], origin_size[1])
                boxes *= (rescale_w, rescale_h, rescale_w, rescale_h)

        if self.box_transforms is not None:
            boxes = self.box_transforms(boxes)

        sample = {'img': img,
                  'boxes': boxes,
                  'labels': labels,
                  'origin_size': origin_size,
                  'img_name': img_name}
        return sample

    def collate_fn(self, batch):
        # size may vary if no Resize transform
        if self.img_transforms is None or 'Resize' not in self.img_transforms:
            img = [x['img'] for x in batch]
        elif type(batch[0]['img']) == torch.Tensor:
            img = torch.stack([x['img'] for x in batch])
        else:
            img = np.stack([x['img'] for x in batch])
        samples = {'img': img,
                   'boxes': [x['boxes'] for x in batch],
                   'labels': [x['labels'] for x in batch],
                   'origin_size': np.stack([x['origin_size'] for x in batch]),
                   'img_name': [x['img_name'] for x in batch]}
        return samples

    def __len__(self):
        return len(self.imgs_list)

    def draw_bbox(self, idx, save_path=None):
        img_info = self[idx]
        img = img_info['img']
        for x, y, w, h in img_info['boxes']:
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
        cv2.imshow("", img);
        cv2.waitKey(0);
        cv2.destroyAllWindows()

    def _label_to_cat_name(self, label):
        cat_id = self.label_to_cat_id[label]
        return self.cats[cat_id]['name']

    @property
    def num_class(self):
        return len(self.label_to_cls_name)
