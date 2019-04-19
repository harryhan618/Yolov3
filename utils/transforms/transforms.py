import cv2

import numpy as np
import torch
from torchvision.transforms import Normalize as Normalize_th

__all__ = ['CustomTransform', 'Compose', 'Resize', 'RandomResize', 'Normalize', 'To_torch_channel', 'ToTensor',
           'Box_TopWH_To_CenterWH', 'Box_TopBottom_To_CenterWH', 'Box_CenterWH_To_TopBottom',]

class CustomTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, name):
        return str(self) == name

    def __iter__(self):
        def iter_fn():
            for t in [self]:
                yield t
        return iter_fn()

    def __contains__(self, name):
        for t in self.__iter__():
            if isinstance(t, Compose):
                if name in t:
                    return True
            elif name == t:
                return True
        return False


class Compose(CustomTransform):
    """
    All transform in Compose should be able to accept two non None variable, img and boxes
    """
    def __init__(self, *transforms):
        self.transforms = [*transforms]

    def __call__(self, img=None, boxes=None):
        for t in self.transforms:
            img, boxes = t(img=img, boxes=boxes)
        return img, boxes

    def __iter__(self):
        return iter(self.transforms)

    def modules(self):
        yield self
        for t in self.transforms:
            if isinstance(t, Compose):
                for _t in t.modules():
                    yield _t
            else:
                yield t


class Resize(CustomTransform):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        if isinstance(size, int):
            size = (size, size)
        self.size = size #(W, H)
        self.interpolation = interpolation

    def __call__(self, img, boxes=None):
        origin_h, origin_w = img.shape[:2]
        rescale_w, rescale_h = self.size
        img = cv2.resize(img, self.size, interpolation=self.interpolation)

        if boxes is not None:
            boxes = boxes.astype('float')
            boxes /= np.array([[origin_w, origin_h, origin_w, origin_h]])
            boxes *= np.array([[rescale_w, rescale_h, rescale_w, rescale_h]])
        return img, boxes

    def reset_size(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size


class RandomResize(Resize):
    """
    Randomly resize to size: `stride` * m, m randomly sample from `multiples`
    """
    def __init__(self, stride, multiples, interpolation=cv2.INTER_LINEAR):
        size = (stride, stride) if isinstance(stride, int) else stride
        size = tuple((s * multiples[0] for s in size))
        super(RandomResize, self).__init__(size, interpolation)
        self.stride = stride #(W, H)
        self.multiples = multiples

    def __call__(self, img, boxes=None):
        m = np.random.choice(self.multiples)
        self.reset_size(tuple((s * m for s in self.stride)))

        origin_w, origin_h = img.shape[:2]
        rescale_w, rescale_h = self.size
        img = cv2.resize(img, self.size, interpolation=self.interpolation)

        if boxes is not None:
            boxes = boxes.astype('float')
            boxes /= (origin_h, origin_w, origin_h, origin_w)
            boxes *= (rescale_w, rescale_h, rescale_w, rescale_h)
        return img, boxes

    def __eq__(self, name):
        return str(self) == name or name == 'Resize'


class Normalize(CustomTransform):
    def __init__(self, mean, std):
        self.transform = Normalize_th(mean, std)

    def __call__(self, img, boxes=None):
        img = self.transform(img)
        return img, boxes


class To_torch_channel(CustomTransform):
    def __call__(self, img, boxes=None):
        if isinstance(img, torch.Tensor):
            img = img.permute(2, 0, 1)
        elif isinstance(img, np.ndarray):
            img = img.transpose(2, 0, 1)

        return img, boxes


class ToTensor(CustomTransform):
    def __init__(self, dtype=torch.float):
        self.dtype=dtype

    def __call__(self, img, boxes=None):
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(self.dtype) / 255.

        return img, boxes


class Box_TopWH_To_CenterWH(CustomTransform):
    def __call__(self, boxes, img=None):
        if isinstance(boxes, np.ndarray):
            box_center = np.zeros_like(boxes)
        elif isinstance(boxes, torch.Tensor):
            box_center = torch.zeros_like(boxes)
        else:
            raise TypeError("Input parameter 'boxes' should be numpy.ndarray or torch.Tensor, got {}".format(type(boxes)))
        box_center[:, :2] = boxes[:, :2] + boxes[:, 2:] / 2

        if img is not None:
            return img, box_center
        else:
            return box_center


class Box_TopBottom_To_CenterWH(CustomTransform):
    def __call__(self, boxes, img=None):
        if isinstance(boxes, np.ndarray):
            box_center = np.zeros_like(boxes)
        elif isinstance(boxes, torch.Tensor):
            box_center = torch.zeros_like(boxes)
        else:
            raise TypeError("Input parameter 'boxes' should be numpy.ndarray or torch.Tensor, got {}".format(type(boxes)))
        box_center[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        box_center[:, 2:] = (boxes[:, 2:] - boxes[:, :2])

        if img is not None:
            return img, box_center
        else:
            return box_center


class Box_CenterWH_To_TopBottom(CustomTransform):
    def __call__(self, boxes, img=None):
        if isinstance(boxes, np.ndarray):
            boxes_TB = np.zeros_like(boxes)
        elif isinstance(boxes, torch.Tensor):
            boxes_TB = torch.zeros_like(boxes)
        else:
            raise TypeError("Input parameter 'boxes' should be numpy.ndarray or torch.Tensor, got {}".format(type(boxes)))
        boxes_TB[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
        boxes_TB[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2

        if img is not None:
            return img, boxes_TB
        else:
            return boxes_TB
