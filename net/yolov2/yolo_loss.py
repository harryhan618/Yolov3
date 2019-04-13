import numpy as np
import torch
import torch.nn as nn

from utils.iou import *
from utils.postprocess import yolo_to_box
from utils.transforms import Box_TopBottom_To_CenterWH

"""
本loss除以batch_size
"""

class YOLOLoss(nn.Module):
    box_transform = Box_TopBottom_To_CenterWH()

    def __init__(self, num_classes, img_size):
        '''
        calculate loss for one feature scale of yolo_output

        Arguments:
            img_size: size of CNN input, (w, h)
        '''
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size  # in form (width, height)
        # self.pool = mp.Pool(10)

        self.scale_obj = 5.0
        self.scale_no_obj = 0.5
        self.scale_box = 1.0
        self.scale_class = 1.0

        self.iou_threshold = 0.4

        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.CEloss = nn.CrossEntropyLoss(reduction='sum')
        self.log_loss = nn.NLLLoss(reduction='sum')
        self.loss_dict = {
            'loss_box_xy': 0,
            'loss_box_wh': 0,
            'loss_iou': 0,
            'loss_class': 0,
            'loss': 0
        }

    def forward(self, yolo_output, boxes, labels, anchor):
        device = yolo_output.device

        target_box, target_iou, target_onehot, obj_mask, no_obj_mask = self.build_target(yolo_output, boxes, labels, anchor)

        target_box = torch.from_numpy(target_box).to(device)
        target_iou = torch.from_numpy(target_iou).to(device)
        target_onehot = torch.from_numpy(target_onehot).to(device)
        obj_mask = torch.from_numpy(obj_mask).to(device)
        no_obj_mask = torch.from_numpy(no_obj_mask).to(device)

        nb = float(yolo_output.shape[0])
        n1 = float(obj_mask.type(torch.int).sum().item())
        n2 = float(no_obj_mask.type(torch.int).sum().item())

        # ******************** DEBUG **************************
        # wh1 = yolo_output[..., 2:4][obj_mask]
        # wh2 = target_box[obj_mask][..., 2:4]
        # print(wh1)
        # print()
        # print(wh2)
        # print("=====================================")
        # *****************************************************

        loss_iou = (self.mse_loss(yolo_output[..., 4][obj_mask], target_iou[obj_mask]) / n1 +
                    self.mse_loss(yolo_output[..., 4][no_obj_mask], target_iou[no_obj_mask]) / n2)
        loss_box_xy = self.mse_loss(yolo_output[..., :2][obj_mask],
                                    target_box[..., :2][obj_mask]) / n1
        loss_box_wh = self.mse_loss(yolo_output[..., 2:4][obj_mask],
                                    target_box[..., 2:4][obj_mask]) / n1
        loss_box = loss_box_xy + loss_box_wh

        loss_class = self.mse_loss(yolo_output[..., 5:][obj_mask],
                                   target_onehot[obj_mask]) / n1
        loss = loss_box + loss_iou + loss_class

        if self.training:
            loss_dict = {
                'loss_box_xy': loss_box_xy.item(),
                'loss_box_wh': loss_box_wh.item(),
                'loss_iou': loss_iou.item(),
                'loss_class': loss_class.item(),
                'loss': loss.item()
            }
            for k in self.loss_dict:
                self.loss_dict[k] += loss_dict[k]

        return loss


    def build_target(self, yolo_output, boxes, labels, anchor):
        '''
        Arguments:
            yolo_output: tensor
            boxes: batch list of numpy of TOP_BOTTOM form, e.g.:[np.array([[x,y,w,h],[]]), np.array([[],[],[]])]
            labels: list of label list, e.g.:[[0,10,14], [...]]
            anchor: list of anchor, e.g.: [[w0,h0], [w1,h1], [w2,h2]]
        '''
        batch_size, grid_num_h, grid_num_w = yolo_output.shape[:3]
        grid_size_w = self.img_size[0] // grid_num_w
        grid_size_h = self.img_size[1] // grid_num_h
        num_anchor = len(anchor)
        anchor = anchor.astype(np.float)

        # obj_mask use uint8 type later converts to True/False position mask in torch
        target_box = np.zeros((batch_size, grid_num_h, grid_num_w, num_anchor, 4), dtype=np.float32)
        target_iou = np.zeros((batch_size, grid_num_h, grid_num_w, num_anchor), dtype=np.float32)
        target_onehot = np.zeros((batch_size, grid_num_h, grid_num_w, num_anchor, self.num_classes),
                                 dtype=np.float32)
        obj_mask = np.zeros((batch_size, grid_num_h, grid_num_w, num_anchor), dtype=np.uint8)
        no_obj_mask = np.ones((batch_size, grid_num_h, grid_num_w, num_anchor), dtype=np.uint8)

        pred_box = yolo_output[..., :4].detach().cpu().numpy().astype(np.float)
        pred_box = yolo_to_box(pred_box, anchor, (self.img_size[0], self.img_size[1]))

        for b in range(batch_size):
            box = boxes[b].copy()  # 由于有多个feature scale，所以这里要复制
            box_CenterWH = self.box_transform(box)
            box_CenterWH[:, 0] /= grid_size_w
            box_CenterWH[:, 1] /= grid_size_h
            cell_idx = np.floor(box_CenterWH[:, :2]).astype('int')
            box_CenterWH[:, :2] -= cell_idx

            pred_expand = pred_box[b].reshape(-1, 4)
            pred_box_iou = bbox_ious(pred_expand, box)
            pred_box_iou = pred_box_iou.reshape(grid_num_h, grid_num_w, num_anchor, len(box_CenterWH))
            pred_box_iou_max = np.max(pred_box_iou, axis=-1)
            no_obj_mask[b][pred_box_iou_max >= self.iou_threshold] = 0

            box_anchor_iou = anchor_intersections(anchor, box)
            anchor_idx = np.argmax(box_anchor_iou, axis=0)

            # place label to their position
            for i in range(len(box_CenterWH)):
                a = anchor_idx[i]
                c_x, c_y = cell_idx[i]
                box_CenterWH[i, 2:] /= anchor[a]
                box_CenterWH[i, 2:] = np.log(box_CenterWH[i, 2:])

                target_box[b, c_y, c_x, a] = box_CenterWH[i]
                target_iou[b, c_y, c_x, a] = pred_box_iou[c_y, c_x, a, i]
                target_onehot[b, c_y, c_x, a, labels[b][i]] = 1
                obj_mask[b, c_y, c_x, a] = 1
                no_obj_mask[b, c_y, c_x, a] = 0

        return target_box, target_iou, target_onehot, obj_mask, no_obj_mask

    def reset_input_size(self, size):
        self.input_size = size

    def reset_loss_dict(self):
        for k in self.loss_dict:
            self.loss_dict[k] = 0

    def build_target1(self, yolo_output, boxes, labels, anchor):
        batch_size, grid_num_h, grid_num_w = yolo_output.shape[:3]
        grid_size_w = self.img_size[0] // grid_num_w
        grid_size_h = self.img_size[1] // grid_num_h

        targets = self.pool.starmap(
            YOLOLoss._build_target_batch,
            ((self.num_classes, (grid_num_w, grid_num_h), (grid_size_w, grid_size_h), boxes[i], labels[i], anchor) for i
             in range(batch_size))
        )

        target_box = np.stack([t[0] for t in targets])
        target_iou = np.stack([t[1] for t in targets])
        target_onehot = np.stack([t[2] for t in targets])
        target_mask = np.stack([t[3] for t in targets])

        return target_box, target_iou, target_onehot, target_mask

    @staticmethod
    def _build_target_batch(num_classes, grid_num, grid_size, box, label, anchor):
        grid_num_w, grid_num_h = grid_num
        grid_size_w, grid_size_h = grid_size
        num_anchor = len(anchor)
        box = box.copy()

        target_box = np.zeros((grid_num_h, grid_num_w, num_anchor, 4), dtype=np.float32)
        target_iou = np.zeros((grid_num_h, grid_num_w, num_anchor), dtype=np.float32)
        target_onehot = np.zeros((grid_num_h, grid_num_w, num_anchor, num_classes),
                                 dtype=np.float32)
        target_mask = np.zeros((grid_num_h, grid_num_w, num_anchor), dtype=np.uint8)

        box_TopBottom = YOLOLoss.box_transform(box)
        box[:, 0] /= grid_size_w
        box[:, 1] /= grid_size_h
        cell_idx = np.floor(box[:, :2]).astype('int')
        box[:, :2] -= cell_idx

        box_anchor_iou = anchor_intersections(anchor, box_TopBottom)
        anchor_idx = np.argmax(box_anchor_iou, axis=0)

        # place label to their position
        for i in range(len(box)):
            a = anchor_idx[i]
            c_x, c_y = cell_idx[i]
            box[i, 2:] /= anchor[a]
            box[i, 2:] = np.log(box[i, 2:])

            target_box[c_y, c_x, a] = box[i]
            target_iou[c_y, c_x, a] = 1
            target_onehot[c_y, c_x, a, label[i]] = 1
            target_mask[c_y, c_x, a] = 1

        return target_box, target_iou, target_onehot, target_mask
