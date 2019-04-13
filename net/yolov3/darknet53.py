import torch
import torch.nn as nn
import torch.nn.functional as F

from net.darknet import darknet53, _Conv2d,BasicBlock
from .yolo_loss import YOLOLoss

pretrained_path = './net/pretrained_weights/[darknet]_darknet53.pth'


class DarkNet53(nn.Module):
    def __init__(self, input_size, num_feature_scale, num_classes, num_anchors=None, anchors=None, pretrained=True):
        '''
        In forward function, feature_map_scale is processed from deep feature map to shallow one.
        Thus, anchors should be store in deep to shallow order

        Arguments:
        ----------
        input_size: input size for CNN, (W, H)
        num_feature_scale: num of feature scale in which yolo detection is done
        anchors: np array of shape (num_feature_scale, num_anchors, 4)
        pretrained: use pretrained backbone feature extractor
        '''
        super(DarkNet53, self).__init__()
        self.input_size = input_size
        self.num_feature_scale = num_feature_scale
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchors = anchors

        self.backbone = darknet53(pretrained_path) if pretrained else darknet53()
        self.feature_map_channel = self.backbone.feature_map_channel

        self.yolo_detection = nn.ModuleList()
        self.route = nn.ModuleList()
        self.route.add_module("route_0 - empty_layer", nn.Sequential()) # a placeholder layer for iteration
        self.yolo_output = nn.ModuleList()
        for i in range(1, num_feature_scale + 1):
            # module name "yolo_1" represents the deepest layer, and so on
            yolo_detection = nn.Sequential()
            for j in range(2):
                yolo_layer = BasicBlock(self.feature_map_channel[-i], self.feature_map_channel[-i-1])
                yolo_detection.add_module("yolo_block_{}".format(j), yolo_layer)
            self.yolo_detection.add_module("yolo_detection_{}".format(i), yolo_detection)

            if i < num_feature_scale:
                route_layer = _Conv2d(self.feature_map_channel[-i], self.feature_map_channel[-i-1], 1)
                self.route.add_module("route_{}".format(i), route_layer)

            yolo_out_layer = nn.Sequential(
                BasicBlock(self.feature_map_channel[-i], self.feature_map_channel[-i - 1]),
                nn.Conv2d(self.feature_map_channel[-i], num_anchors * (5 + num_classes), 1)
            )
            self.yolo_output.add_module("yolo_output_{}".format(i), yolo_out_layer)

        self.loss_net = YOLOLoss(num_classes, input_size)

    def forward(self, img, boxes, labels):
        """
        Return:
            outputs list is from deepest feature map to shallow feature map
        """
        features = self.backbone(img)[::-1]  # from deep layer to shallow layer
        outputs = []
        loss = 0

        for i, (feature, yolo_layer, route_layer, yolo_out_layer) in enumerate(
                zip(features, self.yolo_detection, self.route, self.yolo_output)):
            if i > 0:
                upsample = F.interpolate(y, scale_factor=2)
                route = route_layer(upsample)
                feature = route + feature

            y = yolo_layer(feature)
            output = yolo_out_layer(y)
            output = self.yolo_process(output)
            outputs.append(output)
            loss += self.loss_net(output, boxes, labels, self.anchors[i])

        return outputs, loss

    def yolo_process(self, y):
        box_attrs = 5 + self.num_classes
        batch_size = y.shape[0]
        grid_num_h, grid_num_w = y.shape[-2], y.shape[-1]

        y = y.permute(0, 2, 3, 1)
        y = y.view(batch_size, grid_num_h, grid_num_w, self.num_anchors, box_attrs).contiguous()

        output_xy = torch.sigmoid(y[..., :2])
        output_wh = y[..., 2:4]
        output_iou = torch.sigmoid(y[..., 4:5])
        output_onehot = torch.sigmoid(y[..., 5:])

        output = torch.cat([output_xy, output_wh, output_iou, output_onehot], dim=-1)

        return output

    def reset_input_size(self, size):
        self.input_size = size
        self.loss_net.reset_input_size(size)



if __name__ == '__main__':
    x = torch.randn(2, 3, 416, 416, dtype=torch.float32)
    net = DarkNet53((416, 416), 3, 80, 3, pretrained=False)
    print(net)
    y = net(x, None, None)
    for out in y[0]:
        print(out.shape)
