import argparse

import cv2
import os

import torch

from config import *
from net.yolov3.darknet53 import DarkNet53
from utils.transforms import *
import utils.postprocess as post


device = "cuda" if torch.cuda.is_available() else "cpu"
net = DarkNet53((416, 416), 3, 20, NUM_ANCHOR, anchors, pretrained=True)
transform = Compose(Resize((416, 416)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
net = net.to(device)





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", '-i', type=str, default="demo/demo.jpg", help="Path to demo img")
    parser.add_argument("--weight_path", '-w', type=str, help="Path to model weights")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    img_path = args.img_path
    weight_path = args.weight_path

    img = cv2.imread(img_path)
    origin_size = np.array(img.shape[:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = transform(img, None)[0]
    x = x.unsqueeze(0).to(device)

    yolo_outputs, loss = net(x)
    outs = []
    for yolo_out, anchor in zip(yolo_outputs, anchors):
        y = yolo_out[0].detach().cpu().numpy()
        out = post.yolo_postprocess(y, anchor, origin_size, (416, 416), OBJ_THRESHOLD)
        if out is not None:
            outs.append(out)
    outs = np.vstack(outs).astype(np.float32)
    predict = post.yolo_nms(outs, NMS_THRESHOLD)
    predict = np.array(sorted(predict, key=lambda x: x[4], reverse=True))

    for x1, y1, x2, y2 in predict[:, :4].astype('int'):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()