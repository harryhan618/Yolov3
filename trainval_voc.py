import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from config import *
from dataset.pascal_voc import VOCDataset
from net.yolov3.darknet53 import DarkNet53
from utils.tensorboard import TensorBoard
from utils.transforms import *

exp_dir = './experiments/exp4'


with open(os.path.join(exp_dir, "pid.txt"), "w") as f:
    f.write("pid: {}".format(os.getpid()))

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)


device = torch.device(exp_cfg['device'])
tensorboard = TensorBoard(exp_dir)


# ------------ train data ------------
transform_augment = Compose(SquarePad(), RandomFlip(px=0.5), RandomTranslate())
transform_img = Compose(Resize((416, 416)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
transforms = Compose(transform_augment, transform_img)
transform_box_reverse = Box_CenterWH_To_TopBottom()

train_dataset = VOCDataset(VOC_DIR_PATH, exp_cfg['dataset']['image_set'], transforms)
train_loader = DataLoader(train_dataset, batch_size=exp_cfg['dataset']['batch_size'], shuffle=True,
                          collate_fn=train_dataset.collate_fn, num_workers=4)


# ------------ val data ------------
val_year = 2007
val_set = 'test'
transform_img_val = Compose(Resize((416, 416)), ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
val_dataset = VOCDataset(VOC_DIR_PATH, '{}_{}'.format(val_set, val_year), transform_img_val)
val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=val_dataset.collate_fn)

net = DarkNet53((416, 416), 3, train_dataset.num_class, NUM_ANCHOR, anchors, pretrained=True)
# net = nn.DataParallel(net) ## Don't use dataparallel, because loss of different feature scale will be computed on different gpu
if device.type == 'cuda':
    net.cuda(device=device)

if 'warmup' in exp_cfg['lr_scheduler']:
    exp_cfg['optim']['lr'] /= exp_cfg['lr_scheduler']['warmup']
optimizer = optim.SGD(net.parameters(), **exp_cfg['optim'])
if 'warmup' in exp_cfg['lr_scheduler']:
    lr_scheduler_warmup = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: epoch+1)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **exp_cfg['lr_scheduler']['config'])

print("training {}".format(exp_dir))
print("Ready!")


def reset_input_size():
    stride = exp_cfg['dataset']['random scale']['stride']
    m1 = exp_cfg['dataset']['random scale']['min']
    m2 = exp_cfg['dataset']['random scale']['max']
    input_size = np.random.randint(m1, m2+1) * stride
    input_size = (input_size, input_size)
    train_dataset.reset_resize_shape(input_size)
    net.reset_input_size(input_size)


def train(epoch):
    print("Train Epoch: {}".format(epoch))
    net.train()
    net.loss_net.reset_loss_dict()
    reset_input_size()
    train_loss = 0
    progressbar = tqdm(range(len(train_loader)))

    if ('warmup' in exp_cfg['lr_scheduler']) and (epoch<exp_cfg['lr_scheduler']['warmup']):
        lr_scheduler_warmup.step(epoch)
    else:
        lr_scheduler.step(epoch)

    for batch_idx, sample in enumerate(train_loader):
        img = sample['img'].to(device)
        boxes = sample['boxes']
        labels = sample['labels']

        optimizer.zero_grad()
        yolo_outputs, loss = net(img, boxes, labels)
        loss = loss.sum()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
        progressbar.update(1)

    loss_dict = net.loss_net.loss_dict
    for k in loss_dict:
        tensorboard.scalar_summary(k, loss_dict[k], epoch)
    # tensorboard.scalar_summary("train_loss", train_loss, epoch)
    tensorboard.writer.flush()
    progressbar.close()


    if epoch % 10 == 0:
        torch.save(yolo_outputs, os.path.join(exp_dir, "train_outputs.pth"))
        net.cpu()
        save_dict = {
            "epoch": epoch,
            "net": net.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }
        save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth')
        torch.save(save_dict, save_name)
        net.cuda(device=device)
        print("model is saved: {}".format(save_name))

    print("------------------------\n")


def val():
    from utils.evaluate import write_results_voc
    import utils.postprocess as post
    from utils.voc_eval import voc_eval

    save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth')
    save_dict = torch.load(save_name)
    epoch = save_dict['epoch']
    state_dict = save_dict['net']
    net.load_state_dict(state_dict)
    net.cuda(device)

    # reset input_size for net in validation
    input_size = (416, 416)
    val_dataset.reset_resize_shape(input_size)
    net.reset_input_size(input_size)

    bs = val_loader.batch_size
    print("Val Epoch: {}".format(epoch))
    net.eval()
    val_loss = 0
    progressbar = tqdm(range(len(val_loader)))

    det_results = {}
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            img = sample['img'].float().to(device)
            boxes = sample['boxes']
            labels = sample['labels']
            yolo_outputs, loss = net(img, boxes, labels)
            val_loss += loss.item()

            for b in range(len(labels)):
                origin_size = sample['origin_size'][b]
                img_name = sample['img_name'][b]
                img_id = os.path.basename(img_name).split(".")[0]

                outs = []
                for yolo_out, anchor in zip(yolo_outputs, anchors):
                    y = yolo_out[b].detach().cpu().numpy()
                    if batch_idx == 0:
                        np.save(os.path.join(exp_dir, "debug.npy"), y)
                    out = post.yolo_postprocess(y, anchor, origin_size, (416, 416), OBJ_THRESHOLD)
                    if out is not None:
                        outs.append(out)
                if len(outs)==0:
                    continue
                outs = np.vstack(outs).astype(np.float32)
                predict = post.yolo_nms(outs, NMS_THRESHOLD, post_nms=100)
                det_results[img_id] = predict

                if batch_idx < 50:
                    batch_img_tensorboard = []
                    img_name = sample['img_name'][b]
                    origin_img = cv2.imread(img_name)
                    h, w = origin_img.shape[:2]

                    # draw predict
                    draw = origin_img.copy()
                    for i, (x1, y1, x2, y2, cls_index) in enumerate(predict[:, (*range(4), -1)].astype('int')):
                        if i>10: # just draw top 10 predictions
                            break
                        cv2.rectangle(draw, (x1, y1), (x2, y2), (0,255,0), 2)
                        class_name = val_dataset.label_to_cls_name[cls_index]
                        cv2.putText(draw, class_name, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 55), 2)
                    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
                    batch_img_tensorboard.append(draw)

                    # draw ground truth
                    draw = origin_img.copy()
                    box = boxes[b]
                    box = box / 416
                    box = box * [[w, h, w, h]]
                    for i, (x1, y1, x2, y2) in enumerate(box.astype('int')):
                        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        class_name = val_dataset.label_to_cls_name[labels[b][i]]
                        cv2.putText(draw, class_name, (x1, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,55,200), 2)
                    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
                    batch_img_tensorboard.append(draw)

                    tensorboard.image_summary("img_{}".format(batch_idx), batch_img_tensorboard, epoch)
            progressbar.set_description("batch loss: {:.3f}".format(loss.detach().cpu().data.numpy()))
            progressbar.update(1)

    progressbar.close()
    tensorboard.scalar_summary("val_loss", val_loss, epoch)
    tensorboard.writer.flush()

    if len(det_results)==0:
        print("------------------------\n")
        return

    write_results_voc(det_results, val_dataset.label_to_cls_name, os.path.join(exp_dir, "det_results"))

    det_path = os.path.join(exp_dir, "det_results", "{}.txt")
    annopath = os.path.join(VOC_DIR_PATH, "VOCdevkit", "VOC{}".format(val_year), "Annotations", "{}.xml")
    imagesetfile = os.path.join(VOC_DIR_PATH, "VOCdevkit", "VOC{}".format(val_year), "ImageSets", "Main", "{}.txt".format(val_set))
    cache_dir = os.path.join(VOC_DIR_PATH, "cache")

    map = {}
    for class_name in val_dataset.class_names:
        ap = voc_eval(det_path, annopath, imagesetfile, class_name, cache_dir, use_07_metric=False)[2]
        map[class_name] = ap
    with open(os.path.join(exp_dir, "map.txt"), "w") as f:
        f.writelines(["{}: {}\n".format(class_name, ap) for class_name, ap in map.items()])
        f.write("map: {}".format(np.mean(list(map.values()))))

    print("val_map", np.mean(list(map.values())))
    tensorboard.scalar_summary("val_map", np.mean(list(map.values())), epoch)

    print("------------------------\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-r", action="store_true")
    args = parser.parse_args()
    if args.resume:
        save_dict = torch.load(os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth'))
        net.load_state_dict(save_dict['net'])
        optimizer.load_state_dict(save_dict['optim'])
        lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
        start_epoch = save_dict['epoch']+1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, 800):
        train(epoch)
        torch.cuda.empty_cache()
        if epoch%20==0:
            print("\nValidation For Experiment: ", exp_dir)
            print(time.strftime('%H:%M:%S', time.localtime()))
            val()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
