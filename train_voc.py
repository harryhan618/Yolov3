import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset.pascal_voc import VOCDataset
from net.yolov3.darknet53 import DarkNet53
# from net.darknet53_1 import DarkNet53
from utils.tensorboard import TensorBoard
from utils.transforms import *

exp_dir = './experiments/exp0'
with open(os.path.join(exp_dir, "pid.txt"), "w") as f:
    f.write("pid: {}".format(os.getpid()))

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)


device = torch.device(exp_cfg['device'])
tensorboard = TensorBoard(exp_dir)

transform_augment = Compose(SquarePad(), RandomFlip(px=0.5), RandomTranslate())
transform_img = Compose(Resize((416, 416)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
transforms = Compose(transform_augment, transform_img)
transform_box_reverse = Box_CenterWH_To_TopBottom()

train_dataset = VOCDataset(VOC_DIR_PATH, exp_cfg['dataset']['image_set'], transforms)
train_loader = DataLoader(train_dataset, batch_size=exp_cfg['dataset']['batch_size'], shuffle=True,
                          collate_fn=train_dataset.collate_fn, num_workers=4)

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
    tensorboard.writer.flush()
    progressbar.close()


    if epoch % 20 == 0:
        torch.save(yolo_outputs, os.path.join(exp_dir, "train_outputs.pth"))
        save_dict = {
            "epoch": epoch,
            "net": net.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }
        save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth')
        torch.save(save_dict, save_name)
        print("model is saved: {}".format(save_name))

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
        start_epoch = save_dict['epoch']
    else:
        start_epoch = 0

    for epoch in range(start_epoch, 30000):
        train(epoch)


if __name__ == "__main__":
    main()
