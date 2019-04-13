import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from net.yolov3.darknet53 import DarkNet53
from utils.tensorboard import TensorBoard
from utils.transforms import *
import utils.postprocess as post
from utils.evaluate import *
from utils.voc_eval import *


exps = [0, 1]
exp_dirs = ['./experiments/exp{}'.format(i) for i in exps]


transform_img = Compose(Resize((416, 416)), ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
transform_box = Box_TopBottom_To_CenterWH()
transform_box_reverse = Box_CenterWH_To_TopBottom()

val_dataset = VOCDataset(VOC_DIR_PATH, 'val_2012', transform_img, transform_box)
val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=val_dataset.collate_fn)


def val(exp_dir):
    with open(os.path.join(exp_dir, "cfg.json")) as f:
        exp_cfg = json.load(f)

    device = torch.device(exp_cfg['device'])
    tensorboard = TensorBoard(exp_dir)

    net = DarkNet53((416, 416), 3, val_dataset.num_class, NUM_ANCHOR, anchors, pretrained=True)
    net.cuda(device)

    save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth')
    save_dict = torch.load(save_name)
    epoch = save_dict['epoch']
    state_dict = save_dict['net']
    net.load_state_dict(state_dict)
    net.cuda(device)

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
                    out = post.yolo_postprocess(y, anchor, origin_size, (416, 416), OBJ_THRESHOLD)
                    if out is not None:
                        outs.append(out)
                if len(outs)==0:
                    # det_results[img_id] = np.array([], dtype=np.float).reshape(0, 6)
                    continue
                outs = np.vstack(outs).astype(np.float32)
                predict = post.yolo_nms(outs, NMS_THRESHOLD)
                predict = np.array(sorted(predict, key=lambda x: x[4], reverse=True))

                det_results[img_id] = predict

            progressbar.set_description("batch loss: {:.3f}".format(loss.detach().cpu().data.numpy()))
            progressbar.update(1)

    progressbar.close()
    print("------------------------\n")
    write_results_voc(det_results, val_dataset.label_to_cls_name, os.path.join(exp_dir, "det_results"))
    det_results = load_results_voc(val_dataset.cls_name_to_label, os.path.join(exp_dir, "det_results"))

    class_name = "person"

    with open(os.path.join(exp_dir, "map.txt"), "w") as f:
        print(class_name, file=f)
        print(ap_voc(det_results, "val_2012", class_name, 0.5, metric_07=False), file=f)

    det_path = os.path.join(exp_dir, "det_results", "{}.txt")
    annopath = os.path.join(VOC_DIR_PATH, "VOCdevkit", "VOC2012", "Annotations", "{}.xml")
    imagesetfile = os.path.join(VOC_DIR_PATH, "VOCdevkit", "VOC2012", "ImageSets", "Main", "val.txt")
    cache_dir = os.path.join(VOC_DIR_PATH, "cache")
    with open(os.path.join(exp_dir, "map.txt"), "a") as f:
        print(class_name, file=f)
        print(voc_eval(det_path, annopath, imagesetfile, class_name, cache_dir, use_07_metric=False)[2], file=f)



if __name__ == '__main__':
    while True:
        for exp_dir in exp_dirs:
            print("\nExperiment: ", exp_dir)
            print(time.strftime('%H:%M:%S',time.localtime()))
            val(exp_dir)
        torch.cuda.empty_cache()

        import sys
        sys.exit(0)
        time.sleep(60 * 60)