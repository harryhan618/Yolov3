import json
import os
from dataset.pascal_voc import VOCDataset
from utils.iou import bbox_ious
from config import *


def write_results_voc(det_results, label_to_cls_name, path):
    """
    Write results to file. Each file stores one class of det_results.
    Each line is in form "img_id  score  x1  y1  x2  y2"

    Arguments:
    ----------
    det_results: dict, {"img_id": np.array[[x1, y1, x2, y2, confidence, class_label]]}
    label_to_cls_name: dict, map label to class name
    path: directory path to write in
    """
    os.makedirs(path, exist_ok=True)

    for label_idx, class_name in label_to_cls_name.items():
        with open(os.path.join(path, "{}.txt".format(class_name)), "w") as f:
            for img_id, det in det_results.items():
                det_cls = det[det[:, -1] == label_idx]
                if len(det_cls) == 0:
                    continue
                f.writelines(
                    "{img_id} {score} {x1} {y1} {x2} {y2}\n".format(img_id=img_id, score=s, x1=x1, y1=y1, x2=x2, y2=y2)
                    for x1, y1, x2, y2, s, l in det_cls)


def load_results_voc(cls_name_to_label, path):
    """
    Arguments:
    -----------
    cls_name_to_label: dict, map class name to label
    path: directory path to save det_results, in which every class with a text file
    :return:
    """
    det_results = {}
    for class_name in cls_name_to_label:
        with open(os.path.join(path, "{}.txt".format(class_name))) as f:
            for line in f:
                line = line.strip().split(" ")
                img_id = line[0]
                if not img_id in det_results:
                    det_results[img_id] = []
                det = [float(x) for x in line[2:]]
                det.append(float(line[1])) # confidence
                det.append(cls_name_to_label[class_name])
                det_results[img_id].append(det)
    for img_id in det_results:
        det_results[img_id] = np.array(det_results[img_id])
    return det_results


def write_results_coco(det_results, label_to_cls_name, path):
    result_list = []
    for img_id, det in det_results.items():
        for x1, y1, x2, y2, s, l in det:
            result_list.append(
                {
                    "category_id": l,
                    "score": s,
                    "image_id": img_id,
                    "bbox": [x1, y1, x2, y2]
                }
            )

    with open(os.path.join(path, "result_coco_form.json"), "w") as f:
        json.dump(result_list, f, indent=4)


def ap_voc(det_results, image_set, class_name, thresh, use_difficult=False, metric_07=False):
    test_dataset = VOCDataset(VOC_DIR_PATH, image_set, for_map=True)
    label = test_dataset.cls_name_to_label[class_name]

    score = []
    tp = []
    fp = []
    n_positive = 0

    for i, img_id in enumerate(test_dataset.img_list):
        gt_dict = test_dataset[i]
        gt_boxes = gt_dict['boxes'][gt_dict['labels']==label]
        difficult = gt_dict['difficult'][gt_dict['labels']==label].astype('bool')
        n_positive += sum((~difficult)) if not use_difficult else len(difficult)

        # in case false negative
        if img_id not in det_results:
            continue

        pred = det_results[img_id]
        pred = pred[pred[:, -1]==label]
        if len(pred) == 0:
            continue
        pred = pred[np.argsort(pred[:, 4])[::-1]]
        score.extend(pred[:, 4].tolist())

        if len(gt_boxes) == 0: # or (use_difficult and len(gt_boxes[~difficult.astype('bool')])==0):
            tp.extend([0] * len(pred))
            fp.extend([1] * len(pred))
            continue

        img_tp = [0] * len(pred)
        img_fp = [0] * len(pred)
        iou = bbox_ious(pred[:, :4].astype('float'), gt_boxes.astype('float'))

        gt_matched = [False] * len(gt_boxes)
        for pred_idx in range(len(pred)):
            match_idx = np.argmax(iou[pred_idx])
            if iou[pred_idx, match_idx] > thresh:
                if not use_difficult and difficult[match_idx]:
                    continue
                if not gt_matched[match_idx]:
                    gt_matched[match_idx] = True
                    img_tp[pred_idx] = 1
                else:
                    img_fp[pred_idx] = 1
            else:
                img_fp[pred_idx] = 1
        tp.extend(img_tp)
        fp.extend(img_fp)

    idx_sequence = np.argsort(score)[::-1]
    tp = np.array(tp)[idx_sequence]
    fp = np.array(fp)[idx_sequence]

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    precision = tp / np.maximum(tp+fp, np.finfo(np.float64).eps)
    recall = tp / n_positive


    if metric_07:
        ap = 0.0
        for t in np.linspace(0, 1.0, 11):
            if np.sum(recall >= t) > 0:
                ap += np.max(precision[recall >= t])
        ap /= 11.
    else:
        rec = np.concatenate(([0.], recall, [1.]))
        prec = np.concatenate(([0.], precision, [1.]))

        for i in range(prec.size-1, 0, -1):
            prec[i] = np.maximum(prec[i-1], prec[i])

        ap = np.sum([(rec[i+1] - rec[i]) * prec[i+1] for i in range(prec.size-1)])

    return ap


def map_voc(det_results, image_set, thresh=0.5, use_difficult=False, metric_07=False):
    class_names = VOCDataset.class_names
    map = {}
    for name in class_names:
        ap = ap_voc(det_results, image_set, name, thresh, use_difficult, metric_07=metric_07)
        map[name] = ap

    return sum(map.values()) / len(map), map