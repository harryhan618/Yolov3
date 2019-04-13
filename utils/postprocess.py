import numpy as np

from .nms import nms
from utils.transforms.transforms import Box_CenterWH_To_TopBottom

box_transformer = Box_CenterWH_To_TopBottom()


def yolo_to_box(yolo_output, anchors, input_size, origin_size=None):
    """
        Converts yolo output to wh_boxes.
        During training, cuda tensor outperforms numpy array. While on cpu, numpy array runs a little bit faster. Thus, both types are implemented.

    Arguments:
        yolo_output: numpy array or detached cuda tensor; is_batch or not depends on shape size
        anchors: numpy array of size (N, 2)
        input_size: input size of CNN, (W, H)
        origin_size: origin size of input img; unsupported in is_batch mode

    Return:
        box in form (x1, y1, x2 ,y2)
    """
    is_batch = False if len(yolo_output.shape) == 4 else True
    if is_batch:
        batch_size, grid_num_h, grid_num_w, num_anchor = yolo_output.shape[:4]
    else:
        grid_num_h, grid_num_w, num_anchor = yolo_output.shape[:3]

    input_size = np.array(input_size, dtype=np.float)
    if origin_size is not None:
        assert not is_batch, "origin_size can't be provided with batch mode"
        anchors = anchors * origin_size / input_size
        size = np.array(origin_size, dtype=np.float)
    else:
        size = input_size

    if isinstance(yolo_output, np.ndarray):
        yolo_output = yolo_output.copy()

        grid_index_x = np.arange(grid_num_w)
        grid_index_y = np.arange(grid_num_h)
        grid_index_w, grid_index_h = np.meshgrid(grid_index_x, grid_index_y)
        grid_index_w = np.repeat(grid_index_w[..., np.newaxis], num_anchor, axis=-1)
        grid_index_h = np.repeat(grid_index_h[..., np.newaxis], num_anchor, axis=-1)

        yolo_output[..., 0] += grid_index_w
        yolo_output[..., 1] += grid_index_h
        yolo_output[..., :2] *= size / [grid_num_w, grid_num_h]
        yolo_output[..., 2:4] = np.exp(yolo_output[..., 2:4])
        yolo_output[..., 2:4] *= anchors

        box_output = np.zeros_like(yolo_output)
        box_output[..., :2] = yolo_output[..., :2] - yolo_output[..., 2:4] / 2
        box_output[..., 2:4] = yolo_output[..., :2] + yolo_output[..., 2:4] / 2

    # elif isinstance(yolo_output, torch.Tensor):
    #     dtype = yolo_output.dtype
    #     device = yolo_output.device
    #
    #     yolo_output = torch.tensor(yolo_output, device=device)
    #     anchors = torch.tensor(anchors, dtype=dtype, device=device)
    #     grid_num = torch.tensor([grid_num_w, grid_num_h], dtype=dtype, device=device)
    #     size = torch.tensor(size, dtype=dtype, device=device)
    #
    #     grid_index_x = torch.arange(grid_num_w, dtype=dtype, device=device)
    #     grid_index_y = torch.arange(grid_num_h, dtype=dtype, device=device)
    #     grid_index_h, grid_index_w = torch.meshgrid([grid_index_x, grid_index_y])
    #     grid_index_w = grid_index_w.unsqueeze(-1).repeat(1, 1, num_anchor).type(dtype).to(device)
    #     grid_index_h = grid_index_h.unsqueeze(-1).repeat(1, 1, num_anchor).type(dtype).to(device)
    #
    #     yolo_output[..., 0] += grid_index_w
    #     yolo_output[..., 1] += grid_index_h
    #     yolo_output[..., :2] *= size / grid_num
    #     yolo_output[..., 2:4] = torch.exp(yolo_output[..., 2:4])
    #     yolo_output[..., 2:4] *= anchors

    return box_output


def yolo_postprocess(yolo_output_np, anchors, origin_size, input_size, obj_threashold):
    '''
    Arguments:
    ----------
    yolo_output_np: an image's output of a feature scale, numpy array shape of [grid_h, grid_w, num_anchor, 5 + num_class]

    Return:
    ----------
    out: np array [x1, y1, x2, y2, confidence, class_idx], np.float
    '''
    grid_num_h, grid_num_w, num_anchor = yolo_output_np.shape[:3]
    origin_size_w, origin_size_h = origin_size

    anchors = anchors * origin_size / input_size
    grid_index_x = np.arange(grid_num_w)
    grid_index_y = np.arange(grid_num_h)
    grid_index_w, grid_index_h = np.meshgrid(grid_index_x, grid_index_y)
    grid_index_w = np.repeat(grid_index_w[..., np.newaxis], num_anchor, axis=-1)
    grid_index_h = np.repeat(grid_index_h[..., np.newaxis], num_anchor, axis=-1)

    yolo_output_np[..., 0] += grid_index_w
    yolo_output_np[..., 1] += grid_index_h
    yolo_output_np[..., :2] *= origin_size / [grid_num_w, grid_num_h]
    yolo_output_np[..., 2:4][yolo_output_np[..., 2:4] >= 7] = 7
    yolo_output_np[..., 2:4] = np.exp(yolo_output_np[..., 2:4])
    yolo_output_np[..., 2:4] *= anchors

    mask = yolo_output_np[..., 4] > obj_threashold
    predict = yolo_output_np[mask]
    num_predict = len(predict)
    if num_predict == 0:
        return None

    predict[:, :4] = box_transformer(predict[:, :4])
    predict[:, [0, 1]] = np.maximum(predict[:, [0, 1]], 0.)
    predict[:, 2] = np.minimum(predict[:, 2], origin_size_w)
    predict[:, 3] = np.minimum(predict[:, 3], origin_size_h)

    predict_class = np.argmax(predict[:, 5:], axis=1)
    class_score = predict[:, 5:][np.arange(num_predict), predict_class].reshape((num_predict, 1))
    predict_class = predict_class.reshape((num_predict, 1))
    out = np.hstack([predict[:, :4], class_score * predict[:, 4:5], predict_class]).astype(np.float)
    return out


def yolo_nms(predict_outs, nms_thresh, post_nms=0, device=None):
    class_list = np.unique(predict_outs[:, -1])
    nms_out = []
    for cls in class_list:
        predict = predict_outs[predict_outs[:, -1] == cls]
        keep_index = nms(predict, nms_thresh, device)
        predict_nms = predict[keep_index]
        nms_out.append(predict_nms)

    nms_out = np.vstack(nms_out)
    nms_out = np.array(sorted(nms_out, key=lambda x: x[4], reverse=True))

    if post_nms>0:
        nms_out = nms_out[:post_nms]
    return nms_out


