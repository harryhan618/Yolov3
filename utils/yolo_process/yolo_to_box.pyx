import numpy as np
cimport numpy as np

from libc.math cimport exp

DTYPE = np.float
ctypedef np.float_t DTYPE_t


def yolo_to_box(
        np.ndarray[DTYPE_t, ndim=5] pred_box,
        np.ndarray[DTYPE_t, ndim=2] anchors, int img_H, int img_W):
    """
    Argmuents:
        bbox_pred: (batch_size, grid_num_H, grid_num_W, num_anchor, 4) ndarray of float ( sig(x), sig(y), log(w), log(h) )
        anchors: (num_anchor, 2)
        img_H, img_W: input size of CNN
    Return:
         bbox_out: (batch_size, grid_num_H, grid_num_W, num_anchor, 4) (x1, y1, x2, y2)
    """
    return yolo_to_box_c(pred_box, anchors, img_H, img_W)

cdef yolo_to_box_c(
        np.ndarray[DTYPE_t, ndim=5] pred_box,
        np.ndarray[DTYPE_t, ndim=2] anchors, int img_H, int img_W):
    cdef unsigned int batch_size = pred_box.shape[0]
    cdef unsigned int grid_num_H = pred_box.shape[1]
    cdef unsigned int grid_num_W = pred_box.shape[2]
    cdef unsigned int num_anchors = anchors.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=5] box_out = np.zeros((batch_size, grid_num_H, grid_num_W, num_anchors, 4), dtype=DTYPE)

    cdef DTYPE_t x, y, w, h
    cdef unsigned int row, col, a
    for b in range(batch_size):
        for row in range(grid_num_H):
            for col in range(grid_num_W):
                for a in range(num_anchors):
                    x = (pred_box[b, row, col, a, 0] + col) / grid_num_W * img_W
                    y = (pred_box[b, row, col, a, 1] + row) / grid_num_H * img_H
                    w_half = exp(pred_box[b, row, col, a, 2]) * anchors[a][0] * 0.5
                    h_half = exp(pred_box[b, row, col, a, 3]) * anchors[a][1] * 0.5

                    box_out[b, row, col, a, 0] = x - w_half
                    box_out[b, row, col, a, 1] = y - h_half
                    box_out[b, row, col, a, 2] = x + w_half
                    box_out[b, row, col, a, 3] = y + h_half

    return box_out