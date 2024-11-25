# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from PIL import Image

from detectron2.structures.boxes import BoxMode
from torchvision.ops.boxes import box_area

def bbox_from_joints(joints, margin=5):
    """
    bbox en el formato xyxy
    joints de una persona, solo debe ser de dim (n_jts, 2)
    """

    x_coordinates, y_coordinates = zip(*joints)
    d = margin
    margin = np.array([-d, -d, d, d])
    bbox = [min(x_coordinates), min(y_coordinates),
            max(x_coordinates), max(y_coordinates)]
    bbox = np.array(bbox).astype(int)
    bbox = bbox + margin
    bbox = np.array([c if c > 0 else 0 for c in bbox])
    return bbox

def bbox_from_joints_several(all_joints, margin=5):
    bboxes = []
    for joints in all_joints:
        if len(joints)==0:
            bbox = [0, 0, 0, 0]
        else:
            try:
                bbox = bbox_from_joints(joints, margin)
            except:
                print("error")
                continue
        bboxes.append(bbox)
    bboxes = np.stack(bboxes, 0)
    return bboxes

def mask_joints_w_vis(j2d):
    vis = j2d[0, :, 2].astype(bool)
    j2d_masked = j2d[:, vis]
    return j2d_masked

def crop_image_with_bbox(image, bbox):
    """
    Crops an image to a bounding box.

    Args:
        image (H x W x C).
        bbox (4): Bounding box in xywh format.

    Returns:
        np.ndarray
    """
    bbox = bbox_wh_to_xy(bbox)
    return np.array(Image.fromarray(image).crop(tuple(bbox)))


def make_bbox_square(bbox, bbox_expansion=0.0):
    """

    Args:
        bbox (4 or B x 4): Bounding box in xywh format.
        bbox_expansion (float): Expansion factor to expand the bounding box extents from
            center.

    Returns:
        Squared bbox (same shape as bbox).
    """
    bbox = np.array(bbox)
    original_shape = bbox.shape
    bbox = bbox.reshape(-1, 4)
    center = np.stack(
        (bbox[:, 0] + bbox[:, 2] / 2, bbox[:, 1] + bbox[:, 3] / 2), axis=1
    )
    b = np.expand_dims(np.maximum(bbox[:, 2], bbox[:, 3]), 1)
    b *= 1 + bbox_expansion
    square_bboxes = np.hstack((center - b / 2, b, b))
    return square_bboxes.reshape(original_shape)

def make_bbox_square_xyxy(bbox_in, bbox_expansion=0.0):
    """
    Args:
        bbox (4 or B x 4): Bounding box in xyxy format.
        bbox_expansion (float): Expansion factor to expand the bounding box extents from
            center.
    Returns:
        Squared bbox (same shape as bbox).
    """
    bbox = bbox_xy_to_wh(bbox_in)
    bbox = np.array(bbox) # box in xywh
    original_shape = bbox.shape
    bbox = bbox.reshape(-1, 4)
    center = np.stack(
        (bbox[:, 0] + bbox[:, 2] / 2, bbox[:, 1] + bbox[:, 3] / 2), axis=1
    )
    b = np.expand_dims(np.maximum(bbox[:, 2], bbox[:, 3]), 1)
    b *= 1 + bbox_expansion
    square_bboxes = np.hstack((center - b / 2, b, b))
    square_bboxes = square_bboxes.reshape(original_shape)
    square_bboxes = bbox_wh_to_xy(square_bboxes)
    return square_bboxes

def bbox_xy_to_wh(bbox):
    if not isinstance(bbox, (tuple, list)):
        original_shape = bbox.shape
        bbox = bbox.reshape((-1, 4))
    else:
        original_shape = None
    bbox = BoxMode.convert(
        box=bbox, from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYWH_ABS
    )
    if original_shape is not None:
        return bbox.reshape(original_shape)
    return bbox


def bbox_wh_to_xy(bbox):
    if not isinstance(bbox, (tuple, list)):
        original_shape = bbox.shape
        bbox = bbox.reshape((-1, 4))
    else:
        original_shape = None
    bbox = BoxMode.convert(
        box=bbox, from_mode=BoxMode.XYWH_ABS, to_mode=BoxMode.XYXY_ABS
    )
    if original_shape is not None:
        return bbox.reshape(original_shape)
    return bbox


def check_overlap(bbox1, bbox2):
    """
    Checks if 2 boxes are overlapping. Also works for 2D tuples.

    Args:
        bbox1: [x1, y1, x2, y2] or [z1, z2]
        bbox2: [x1, y1, x2, y2] or [z1, z2]

    Returns:
        bool
    """
    if bbox1[0] > bbox2[2] or bbox2[0] > bbox1[2]:
        return False
    if len(bbox1) > 2:
        if bbox1[1] > bbox2[3] or bbox2[1] > bbox1[3]:
            return False
    return True


def compute_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def compute_area_several(bboxes):
    areas = []
    for bbox in bboxes:
        area = compute_area(bbox)
        areas.append(area)
    areas = np.stack(areas, 0)
    return areas


def compute_iou(bbox1, bbox2):
    """
    Computes Intersection Over Union for two boxes.

    Args:
        bbox1 (np.ndarray or torch.Tensor): (x1, y1, x2, y2).
        bbox2 (np.ndarray or torch.Tensor): (x1, y1, x2, y2).
    """
    a1 = compute_area(bbox1)
    a2 = compute_area(bbox2)
    if isinstance(bbox1, np.ndarray):
        lt = np.maximum(bbox1[:2], bbox2[:2])
        rb = np.minimum(bbox1[2:], bbox2[2:])
        wh = np.clip(rb - lt, a_min=0, a_max=None)
    else:
        stack = torch.stack((bbox1, bbox2))
        lt = torch.max(stack[:, :2], 0).values
        rb = torch.min(stack[:, 2:], 0).values
        wh = torch.clamp_min(rb - lt, 0)
    inter = wh[0] * wh[1]
    return inter / (a1 + a2 - inter)


def box_iou_np(boxes1, boxes2):
    '''
    this takes xyxy format
    this convers the np input to torch to operate
    '''
    boxes1 = torch.tensor(boxes1, dtype=torch.float32)
    boxes2 = torch.tensor(boxes2, dtype=torch.float32)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    # return iou.numpy(), union.numpy()
    return iou.numpy()


def clip_bboxes(bboxes):
    cliped_boxes = []
    for this_bbox in bboxes:
        x1, y1, x2, y2 = this_bbox
        x1 = np.clip(x1, 0, 1920)
        x2 = np.clip(x2, 0, 1920)
        y1 = np.clip(y1, 0, 1080)
        y2 = np.clip(y2, 0, 1080)
        bbox = np.array([x1, y1, x2, y2])
        cliped_boxes.append(bbox)
    cliped_boxes = np.stack(cliped_boxes)
    return cliped_boxes