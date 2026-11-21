import torch
import numpy as np
import random

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def get_detections(outs):
    B, BB, _, _ = outs[0][0].shape
    _, A, _, _ = outs[0][2].shape
    A = A // 4
    num_classes = BB // A
    
    pred_bboxes, pred_clss, anchors = [], [], []
    for pred_cls, pred_bbox, anchor in outs:
        # Get all the anchors, pred and bboxes
        H, W = pred_cls.shape[-2:]
        pred_cls = pred_cls.reshape(B, A, -1, H, W)
        pred_bbox = pred_bbox.reshape(B, A, -1, H, W)

        pred_clss.append(pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes))
        pred_bboxes.append(pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))

        anchor = anchor.reshape(B, A, -1, H, W)
        anchors.append(anchor.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))
    pred_clss = torch.cat(pred_clss, dim=1)
    pred_bboxes = torch.cat(pred_bboxes, dim=1)
    anchors = torch.cat(anchors, dim=1)
    return pred_clss, pred_bboxes, anchors

def compute_bbox_iou(bbox1, bbox2, dim=1):
    """
    Args:
        bbox1: (N, 4) tensor of (x1, y1, x2, y2)
        bbox2: (M, 4) tensor of (x1, y1, x2, y2)
    Returns
        iou: (N, M) tensor of IoU values
    """
    bbox1 = bbox1.unsqueeze(1)
    bbox2 = bbox2.unsqueeze(0)

    # print("bbox1 device:", bbox1.device)
    # print("bbox2 device:", bbox2.device)

    max_min_x = torch.max(bbox1[...,0], bbox2[...,0])
    min_max_x = torch.min(bbox1[...,2], bbox2[...,2])
    max_min_y = torch.max(bbox1[...,1], bbox2[...,1])
    min_max_y = torch.min(bbox1[...,3], bbox2[...,3])
    intersection = torch.clamp(min_max_x - max_min_x, min=0) * torch.clamp(min_max_y - max_min_y, min=0)
    area1 = (bbox1[...,2] - bbox1[...,0]) * (bbox1[...,3] - bbox1[...,1])
    area2 = (bbox2[...,2] - bbox2[...,0]) * (bbox2[...,3] - bbox2[...,1])
    iou = intersection / (area1 + area2 - intersection)
    return iou

def compute_targets(anchor, cls, bbox):
    """
    Args:
        anchor: batch of anchors in the format (x1, y1, x2, y2) or in other
        words (xmin, ymin, xmax, ymax); shape is (B, A, 4), where B denotes
        image batch size and A denotes the number of anchors
        cls: groundtruth object classes of shape (B, number of objects in the image, 1)
        bbox: groundtruth bounding boxes of shape (B, number of objects in the image, 4)
    Returns:
        gt_clss: groundtruth class targets of shape (B, A, 1)
        gt_bboxes: groundtruth bbox targets of shape (B, A, 4)
    
    Hint: remember if the max_iou for that bounding box is between [0, 0.4)
    then the gt_cls should equal 0(because it is being assigned as background)
    and the gt_bbox should be all zero(it can be anything since it will be
    ignored however our tests set them to zero so you should too).  Also, if
    the max iou is between [0.4, 0.5) then the gt_cls should be equal to
    -1(since it's neither background or assigned to a class. This is basically
    tells the model to ignore this box) and the gt_bbox should again
    arbitrarilarly be set to all zeros).  Otherwise if the max_iou > 0.5, you
    should assign the anchor to the gt_box with the max iou, and the gt_cls
    will be the ground truth class of that max_iou box
    
    Hint: use torch.max to get both the max iou and the index of the max iou.

    Hint: We recommend using the compute_bbox_iou function which efficently
    computes the ious between two lists of bounding boxes as a helper function.

    Hint: make sure that the returned gt_clss tensor is of type int(since it
    will be used as an index in the loss function). Also make sure that both
    the gt_bboxes and gt_clss are on the same device as the anchor.  You can do
    this by calling .to(anchor.device) on the tensor you want to move to the
    same device as the anchor.

    VECTORIZING CODE: Again, you can use for loops initially to make the tests
    pass, but in order to make your code efficient during training, you should
    only have one for loop over the batch dimension and everything else should
    be vectorized. We recommend using boolean masks to do this. i.e
    you can compute the max_ious for all the anchor boxes and then do
    gt_cls[max_iou < 0.4] = 0 to access all the anchor boxes that should be set
    to background and setting their gt_cls to 0.  This will remove the need for
    a for loop over all the anchor boxes. You can then do the same for the
    other cases. This will make your code much more efficient and faster to
    train.
    """
    # Args: 
    #   (B, n, 4) number of ground truth boxes
    #   (B, n, 1) number of ground truth classes
    #   (B, A, 4) number of current anchor boxes, where A is the number of anchor boxes. 
    

    anchors = anchor.cuda()
    bbox = bbox.cuda()
    cls = cls.cuda()

    batch_size = anchor.shape[0]
    number_of_anchor_boxes = anchor.shape[1]

    gt_cls = torch.zeros(
        batch_size, number_of_anchor_boxes, 1,
        device=anchors.device
    )

    gt_bboxes = torch.zeros(
        batch_size, number_of_anchor_boxes, 4,
        device=anchors.device
    )


    # print("anchors:", anchors.device)
    # print("bbox:", bbox.device)
    # print("cls:", cls.device)

    for batch in range(batch_size):
            current_batch_of_anchor_boxes = anchors[batch]
            current_batch_of_gt_boxes = bbox[batch]
            ious = compute_bbox_iou(current_batch_of_anchor_boxes,
                                    current_batch_of_gt_boxes)



            # print("ious", ious)
            # print("anchors", current_batch_of_anchor_boxes)
            # print("gt_boxes", current_batch_of_gt_boxes)


            # (A, 1).            (A, 1)
            max_ious_for_boxes, indices_where_they_occur = torch.max(ious, dim=1)

            # print(ious)

            background_mask = (max_ious_for_boxes < 0.4)
            ignore_mask = (max_ious_for_boxes >= 0.4) & (max_ious_for_boxes < 0.5)
            good_mask = (max_ious_for_boxes >= 0.5) & (max_ious_for_boxes <= 1)

            # print("Background",background_mask)
            # print("Ignor",ignore_mask)
            # print("Good",good_mask)
            # print(max_ious_for_boxes)
            # print(indices_where_they_occur)
            

            gt_cls[batch, background_mask] = 0
            gt_cls[batch, ignore_mask] = -1
            gt_cls[batch, good_mask] = cls[batch][indices_where_they_occur[good_mask]].reshape(-1, 1).to(gt_cls.dtype)

            gt_bboxes[batch, good_mask] = bbox[batch][indices_where_they_occur[good_mask]]


    
    # TODO(student): Complete this function
    
    return gt_cls.to(torch.int), gt_bboxes

def compute_bbox_targets(anchors, gt_bboxes):
    """
    Args:
        anchors: anchors in the format (x1, y1, x2, y2); shape is (A, 4)
        gt_bboxes: groundtruth bounding boxes assigned to the anchors, in the
        format (x1, y1, x2, y2); shape is (A, 4)
    Returns:
        bbox_reg_target: regression offset of shape (A, 4)
    
    Remember that the delta_x and delta_y we compute are with respect to the
    center of the anchor box. I.E, we're seeing how much that center of the
    anchor box changes.  We also need to normalize delta_x and delta_y which
    means that we need to divide them by the width or height of the anchor box
    respectively. This is to make our regression targets more invariant to the
    size of the original anchor box. So, this means that: delta_x =
    (gt_bbox_center_x - anchor_center_x) / anchor_width  and delta_y would be
    computed in a similar manner.

    When computing delta_w and delta_h, there are a few things to note.
    1. We also want to normalize these with respect to the width and height of
    the anchor boxes. so delta_w = gt_bbox_width / anchor_width
    2. Logarithm: In order to make our regresssion targets better handle
    varying sizees of the bounding boxes, we use the logarithmic scale for our
    delta_w and delta_h. This is to ensure that if for example the gt_width is
    twice or 1/2 the size of the anchor_width, the magnitude in the log scale
    would stay the same but only the sign of our regression target would be
    different. Therefore our formula changes to delta_w = log(gt_bbox_width /
    anchor_width)
    3. Clamping: Remember that logarithms can't handle negative values and that
    the log of values very close to zero will have very large magnitudes and
    have extremly high gradients which might make training unstable. To
    mitigate this we use clamping to ensure that the value that we log isn't
    too small. Therefore, our final formula will be
       delta_w = log(max(gt_bbox_width,1) / anchor_width)
    """

    anchor_sizes = anchors[:, 2:] - anchors[:, :2]
    gt_sizes = gt_bboxes[:, 2:] - gt_bboxes[:, :2]

    anchor_centers = (anchors[:, :2] + anchors[:, 2:]) * 0.5
    gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) * 0.5

    center_deltas = (gt_centers - anchor_centers) / anchor_sizes

    size_deltas = torch.log(gt_sizes.clamp_min(1) / anchor_sizes)

    return torch.cat((center_deltas, size_deltas), dim=1)

    # anchors_size = anchors.shape[0]
    # deltas = []

    # for anchor in rsange(anchors_size):

    #     (x_min, y_min, x_max, y_max) = anchors[anchor] 

    #     anchor_center_x = (x_min + x_max) * 0.5
    #     anchor_center_y = (y_min + y_max) * 0.5

    #     anchor_width = (x_max - x_min)
    #     anchor_height = (y_max - y_min)

    #     (x_min_gt, y_min_gt, x_max_gt, y_max_gt) = gt_bboxes[anchor] 

    #     gt_center_x = (x_min_gt + x_max_gt) * 0.5
    #     gt_center_y = (y_min_gt + y_max_gt) * 0.5


    #     gt_bbox_width = (x_max_gt - x_min_gt)
    #     gt_bbox_height = (y_max_gt - y_min_gt)

    #     delta_x = (gt_center_x - anchor_center_x) / anchor_width
    #     delta_y = (gt_center_y - anchor_center_y) / anchor_height

    #     delta_w = np.log(max(gt_bbox_width, 1) / anchor_width)
    #     delta_h = np.log(max(gt_bbox_height, 1) / anchor_height)

    #     deltas.append((delta_x, delta_y, delta_w, delta_h))

    # deltas = torch.tensor(deltas)
    # deltas = deltas.reshape(anchors_size, 4)

    # print(deltas)
    # return deltas

def apply_bbox_deltas(boxes, deltas):
    """
    Args:
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        deltas: (N, 4) tensor of (dxc, dyc, dlogw, dlogh)
    Returns
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        
    """
    # TODO(student): Complete this function
    deltas = deltas.to(boxes.device)

    top_left = boxes[:, :2]
    bottom_right = boxes[:, 2:]

    sizes = bottom_right - top_left
    centers = (top_left + bottom_right) * 0.5

    new_centers = centers + sizes * deltas[:, :2]
    half_sizes = 0.5 * sizes * torch.exp(deltas[:, 2:])

    return torch.cat(
        (new_centers - half_sizes, new_centers + half_sizes),
        dim=1,
    )


    # # (pw, ph)
    # anchor_box_sizes = boxes[:, 2:] - boxes[:, :2]

    # # (pw*tx, ph*ty)
    # scaled = anchor_box_sizes * deltas[:, :2]
   

    # # (px, py)
    # anchor_box_centers = (boxes[:, 2:] + boxes[:, :2]) * 0.5


    # new_sizes = anchor_box_sizes *  torch.exp(deltas[:, 2:])

    # # (bx, by)
    # new_centers = anchor_box_centers + scaled

    # x_min = new_centers[:, :1] - (0.5 * new_sizes[:, :1])
    # x_max = new_centers[:, :1] + (0.5 * new_sizes[:, :1])

    # y_min = new_centers[:, 1:] - (0.5 * new_sizes[:, 1:])
    # y_max = new_centers[:, 1:] + (0.5 * new_sizes[:, 1:])

    # s = torch.stack([x_min, y_min, x_max, y_max], dim=1).reshape(x_min.shape[0], 4)


    return s

def nms(bboxes, scores, threshold=0.5):
    """
    Args:
        bboxes: (N, 4) tensor of (x1, y1, x2, y2)
        scores: (N,) tensor of scores
    Returns:
        keep: (K,) tensor of indices to keep
    
    Remember that nms is used to prevent having many boxes that overlap each
    other. To do this, if multiple boxes overlap each other beyond a threshold
    iou, nms will pick the "best" box(the one with the highest score) and
    remove the rest. One way to implement this is to first compute the ious
    between all pairs of bboxes. Then loop over the bboxes from highest score
    to lowest score. Since this is the best bbox(the one with the highest
    score), It will be choosen over all overlapping boxes. Therefore, you
    should add this bbox to your final resulting bboxes and remove all the
    boxes that overlap with it from consideration. Then repeat until you've
    gone through all of the bboxes. Make sure that the indices tensor that you
    return is of type int or long(since it will be used as an index to select
    the relevant bboxes to output).
    """
    order = scores.argsort(descending=True)
    if order.numel() == 0:
        return order

    keep = []


    # If the current list of boxes are non empty keep doing
    while order.numel() > 0:

        # Take out the best score
        keep.append(order[0])

        remaining = order[1:]

        if remaining.numel() == 0:
            break

        # the box of the most confident score
        best_box = bboxes[order[:1]]

        # compute all pairs of ious
        all_pairs_ious = compute_bbox_iou(best_box, bboxes[remaining]).squeeze(0)

        # keep the entries where ious are better than some threshold
        order = remaining[all_pairs_ious <= threshold]

    return torch.stack(keep)
