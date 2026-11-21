import numpy as np
import torch
import torch.nn as nn
from detection_utils import compute_bbox_targets

class LossFunc(nn.Module):

    def forward(self, classifications, regressions, anchors, gt_clss, gt_bboxes):

        device = classifications.device
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :].to(device)

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            targets_cls = gt_clss[j, :, :]
            targets_bbox = gt_bboxes[j, :, :]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            labels = targets_cls.reshape(-1)


            
            positive_indices = labels > 0
            valid_indices = labels >= 0
            num_positive_anchors = positive_indices.sum()

            probs = classification.float().clamp(1e-4, 1.0 - 1e-4)
            targets = torch.zeros_like(probs, dtype=torch.bool)
            targets[positive_indices, labels[positive_indices].long() - 1] = True

            probs = probs[valid_indices]
            targets = targets[valid_indices]

            p_t = torch.where(targets, probs, 1.0 - probs)
            alpha_t = torch.where(targets, 0.25, 0.75)
            focal_loss = -alpha_t * (1.0 - p_t).square() * p_t.log()

            classification_losses.append(
                focal_loss.sum() / num_positive_anchors.clamp_min(1)
            )

            if num_positive_anchors == 0:
                regression_losses.append(regression.new_zeros(()))
                continue

            # compute the loss for regression
            targets_bbox = targets_bbox[positive_indices, :]

            bbox_reg_target = compute_bbox_targets(anchor[positive_indices, :].reshape(-1,4), targets_bbox.reshape(-1,4))
            targets = bbox_reg_target.to(device)
            # regression_diff = torch.abs(targets - regression[positive_indices, :])
            regression_loss = torch.nn.functional.smooth_l1_loss(
                regression[positive_indices, :],
                targets,
                beta=1.0 / 9.0
            )
            regression_losses.append(regression_loss)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)
