# import numpy as np
# import torch
# import torch.nn as nn
# from detection_utils import compute_bbox_targets


# import torch
# import math
# import torch.nn as nn
# from torchvision.models import resnet50, ResNet50_Weights
# from torchvision.models.feature_extraction import (
#     get_graph_node_names,
#     create_feature_extractor,
# )


# def get_group_gn(dim, dim_per_gp, num_groups):
#     """get number of groups used by GroupNorm, based on number of channels."""
#     assert dim_per_gp == -1 or num_groups == -1, "GroupNorm: can only specify G or C/G."

#     if dim_per_gp > 0:
#         assert dim % dim_per_gp == 0, "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
#         group_gn = dim // dim_per_gp
#     else:
#         assert dim % num_groups == 0, "dim: {}, num_groups: {}".format(dim, num_groups)
#         group_gn = num_groups

#     return group_gn


# def group_norm(out_channels, affine=True, divisor=1):
#     out_channels = out_channels // divisor
#     dim_per_gp = -1 // divisor
#     num_groups = 32 // divisor
#     eps = 1e-5  # default: 1e-5
#     return torch.nn.GroupNorm(
#         get_group_gn(out_channels, dim_per_gp, num_groups), out_channels, eps, affine
#     )


# class Anchors(nn.Module):
#     def __init__(
#         self,
#         stride,
#         sizes=[4, 4 * math.pow(2, 1 / 3), 4 * math.pow(2, 2 / 3)],
#         aspect_ratios=[0.5, 1, 2],
#     ):
#         """
#         Args:
#             stride: stride of the feature map relative to the original image
#             sizes: list of sizes (sqrt of area) of anchors in units of stride
#             aspect_ratios: list of aspect ratios (h/w) of anchors
#         __init__ function does the necessary precomputations.
#         forward function computes anchors for a given feature map

#         Ex. if you are trying the compute the anchor boxes for the above image,
#         you should store a 9*4 tensor in self.anchor_offsets(because we have 9
#         types of anchor boxes and each anchor box has 4 offsets from the
#         location of the center of the box).  

#         HINT: Try using a double for loop to loop over every possible
#         combination of sizes and aspect ratios to find out the coordinates of
#         the anchor box for that particular size and aspect ratio.

#         When calculating the width and height of the anchor box for a
#         particular size and aspect ratio, remember that if the anchor box were
#         a square then the length of each side would just be size*stride;
#         however, we want the height and width of our anchor box to have a
#         different width and height depending on the aspect ratio, so think
#         about how you would use the aspect ratio to appropriately scale the
#         width and height such that w*h = (size*stride)^2 and aspect_ratio = h/w
#         """
#         super(Anchors, self).__init__()
#         self.stride = stride



#         #Aspect ratio: h/w

#         #Stride: How far I move in the original image if I move one unit in the fature map




#         # Initialize a tensor of shape 9*4

#         anchor_boxes = torch.zeros(9, 4)


#         # Have originally square boxes of 3 different sizes
#         # Want to modify them to have different aspect ratios but keep the area in the original image the same

        
#         for i, size in enumerate(sizes):
#             # side lengths of the original square
#             original_side_lengths = size * stride

#             for j, aspect_ratio in enumerate(aspect_ratios):
#                 scaling_factor_k = aspect_ratio ** (0.5)

#                 height = original_side_lengths * scaling_factor_k
#                 width = original_side_lengths / scaling_factor_k

#                 new_x_min = -1 * 0.5 * width 
#                 new_x_max = 0.5 * width
#                 new_y_min = -1 * 0.5 * height
#                 new_y_max = 0.5 * height

#                 anchor_boxes[i*3+j] = torch.tensor((new_x_min, new_y_min, new_x_max, new_y_max))

#         self.register_buffer("anchor_boxes", anchor_boxes)


#     def forward(self, x):
#         """
#         Args:
#             x: feature map of shape (B, C, H, W)
#         Returns:
#             anchors: list of anchors in the format (x1, y1, x2, y2) (in other
#             words (xmin, ymin, xmax, ymax)), giving the shape of (B, A*4, H, W)
#             where A is the number of types of anchor boxes we have.
            
#         HINT 1: We want to loop over every pixel of the input and use that
#         as the center of our bounding boxes. Then we can apply the offsets
#         for each bounding box that we found earlier to get all the bounding
#         boxes for that pixel.  However, remember that this feature map is
#         scaled down from the original image, so when finding the base y, x
#         values(the location of the center of the anchor box with respect to
#         the original image), remember that you need to multiply the current
#         position in our feature map (i,j) by the stride to get what the
#         position of the center would be in the base image.
        
#         HINT 2: the anchor boxes will be identical for all elements of the
#         batch so try just calculating the anchor boxes for one element of
#         the batch and then using torch.repeat to duplicate the tensor B
#         times Hint3, remember to transfer your anchors to the same device
#         that the input x is on before returning(you can access this with
#         x.device). This is so we can use a gpu when training.

#         MAKING CODE EFFICIENT: We recommend first just using for loops and
#         to make sure that your logic is correct. However, this will be very
#         slow. Therefore, we recommend using torch.meshgrid to create the
#         grid of all possible y and x values and then adding them to your
#         anchor offsets tensor that you stored from before to get tensors
#         for x1, x2, y1, and y2 for all the anchor boxes. Then you should
#         simply stack those tensors together, reshape them to match the
#         expected output size and use torch.repeat to repeat that tensor
#         B times across the batch dimension.  Your final code should be
#         fully vectorized and not have any for loops. Also make sure that
#         when you create a tensor you put it on the same device that x is
#         on.
#         """
#         feature_map = x

#         batch_size = feature_map.shape[0]
#         height_of_feature_map = feature_map.shape[2]
#         width_of_feature_map = feature_map.shape[3]
#         num_of_anchors = 9
        

#         x_values = torch.arange(0, feature_map.shape[3], device=x.device)
#         y_values = torch.arange(0, feature_map.shape[2], device=x.device)
        
#         # grid_x contains the x values at each (x,y) coordinate. The indexing is ij so every column
#         # contains the same x values
        
#         # grid_y contains the y values at each (x,y) coordinate. The indexing ij so every row 
#         # contains the same y values

#         grid_y, grid_x = torch.meshgrid(y_values, x_values, indexing='ij')

#         # these extract the corners of every box and adds two dimensions to it so we have shape (9, 1, 1)
#         min_x = self.anchor_boxes[:, 0, None, None]
#         max_x = self.anchor_boxes[:, 2, None, None]

#         min_y = self.anchor_boxes[:, 1, None, None]
#         max_y = self.anchor_boxes[:, 3, None, None]

#         # This maps their coordinates to the coordinates of the input space.
#         center_x = grid_x * self.stride
#         center_y = grid_y * self.stride

#         # this broadcasts the element of each batch to each pixel location and adds them
#         # we have shapes (9, H, W)
#         new_x_min = min_x + center_x
#         new_x_max = max_x + center_x

#         new_y_min = min_y + center_y
#         new_y_max = max_y + center_y

#         # Now we stack them along the second dimension to have shape (9, 4, H, W)
#         anchor_ps = torch.stack([new_x_min, new_y_min, new_x_max, new_y_max], dim = 1)

#         # reshape them to have shape (9*4, H, W)
#         anchor_ps = anchor_ps.reshape(9*4, height_of_feature_map, width_of_feature_map)
#         return anchor_ps.repeat(batch_size, 1, 1, 1).to(x.device)
#         # return the anchors braodcasted across the batch size

#         # new_anchors = torch.zeros(batch_size, num_of_anchors*4, height_of_feature_map, width_of_feature_map)

#         # for anchor_box in range(num_of_anchors):
#         #     for corner in range(4):
#         #         for row in range(feature_map.shape[2]):
#         #             for col in range(feature_map.shape[3]):
#         #                 current_anchor_box = self.anchor_boxes[anchor_box]

#         #                 base_image_x = col * self.stride 
#         #                 base_image_y = row * self.stride


#         #                 current_anchor_point = 0


#         #                 if corner == 0 or corner == 2:
#         #                     current_anchor_point = current_anchor_box[corner] + base_image_x
#         #                 else:
#         #                     current_anchor_point = current_anchor_box[corner] + base_image_y

#         #                 new_anchors[0][anchor_box * 4 + corner][row][col] = current_anchor_point

#         # new_anchors = new_anchors.repeat(batch_size, 1, 1, 1)
#         # new_anchors = new_anchors.to(x.device)
#         # return new_anchors
                


# class RetinaNet(nn.Module):
#     def __init__(self, p67=False, fpn=False,num_anchors=9):
#         super(RetinaNet, self).__init__()
#         self.resnet = create_feature_extractor(
#                 resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
#                 return_nodes={
#                     "layer2.3.relu_2": "conv3",
#                     "layer3.5.relu_2": "conv4",
#                     "layer4.2.relu_2": "conv5",
#                 },
#             )

#         for param in self.resnet.parameters():
#             param.requires_grad = False

#         for name, param in self.resnet.named_parameters():
#             if name.startswith(("layer1.", "layer2.", "layer3.", "layer4.")):
#                 param.requires_grad = True
        
#         self.cls_head, self.bbox_head = self.get_heads(9, num_anchors)

#         self.p67 = p67
#         self.fpn = fpn

#         anchors = nn.ModuleList()

#         self.p5 = nn.Sequential(
#             nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0),
#             group_norm(256),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             group_norm(256),
#         )
#         self._init(self.p5)
#         anchors.append(Anchors(stride=32))

#         if self.p67:
#             self.p6 = nn.Sequential(
#                 nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1),
#                 group_norm(256),
#             )
#             self._init(self.p6)
#             self.p7 = nn.Sequential(
#                 nn.ReLU(),
#                 nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
#                 group_norm(256),
#             )
#             self._init(self.p7)
#             anchors.append(Anchors(stride=64))
#             anchors.append(Anchors(stride=128))

#         if self.fpn:
#             self.p4_lateral = nn.Sequential(
#                 nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
#                 group_norm(256),
#             )
#             self.p4 = nn.Sequential(
#                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), group_norm(256)
#             )
#             self._init(self.p4)
#             self._init(self.p4_lateral)
#             anchors.append(Anchors(stride=16))

#             self.p3_lateral = nn.Sequential(
#                 nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0), group_norm(256)
#             )
#             self.p3 = nn.Sequential(
#                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), group_norm(256)
#             )
#             self._init(self.p3)
#             self._init(self.p3_lateral)
#             anchors.append(Anchors(stride=8))

#         self.anchors = anchors

#     def _init(self, modules):
#         for layer in modules.modules():
#             if isinstance(layer, nn.Conv2d):
#                 nn.init.kaiming_uniform_(layer.weight, a=1)
#                 nn.init.constant_(layer.bias, 0)

#     def to(self, device):
#         super(RetinaNet, self).to(device)
#         self.anchors.to(device)
#         return self

#     def get_heads(self, num_classes, num_anchors, prior_prob=0.01):
#         cls_head = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             group_norm(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             group_norm(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             group_norm(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             group_norm(256),
#             nn.ReLU(),
#             nn.Conv2d(
#                 256, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
#             ),
#         )
#         bbox_head = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             group_norm(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             group_norm(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             group_norm(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             group_norm(256),
#             nn.ReLU(),
#             nn.Conv2d(256, num_anchors * 4, kernel_size=3, stride=1, padding=1),
#         )

#         # Initialization
#         for modules in [cls_head, bbox_head]:
#             for layer in modules.modules():
#                 if isinstance(layer, nn.Conv2d):
#                     torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
#                     torch.nn.init.constant_(layer.bias, 0)

#         # Use prior in model initialization to improve stability
#         bias_value = -(math.log((1 - prior_prob) / prior_prob))
#         torch.nn.init.constant_(cls_head[-1].bias, bias_value)

#         return cls_head, bbox_head

#     def get_ps(self, feats):
#         conv3, conv4, conv5 = feats["conv3"], feats["conv4"], feats["conv5"]
#         p5 = self.p5(conv5)
#         outs = [p5]

#         if self.p67:
#             p6 = self.p6(conv5)
#             outs.append(p6)

#             p7 = self.p7(p6)
#             outs.append(p7)

#         if self.fpn:
#             p4 = self.p4(
#                 self.p4_lateral(conv4)
#                 + nn.Upsample(size=conv4.shape[-2:], mode="nearest")(p5)
#             )
#             outs.append(p4)

#             p3 = self.p3(
#                 self.p3_lateral(conv3)
#                 + nn.Upsample(size=conv3.shape[-2:], mode="nearest")(p4)
#             )
#             outs.append(p3)
#         # outs = [outs[:]]
#         return outs

#     def forward(self, x):
#         feats = self.resnet(x)

#         feats = self.get_ps(feats)

#         # apply the class head and box head on top of layers
#         outs = []
#         for f, a in zip(feats, self.anchors):
#             cls = self.cls_head(f)
#             bbox = self.bbox_head(f)
#             outs.append((cls, bbox, a(f)))
#         return outs


# class LossFunc(nn.Module):

#     def forward(self, classifications, regressions, anchors, gt_clss, gt_bboxes):

#         device = classifications.device
#         batch_size = classifications.shape[0]
#         classification_losses = []
#         regression_losses = []

#         anchor = anchors[0, :, :].to(device)

#         for j in range(batch_size):

#             classification = classifications[j, :, :]
#             regression = regressions[j, :, :]

#             targets_cls = gt_clss[j, :, :]
#             targets_bbox = gt_bboxes[j, :, :]

#             classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

#             labels = targets_cls.reshape(-1)


#             #Gets positive indices
#             positive_indices = labels > 0

#             # Gets the valid indices (background or positive)
#             valid_indices = labels >= 0
#             num_positive_anchors = positive_indices.sum()

#             # Clamp to avoid exploding values
#             probs = classification.float().clamp(1e-4, 1.0 - 1e-4)
#             targets = torch.zeros_like(probs, dtype=torch.bool)

#             # For the positive cases, set teh matches as True otherwise false
#             targets[positive_indices, labels[positive_indices].long() - 1] = True

#             probs = probs[valid_indices]
#             targets = targets[valid_indices]

#             p_t = torch.where(targets, probs, 1.0 - probs)
#             alpha_t = torch.where(targets, 0.25, 0.75)
#             focal_loss = -alpha_t * (1.0 - p_t).square() * p_t.log()

#             # avoidnumerical instability
#             classification_losses.append(
#                 focal_loss.sum() / num_positive_anchors.clamp_min(1)
#             )

#             if num_positive_anchors == 0:
#                 regression_losses.append(regression.new_zeros(()))
#                 continue

#             # compute the loss for regression
#             targets_bbox = targets_bbox[positive_indices, :]

#             bbox_reg_target = compute_bbox_targets(anchor[positive_indices, :].reshape(-1,4), targets_bbox.reshape(-1,4))
#             targets = bbox_reg_target.to(device)
#             regression_diff = torch.abs(targets - regression[positive_indices, :])
#             regression_losses.append(regression_diff.mean())

#         return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)



# import torch
# from torch import nn
# import torchvision
# from torch.utils.data import DataLoader
# from network import RetinaNet
# from detection_utils import compute_targets, get_detections, set_seed
# from predict import validate, test
# from tensorboardX import SummaryWriter
# from absl import app, flags
# import numpy as np
# from dataset import CocoDataset, Resizer, Normalizer, collater
# from torchvision import transforms
# import losses
# import logging
# import time

# FLAGS = flags.FLAGS
# flags.DEFINE_float('lr', 1e-2, 'Learning Rate')
# flags.DEFINE_float('momentum', 0.9, 'Momentum for optimizer')
# flags.DEFINE_float('weight_decay', 1e-4, 'Weight Decay for optimizer')
# flags.DEFINE_string('output_dir', 'runs/retina-net-basic/', 'Output Directory')
# flags.DEFINE_integer('batch_size', 2, 'Batch Size')
# flags.DEFINE_integer('seed', 2, 'Random seed')
# flags.DEFINE_integer('max_iter', 100000, 'Total Iterations')
# flags.DEFINE_integer('val_every', 10000, 'Iterations interval to validate')
# flags.DEFINE_integer('save_every', 5000, 'Iterations interval to validate')
# flags.DEFINE_integer('preload_images', 1, 
#     'Whether to preload train and val images at beginning of training. Preloading takes about 7 minutes on campus cluster but speeds up training by a lot. Set to 0 to disable.')
# flags.DEFINE_multi_integer('lr_step', [75000, 95000], 'Iterations to reduce learning rate')

# log_every = 20

# def setup_logging():
#     log_formatter = logging.Formatter(
#         '%(asctime)s: %(levelname)s %(filename)s:%(lineno)d] %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S')
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(log_formatter)
#     logging.getLogger().handlers = []
#     if not len(logging.getLogger().handlers): 
#         logging.getLogger().addHandler(console_handler)
#     logging.getLogger().setLevel(logging.INFO)

# def logger(tag, value, global_step):
#     if tag == '':
#        logging.info('')
#     else:
#        logging.info(f'  {tag:>15s} [{global_step:07d}]: {value:5f}')

# def main(_):
#     setup_logging()
#     torch.set_num_threads(4)
#     torch.manual_seed(FLAGS.seed)
#     set_seed(FLAGS.seed)
    
#     dataset_train = CocoDataset('train', seed=FLAGS.seed,
#         preload_images=FLAGS.preload_images > 0,
#         transform=transforms.Compose([Normalizer(), Resizer()]))
#     dataset_val = CocoDataset('val', seed=0, 
#         preload_images=FLAGS.preload_images > 0,
#         transform=transforms.Compose([Normalizer(), Resizer()]))
#     dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, pin_memory=True) 
    
#     model = RetinaNet(p67=True, fpn=True)

#     num_classes = dataset_train.num_classes
#     device = torch.device('cuda:0')
#     # For Mac users
#     # device = torch.device("mps") 
#     model.to(device)


#     writer = SummaryWriter(FLAGS.output_dir, max_queue=1000, flush_secs=120)
#     optimizer = torch.optim.SGD([
#         {"params": model.resnet.parameters(), "lr": FLAGS.lr * 0.1},
#         {"params": [
#             p for name, p in model.named_parameters()
#             if not name.startswith("resnet.")
#         ], "lr": FLAGS.lr}
#     ], momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)
    
#     milestones = [int(x) for x in FLAGS.lr_step]


#     scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=2000)
    
    
#     scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
#         optimizer, milestones=milestones, gamma=0.1)


#     scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2], optimizer=optimizer)

#     optimizer.zero_grad()
#     dataloader_iter = None
    
#     times_np, cls_loss_np, bbox_loss_np, total_loss_np = [], [], [], []
#     lossFunc = losses.LossFunc()
     
#     for i in range(FLAGS.max_iter):
#         iter_start_time = time.time()
        
#         if dataloader_iter is None or i % len(dataloader_iter) == 0:
#             dataloader_iter = iter(dataloader_train)
        
#         image, cls, bbox, is_crowd, image_id, _ = next(dataloader_iter)
        
#         if len(bbox) == 0:
#             continue

#         image = image.to(device, non_blocking=True)
#         bbox = bbox.to(device, non_blocking=True)
#         cls = cls.to(device, non_blocking=True)

#         outs = model(image)
#         pred_clss, pred_bboxes, anchors = get_detections(outs)
#         gt_clss, gt_bboxes = compute_targets(anchors, cls, bbox)
        
#         pred_clss = pred_clss.sigmoid()
#         classification_loss, regression_loss = lossFunc(pred_clss, pred_bboxes,
#                                                         anchors, gt_clss,
#                                                         gt_bboxes)
#         cls_loss = classification_loss.mean()
#         bbox_loss = regression_loss.mean()
#         total_loss = cls_loss + bbox_loss
        
#         if np.isnan(total_loss.item()):
#             logging.error(f'Loss went to NaN at iteration {i+1}')
#             break
        
#         if np.isinf(total_loss.item()):
#             logging.error(f'Loss went to Inf at iteration {i+1}')
#             break
        
#         total_loss.backward()

#         # clip the gradient as in the hint
#         torch.nn.utils.clip_grad_norm_(
#             model.parameters(),
#             max_norm=10.0
#         )


        
#         optimizer.step()
#         optimizer.zero_grad()
#         scheduler.step()

#         # Some logging
#         lr = scheduler.get_last_lr()[0]
#         total_loss_np.append(total_loss.item())
#         cls_loss_np.append(cls_loss.item())
#         bbox_loss_np.append(bbox_loss.item())
#         times_np.append(time.time() - iter_start_time)
                      
#         if (i+1) % log_every == 0:
#             print('')
#             writer.add_scalar('iteration_rate', len(times_np) / np.sum(times_np), i+1)
#             logger('iteration_rate', len(times_np) / np.sum(times_np), i+1)
#             writer.add_scalar('loss_box_reg', np.mean(bbox_loss_np), i+1)
#             logger('loss_box_reg', np.mean(bbox_loss_np), i+1)
#             writer.add_scalar('lr', lr, i+1)
#             logger('lr', lr, i+1)
#             writer.add_scalar('loss_cls', np.mean(cls_loss_np), i+1)
#             logger('loss_cls', np.mean(cls_loss_np), i+1)
#             writer.add_scalar('total_loss', np.mean(total_loss_np), i+1)
#             logger('total_loss', np.mean(total_loss_np), i+1)
#             times_np, cls_loss_np, bbox_loss_np, total_loss_np = [], [], [], []


#         if (i+1) % FLAGS.save_every == 0:
#             torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_{i+1}.pth')
            
#         if (i+1) % FLAGS.val_every == 0 or (i+1) == FLAGS.max_iter:
#             logging.info(f'Validating at {i+1} iterations.')
#             val_dataloader = DataLoader(dataset_val, num_workers=3, collate_fn=collater)
#             result_file_name = f'{FLAGS.output_dir}/results_{i+1}_val.json'
#             model.eval()
#             validate(dataset_val, val_dataloader, device, model, result_file_name, writer, i+1)
#             model.train()

#     torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_final.pth')

#     # Save prediction result on test set
#     dataset_test = CocoDataset('test', preload_images=False,
#                                transform=transforms.Compose([Normalizer(), Resizer()])) 
#     test_dataloader = DataLoader(dataset_test, num_workers=1, collate_fn=collater)
#     result_file_name = f'{FLAGS.output_dir}/results_{FLAGS.max_iter}_test.json'
#     model.eval()
#     test(dataset_test, test_dataloader, device, model, result_file_name)

# if __name__ == '__main__':
#     app.run(main)
