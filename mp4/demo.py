import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from network import RetinaNet
from detection_utils import compute_targets, get_detections, set_seed
from predict import validate, test
from tensorboardX import SummaryWriter
from absl import app, flags
import numpy as np
from dataset import CocoDataset, Resizer, Normalizer, collater
from torchvision import transforms
import losses
import logging
import time
from dataset import RandomHorizontalFlip
from collections import Counter

FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-2, 'Learning Rate')
flags.DEFINE_float('momentum', 0.9, 'Momentum for optimizer')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight Decay for optimizer')
flags.DEFINE_string('output_dir', 'runs/retina-net-basic/', 'Output Directory')
flags.DEFINE_integer('batch_size', 1, 'Batch Size')
flags.DEFINE_integer('seed', 2, 'Random seed')
flags.DEFINE_integer('max_iter', 100000, 'Total Iterations')
flags.DEFINE_integer('val_every', 10000, 'Iterations interval to validate')
flags.DEFINE_integer('save_every', 5000, 'Iterations interval to validate')
flags.DEFINE_integer('preload_images', 1, 
    'Whether to preload train and val images at beginning of training. Preloading takes about 7 minutes on campus cluster but speeds up training by a lot. Set to 0 to disable.')
flags.DEFINE_multi_integer('lr_step', [30000, 50000], 'Iterations to reduce learning rate')

log_every = 20

def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s: %(levelname)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.getLogger().handlers = []
    if not len(logging.getLogger().handlers): 
        logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

def logger(tag, value, global_step):
    if tag == '':
       logging.info('')
    else:
       logging.info(f'  {tag:>15s} [{global_step:07d}]: {value:5f}')

def set_training_mode(model):
    model.train()
    for module in model.resnet.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

def main(_):
    setup_logging()
    torch.set_num_threads(4)
    torch.manual_seed(FLAGS.seed)
    set_seed(FLAGS.seed)
    
    dataset_train = CocoDataset('train', seed=FLAGS.seed,
        preload_images=FLAGS.preload_images > 0,
        transform=transforms.Compose([Normalizer(), RandomHorizontalFlip(p=0.25), Resizer()]))
    dataset_val = CocoDataset('val', seed=0, 
        preload_images=FLAGS.preload_images > 0,
        transform=transforms.Compose([Normalizer(), Resizer()]))
    train_generator = torch.Generator().manual_seed(FLAGS.seed)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=3,
        collate_fn=collater,
        pin_memory=True,
    ) 
    
    model = RetinaNet(p67=True, fpn=True)

    num_classes = dataset_train.num_classes
    device = torch.device('cuda:0')
    # For Mac users
    # device = torch.device("mps") 
    model.to(device)

    # checkpoint_iter = 40000


    writer = SummaryWriter(FLAGS.output_dir, max_queue=1000, flush_secs=120)
    optimizer = torch.optim.SGD([
        {"params": model.resnet.parameters(), "lr": FLAGS.lr * 0.1},
        {"params": [
            p for name, p in model.named_parameters()
            if not name.startswith("resnet.")
        ], "lr": FLAGS.lr}
    ], momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)
    
    milestones = [int(x) for x in FLAGS.lr_step]


    # hrd = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)

    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=2000)
    
    
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1)


    scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2], optimizer=optimizer)

    # checkpoint = torch.load(
    #     f"{FLAGS.output_dir}/checkpoint_{checkpoint_iter}.pth",
    #     map_location=device,
    #     weights_only=True
    # )

    # model.load_state_dict(checkpoint["model"])
    # optimizer.load_state_dict(checkpoint["optimizer"])

    # checkpoint_iter = checkpoint["iteration"]
    # scheduler.load_state_dict(checkpoint["scheduler"])

    # scheduler2.milestones = Counter([30000, 60000])

    set_training_mode(model)

    optimizer.zero_grad()
    dataloader_iter = None
    
    times_np, cls_loss_np, bbox_loss_np, total_loss_np = [], [], [], []
    lossFunc = losses.LossFunc()
     
    for i in range(FLAGS.max_iter):
        iter_start_time = time.time()
        
        if dataloader_iter is None or i % len(dataloader_iter) == 0:
            dataloader_iter = iter(dataloader_train)
        
        image, cls, bbox, is_crowd, image_id, _ = next(dataloader_iter)
        
        if len(bbox) == 0:
            continue

        image = image.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)
        cls = cls.to(device, non_blocking=True)

        outs = model(image)
        pred_clss, pred_bboxes, anchors = get_detections(outs)
        gt_clss, gt_bboxes = compute_targets(anchors, cls, bbox)
        
        pred_clss = pred_clss.sigmoid()
        classification_loss, regression_loss = lossFunc(pred_clss, pred_bboxes,
                                                        anchors, gt_clss,
                                                        gt_bboxes)
        cls_loss = classification_loss.mean()
        bbox_loss = regression_loss.mean()
        total_loss = cls_loss + bbox_loss
        
        if np.isnan(total_loss.item()):
            logging.error(f'Loss went to NaN at iteration {i+1}')
            break
        
        if np.isinf(total_loss.item()):
            logging.error(f'Loss went to Inf at iteration {i+1}')
            break
        
        total_loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0
        )


        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Some logging
        lr = scheduler.get_last_lr()[0]
        total_loss_np.append(total_loss.item())
        cls_loss_np.append(cls_loss.item())
        bbox_loss_np.append(bbox_loss.item())
        times_np.append(time.time() - iter_start_time)
                      
        if (i+1) % log_every == 0:
            print('')
            writer.add_scalar('iteration_rate', len(times_np) / np.sum(times_np), i+1)
            logger('iteration_rate', len(times_np) / np.sum(times_np), i+1)
            writer.add_scalar('loss_box_reg', np.mean(bbox_loss_np), i+1)
            logger('loss_box_reg', np.mean(bbox_loss_np), i+1)
            writer.add_scalar('lr', lr, i+1)
            logger('lr', lr, i+1)
            writer.add_scalar('loss_cls', np.mean(cls_loss_np), i+1)
            logger('loss_cls', np.mean(cls_loss_np), i+1)
            writer.add_scalar('total_loss', np.mean(total_loss_np), i+1)
            logger('total_loss', np.mean(total_loss_np), i+1)
            times_np, cls_loss_np, bbox_loss_np, total_loss_np = [], [], [], []


        if (i+1) % FLAGS.save_every == 0:
            torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_{i+1}.pth')
            
        if (i+1) % FLAGS.val_every == 0 or (i+1) == FLAGS.max_iter:
            logging.info(f'Validating at {i+1} iterations.')
            val_dataloader = DataLoader(dataset_val, num_workers=3, collate_fn=collater)
            result_file_name = f'{FLAGS.output_dir}/results_{i+1}_val.json'
            model.eval()
            validate(dataset_val, val_dataloader, device, model, result_file_name, writer, i+1)
            set_training_mode(model)

    torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_final.pth')

    # Save prediction result on test set
    dataset_test = CocoDataset('test', preload_images=False,
                               transform=transforms.Compose([Normalizer(), Resizer()])) 
    test_dataloader = DataLoader(dataset_test, num_workers=1, collate_fn=collater)
    result_file_name = f'{FLAGS.output_dir}/results_{FLAGS.max_iter}_test.json'
    model.eval()
    test(dataset_test, test_dataloader, device, model, result_file_name)

if __name__ == '__main__':
    app.run(main)
