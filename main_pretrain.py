# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from tqdm import trange

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as torch_datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from util.datasets import build_hf_dataset
from util.logging import get_logger
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae

from engine_pretrain import train_one_epoch

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

LOG = get_logger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help=
        'Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus'
    )
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument(
        '--accum_iter',
        default=1,
        type=int,
        help=
        'Accumulate gradient iterations (for increasing the effective batch size under memory constraints)'
    )

    # Model parameters
    parser.add_argument('--model',
                        default='mae_vit_large_patch16',
                        type=str,
                        metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size',
                        default=224,
                        type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio',
                        default=0.75,
                        type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument(
        '--norm_pix_loss',
        action='store_true',
        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr',
                        type=float,
                        default=None,
                        metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument(
        '--blr',
        type=float,
        default=1e-3,
        metavar='LR',
        help=
        'base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr',
                        type=float,
                        default=0.,
                        metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs',
                        type=int,
                        default=40,
                        metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path',
                        default='israfelsr/img-wikipedia-simple',
                        type=str,
                        help='dataset path')

    parser.add_argument('--output_dir',
                        default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir',
                        default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help=
        'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url',
                        default='env://',
                        help='url used to set up distributed training')

    # Logging into wandb
    parser.add_argument('--use_wandb',
                        action='store_true',
                        help='Use wandb to track the run')
    parser.add_argument("--wandb_run_name",
                        type=str,
                        help="Set name of run and output folder")

    return parser


def main(args):
    misc.init_distributed_mode(args)  # TODO: check what this thing does

    LOG.info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    LOG.info("{}".format(args).replace(', ', '\n --'))

    if has_wandb:
        if args.use_wandb:
            wandb.init(project="Image-to-Text MAE",
                       name=args.wandb_name,
                       config=args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    LOG.info(f"seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    # TODO: do we want to do flips??
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(
            args.input_size,
            scale=(0.2, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # TODO: load my data
    LOG.info(f"Using dataset from {args.data_path}")
    dataset_train = build_hf_dataset(data_path=args.data_path,
                                     split='train',
                                     streaming=True,
                                     transform=transform_train)
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))
    #else:
    #    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # TODO: we need to shuffle the dataset
    # TODO: decide whether we want to download or use the streaming process

    if args.log_dir is not None:
        # TODO: or eliminate tensorboard or use it instead of wandb
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=0,  #args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model  # without distributed data parallel
    #LOG.info("Model = %s" % str(model_without_ddp))

    # TODO: check if accum_iter change with gradient accumulation
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    LOG.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    LOG.info("actual lr: %.2e" % args.lr)

    # TODO: understand how this works
    LOG.info("accumulate grad iterations: %d" % args.accum_iter)
    LOG.info("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp,
                                                  args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    LOG.info(optimizer)

    loss_scaler = NativeScaler()
    misc.load_model(args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler)

    LOG.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in trange(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model,
                                      data_loader_train,
                                      optimizer,
                                      device,
                                      epoch,
                                      loss_scaler,
                                      log_writer=log_writer,
                                      args=args)

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args,
                            model=model,
                            model_without_ddp=model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch=epoch)

        log_stats = {
            **{f'train_{k}': v
               for k, v in train_stats.items()},
            'epoch': epoch,
        }
        print(log_stats)
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"),
                      mode="a",
                      encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
