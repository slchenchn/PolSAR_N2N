'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-05-22
	content: 
'''
''' 
'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# cuda_idx = [0]
import os.path as osp
import sys
import time
import shutil
import random
import argparse


import yaml
import torch
import datetime
import numpy as np
import glob
import natsort
import re
import logging
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torch.utils import data 
from torchlars import LARS

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader 
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.loader.rand_pool import generate_mask_pair, generate_subimages

from mylib import nestargs
from mylib import types
from mylib import file_utils as fu
from mylib import my_torch_tools as tt
from mylib.torchsummary import summary
from utils import args
from utils import utils

def train(cfg, writer, logger):

    # Setup Augmentations
    augmentations = cfg.train.augment
    logger.info(f'using augments: {augmentations}')
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg.data.dataloader)
    data_path = cfg.data.path
    logger.info("data path: {}".format(data_path))

    t_loader = data_loader(
        data_path,
        data_format = cfg.data.format,
        norm = cfg.data.norm,
        split='train',
        split_root = cfg.data.split,
        augments=data_aug,
        )

    v_loader = data_loader(
        data_path,
        data_format = cfg.data.format,
        split='val',
        split_root = cfg.data.split,
        )

    train_data_len = len(t_loader)
    logger.info(f'num of train samples: {train_data_len} \nnum of val samples: {len(v_loader)}')

    batch_size = cfg.train.batch_size
    epoch = cfg.train.epoch
    train_iter = int(np.ceil(train_data_len / batch_size) * epoch)
    logger.info(f'total train iter: {train_iter}')

    trainloader = data.DataLoader(t_loader,
                                  batch_size=batch_size, 
                                  num_workers=cfg.train.n_workers, 
                                  shuffle=True,
                                  persistent_workers=True,
                                  drop_last=True)

    valloader = data.DataLoader(v_loader, 
                                batch_size=cfg.test.batch_size, 
                                # persis
                                num_workers=cfg.train.n_workers,)

    # Setup Model
    device = f'cuda:{cfg.train.gpu[0]}'
    model = get_model(cfg.model).to(device)
    input_size = (cfg.model.in_channels, 512, 512)
    logger.info(f"Using Model: {cfg.model.arch}")
    logger.info(f'model summary: {summary(model, input_size=(input_size, input_size), is_complex=False)}')
    model = torch.nn.DataParallel(model, device_ids=cfg.gpu)      #自动多卡运行，这个好用
    
    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in vars(cfg.train.optimizer).items()
                        if k not in ('name', 'wrap')}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))
    if hasattr(cfg.train.optimizer, 'wrap') and  cfg.train.optimizer.wrap=='lars':
        optimizer = LARS(optimizer=optimizer)
        logger.info(f'warp optimizer with {cfg.train.optimizer.wrap}')
    scheduler = get_scheduler(optimizer, cfg.train.lr)
    loss_fn = get_loss_function(cfg)
    logger.info(f"Using loss ,{str(cfg.train.loss)}")

    if cfg.train.clip:
        logger.info(f'max grad norm: {cfg.train.clip}')

    # load checkpoints
    val_cls_1_acc = 0
    best_cls_1_acc_now = 0
    best_cls_1_acc_iter_now = 0
    val_macro_OA = 0
    best_macro_OA_now = 0
    best_macro_OA_iter_now = 0
    start_iter = 0
    if cfg.train.resume is not None:
        if os.path.isfile(cfg.train.resume):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg.train.resume)
            )

            # load model state
            checkpoint = torch.load(cfg.train.resume)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            # best_cls_1_acc_now = checkpoint["best_cls_1_acc_now"]
            # best_cls_1_acc_iter_now = checkpoint["best_cls_1_acc_iter_now"]

            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg.train.resume, checkpoint["epoch"]
                )
            )

            # copy tensorboard files
            resume_src_dir = osp.split(cfg.train.resume)[0]
            # shutil.copytree(resume_src_dir, writer.get_logdir())
            for file in os.listdir(resume_src_dir):
                if not ('.log' in file or '.yml' in file or '_last_model' in file):
                # if 'events.out.tfevents' in file:
                    resume_dst_dir = writer.get_logdir()
                    fu.copy(osp.join(resume_src_dir, file), resume_dst_dir, )

        else:
            logger.info("No checkpoint found at '{}'".format(cfg.train.resume))

    # Setup Metrics
    running_metrics_val = runningScore(2)
    runing_metrics_train = runningScore(2)
    val_loss_meter = averageMeter()
    train_time_meter = averageMeter()
    train_loss_meter = averageMeter()

    # train
    it = start_iter
    train_start_time = time.time() 
    train_val_start_time = time.time()
    model.train()   
    while it < train_iter:
        for noisy in trainloader:
            it += 1   
            noisy = noisy.to(device)
            mask1, mask2 = generate_mask_pair(noisy)
            noisy_sub1 = generate_subimages(noisy, mask1)
            noisy_sub2 = generate_subimages(noisy, mask2)

            # preparing for the regularization term
            with torch.no_grad():
                noisy_denoised = model(noisy)
            noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
            noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

            # calculating the loss 
            noisy_output = model(noisy_sub1)
            noisy_target = noisy_sub2
            if cfg.train.loss.gamma.const:
                gamma = cfg.train.loss.gamma.base
            else:
                gamma = it / train_iter * cfg.train.loss.gamma.base

            diff = noisy_output - noisy_target
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
            loss1 = torch.mean(diff**2)
            loss2 = gamma * torch.mean((diff - exp_diff)**2)
            loss_all = loss1 + loss2
            loss_all.backward()

            # In PyTorch 1.1.0 and later, you should call `optimizer.step()` before `lr_scheduler.step()`
            optimizer.step()
            scheduler.step()
            
            # record the loss of the minibatch
            train_loss_meter.update(loss_all)
            train_time_meter.update(time.time() - train_start_time)

            if cfg.data.synthetic:
                pass

            if it % cfg.train.print_interval == 0:
                terminal_info = f"Iter [{it:d}/{train_iter:d}]  \
                                train Loss: {train_loss_meter.avg:.4f}  \
                                Time/Image: {train_time_meter.avg / cfg.train.batch_size:.4f}"  

                logger.info(terminal_info)
                writer.add_scalar('loss/train_loss', train_loss_meter.avg, it)
                
                if cfg.data.synthetic:
                    pass

                runing_metrics_train.reset()
                train_time_meter.reset()
                train_loss_meter.reset()

            if it % cfg.train.val_interval == 0 or \
               it == train_iter:
                val_start_time = time.time()
                model.eval()            # change behavior like drop out
                with torch.no_grad():   # disable autograd, save memory usage
                    for noisy in valloader:      
                        noisy = noisy.to(device)
                        noisy_denoised = model(noisy)
                        
                        if cfg.data.synthetic:
                            pass
                        else:
                            val_loss = torch.mean((noisy_denoised-noisy)**2)

                        val_loss_meter.update(val_loss)

                writer.add_scalar('loss/val_loss', val_loss_meter.avg, it)
                logger.info(f"Iter [{it}/{train_iter}], val Loss: {val_loss_meter.avg:.4f} Time/Image: {(time.time()-val_start_time)/len(v_loader):.4f}")
                
                if cfg.data.synthetic:
                    pass

                val_loss_meter.reset()
                running_metrics_val.reset()

                if it % (train_iter/cfg.train.epoch/10) == 0:
                    ep = int(it / ((train_iter/cfg.train.epoch)))
                    state = {
                        "epoch": it,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                    }
                    save_path = osp.join(writer.file_writer.get_logdir(), f"{ep}.pkl")
                    torch.save(state, save_path)


                train_val_time = time.time() - train_val_start_time
                remain_time = train_val_time * (train_iter-it) / it
                m, s = divmod(remain_time, 60)
                h, m = divmod(m, 60)
                if s != 0:
                    train_time = "Remain train time = %d hours %d minutes %d seconds \n" % (h, m, s)
                else:
                    train_time = "Remain train time : train completed.\n"
                logger.info(train_time)
                model.train()

            train_start_time = time.time() 


if __name__ == "__main__":
    cfg = args.get_argparser('configs/hoekman.yml')
    del cfg.test
    
    # choose deterministic algorithms, and disable benchmark for variable size input
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False      
    # Setup random seeds to a determinated value for reproduction
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    np.random.default_rng(seed)
    
    # generate work dir
    run_id = osp.join(r'runs', cfg.model.arch + '_' + cfg.train.loss.name + '_' + cfg.train.optimizer.name+ '_' + cfg.train.epoch)
    run_id = utils.get_work_dir(run_id)
    writer = SummaryWriter(log_dir=run_id)
    config_fig = types.dict2fig(cfg.to_flatten_dict())
    writer.add_figure('config', config_fig, close=True)
    writer.flush()

    # logger
    logger = get_logger(run_id)

    # print('-'*100)
    logger.info(f'RUNDIR: {run_id}')
    logger.info(f'using config file: {cfg.config_file}')
    shutil.copy(cfg.config_file, run_id)

    train(cfg, writer, logger)
    logger.info(f'RUNDIR:{run_id}')
