'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-05-28
'''
import os.path as osp
import matplotlib.pyplot as plt
import shutil
import cv2
import numpy as np
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
import piq

from mylib import polSAR_utils as psr

from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.loader import get_loader 
from ptsemseg.models import get_model

from utils import args
from utils import utils
from mylib import types


__TMP_DIR = r'./tmp'


def test(cfg, logger, run_id):
    # Setup Augmentations
    augmentations = cfg.test.augments
    logger.info(f'using augments: {augmentations}')
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg.data.dataloader)
    data_path = cfg.data.path
    data_loader = data_loader(
        data_path, 
        data_format=cfg.data.format, 
        norm = cfg.data.norm,
        split=cfg.test.dataset,
        split_root = cfg.data.split,
        log = cfg.data.log,
        augments=data_aug,
        logger=logger,
        ENL = cfg.data.ENL,
        )
    os.mkdir(osp.join(run_id, cfg.test.dataset))
    
    logger.info("data path: {}".format(data_path))
    logger.info(f'num of {cfg.test.dataset} set samples: {len(data_loader)}')

    loader = data.DataLoader(data_loader,
                            batch_size=cfg.test.batch_size, 
                            num_workers=cfg.test.n_workers, 
                            shuffle=False,
                            persistent_workers=True,
                            drop_last=False,
                            )

    # Setup Model
    device = f'cuda:{cfg.gpu[0]}'
    model = get_model(cfg.model).to(device)
    input_size = (cfg.model.in_channels, 512, 512)
    logger.info(f'using model: {cfg.model.arch}')
    
    model = torch.nn.DataParallel(model, device_ids=cfg.gpu)

    # load model params
    if osp.isfile(cfg.test.pth):
        logger.info("Loading model from checkpoint '{}'".format(cfg.test.pth))

        # load model state
        checkpoint = torch.load(cfg.test.pth)
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise FileNotFoundError(f'{cfg.test.pth} file not found')

    # Setup Metrics
    running_metrics_val = runningScore(2)
    running_metrics_train = runningScore(2)
    metrics = runningScore(2)
    test_psnr_meter = averageMeter()
    test_ssim_meter = averageMeter()

    # test
    model.eval()
    img_cnt = 0
    data_range = 255
    if cfg.data.log:
        data_range = np.log(data_range)
    with torch.no_grad():
        for clean, noisy, files_path in loader:
             
            noisy = noisy.to(device, dtype=torch.float32)
            noisy_denoised = model(noisy)
    
            # metrics
            if cfg.data.simulate:
                clean = clean.to(device, dtype=torch.float32)
                psnr = piq.psnr(clean, noisy_denoised, data_range=data_range)
                ssim = piq.ssim(clean, noisy_denoised, data_range=data_range)
                test_psnr_meter.update(psnr)
                test_ssim_meter.update(ssim)

            if cfg.data.log:
                noisy 

                
            # save images
            for ii in range(cfg.test.batch_size):
                file_path = files_path[ii][14:]
                    
                file_ori = psr.inverse_Hokeman_decomposition(noisy[ii, ...])
                file_denoise = psr.inverse_Hokeman_decomposition(noisy_denoised[ii, ...])
                pauli_ori = (psr.rgb_by_c3(file_ori)*255).astype(np.uint8)
                pauli_denoise = psr.rgb_by_c3(file_denoise).astype(np.uint8)
                cv2.imwrite(osp.join(run_id, file_path+'-ori.png'), pauli_ori)
                cv2.imwrite(osp.join(run_id, file_path+'-denoise.png'), pauli_denoise)

        score,_ = metrics.get_scores()
        # score_train,_ = running_metrics_train.get_scores()
        # score_val,_ = running_metrics_val.get_scores()
        acc = score['Acc']
        # acc_train = score_train['Acc']
        # acc_val = score_val['Acc']
        logger.info(f'acc : {acc}\tOA:{acc.mean()}')
        micro_OA = score['Overall_Acc']
        miou = score['Mean_IoU']
        logger.info(f'overall acc: {micro_OA}, mean iou: {miou}')
        # logger.info(f'acc of train set: {acc_train} \nacc of val set: {acc_val}')



if __name__=='__main__':
    cfg = args.get_argparser('configs/hoekman_simulate.yml')

    # choose deterministic algorithms, and disable benchmark for variable size input
    utils.set_random_seed(0)

    run_id = utils.get_work_dir(osp.join(cfg.test.out_path, osp.split(osp.split(cfg.test.pth)[0])[1]))
    
    shutil.copy(cfg.config_file, run_id)

    # logger
    logger = get_logger(run_id)
    logger.info(f'RUN DIR: {run_id}')

    test(cfg, logger, run_id)
    logger.info(f'RUN DIR: {run_id}')