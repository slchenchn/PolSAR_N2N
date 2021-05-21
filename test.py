'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-04-12
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

from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.loader import get_loader 
from ptsemseg.models import get_model

import args
import utils
from mylib import types



def test(cfg, logger, run_id):
    # augmemtations 
    augments = cfg.test.augments
    data_aug = get_composed_augmentations(augments)

    # dataloader
    data_loader = get_loader(cfg.data.dataloader)
    data_loader = data_loader(root=cfg.data.path, data_format=cfg.data.format, augments=cfg.test.augments, split=cfg.test.dataset)
    os.mkdir(osp.join(run_id, cfg.test.dataset))
    
    logger.info(f'data path: {cfg.data.path}')
    logger.info(f'num of {cfg.test.dataset} set samples: {len(data_loader)}')

    loader = data.DataLoader(data_loader,
                                  batch_size=cfg.test.batch_size, 
                                  num_workers=cfg.test.n_workers, 
                                  shuffle=False,
                                  persistent_workers=True,
                                  drop_last=False)

    # model
    model = get_model(cfg.model, n_classes=2)
    logger.info(f'using model: {cfg.model.arch}')
    device = f'cuda:{cfg.gpu[0]}'
    model=model.to(device)
    model = torch.nn.DataParallel(model, device_ids=cfg.gpu)

    # load model params
    if osp.isfile(cfg.test.pth):
        logger.info("Loading model from checkpoint '{}'".format(cfg.test.pth))

        # load model state
        checkpoint = torch.load(cfg.test.pth)
        model.load_state_dict(checkpoint["model_state"])
        # best_cls_1_acc_now = checkpoint["best_cls_1_acc_now"]
        # best_cls_1_acc_iter_now = checkpoint["best_cls_1_acc_iter_now"]
    else:
        raise FileNotFoundError(f'{cfg.test.pth} file not found')

    # Setup Metrics
    running_metrics_val = runningScore(2)
    running_metrics_train = runningScore(2)
    metrics = runningScore(2)

    # test
    model.eval()
    img_cnt = 0
    with torch.no_grad():
        for (file_a, file_b, label, mask) in loader:
            file_a = file_a.to(device)            
            file_b = file_b.to(device)   
            label  = label.numpy()
            mask = mask.numpy()

            outputs = model(file_a, file_b)
            pred = outputs.max(1)[1].cpu().numpy()
            confusion_matrix_now = metrics.update(label, pred, mask)

            for idx, cm in enumerate(confusion_matrix_now):
                cm *=100
                pred_filename = osp.join(run_id, cfg.test.dataset, f'{img_cnt}_{cm[0, 0]:.2f}_{cm[1, 1]:.2f}_pred.png')
                gt_filename = osp.join(run_id, cfg.test.dataset, f'{img_cnt}_{cm[0, 0]:.2f}_{cm[1, 1]:.2f}_gt.png')
                img_cnt += 1

                if cv2.imwrite(pred_filename, (pred[idx, :, :]*255).astype(np.uint8)) and cv2.imwrite(gt_filename, (label[idx, :, :]*255).astype(np.uint8)):
                    logger.info(f'writed {pred_filename}')
                else:
                    logger.info(f'fail to writed {pred_filename}')

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
    cfg = args.get_argparser('configs/psr_siamdiff_pauli.yml')
    del cfg.train
    torch.backends.cudnn.benchmark = True

    run_id = utils.get_work_dir(osp.join(cfg.test.out_path, osp.split(osp.split(cfg.test.pth)[0])[1]))
    # writer = SummaryWriter(log_dir=run_id)
    # config_fig = types.dict2fig(cfg.to_flatten_dict())
    # writer.add_figure('config', config_fig, close=True)
    # writer.flush()
    shutil.copy(cfg.config_file, run_id)

    # logger
    logger = get_logger(run_id)
    logger.info(f'RUN DIR: {run_id}')

    test(cfg, logger, run_id)
    logger.info(f'RUN DIR: {run_id}')