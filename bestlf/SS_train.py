import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from Config.SS_config1 import config
from Dataloader.SS_dataloader import get_train_loader, get_test_loader
from models.builder import EncoderDecoder as segmodel
from Dataloader.SS_Dataset import RGBXDataset
from SS_utils.init_func import init_weight, group_weight,print_network
from SS_utils.lr_policy import WarmUpPolyLR
from SS_engine.engine import Engine
from SS_engine.logger import get_logger
from SS_utils.pyt_utils import all_reduce_tensor
from SS_eval import Eval_per_iter 
import subprocess

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default='last', type=str)
logger = get_logger()

os.environ['MASTER_PORT'] = '169710'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        print(engine.distribute)
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)
    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    # Eval_model=Eval_per_iter(args, config)

    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError
    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                            output_device=engine.local_rank, find_unused_parameters=True, broadcast_buffers=False)
    else:
        model_params = print_network(model, 'lf_pvt')#打印参数
        logger.info('params:',model_params)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()
    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')
    for epoch in range(engine.state.epoch, config.nepochs+1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)#数据采样器，分配数据
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)#加载器到迭代器
        sum_loss = 0
        for idx in pbar:
            engine.update_iteration(epoch, idx)
            minibatch = next(dataloader)
            ###['data', 'label', 'modal_x', 'view11', 'view22', 'fn', 'n']
            List_Img_Name = list(minibatch.keys())            
            List_Img = list(minibatch.values())
            for i in range(len(List_Img)-2):
                List_Img[i] = List_Img[i].cuda(non_blocking=True)
            Label = List_Img[1]     
            del List_Img[1]     
            del List_Img[-1]      
            del List_Img[-1]                   
            aux_rate = 0.2    
            print("List length:", len(List_Img))

            # # 遍历列表中的每个子列表，并输出其长度（即维度信息）
            # for i, sublist in enumerate(List_Img):
            #     print(f"Sublist {i+1} shape:", len(sublist)) 

            # tensor_img = torch.tensor(List_Img)
            loss = model(List_Img,Label)
            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr
            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
            del loss
            pbar.set_description(print_str, refresh=False)
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                current_epoch=engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                current_epoch=engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)

            



