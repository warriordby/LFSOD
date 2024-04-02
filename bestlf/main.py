
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from Dataloader.SODdataloader import test_dataset, SalObjDataset

from Config.SOD_options import opt,parser
from SOD_Network_utils.utils import clip_gradient, adjust_lr

from SOD_Network_utils.train import (
    print_network,
    setup_logging,
    load_pretrained_weights,
    ModelSaver,
    Logger,
    preprocess_tensors,
    structure_loss,
    test_process_data,
    compute_mae,
    log_and_save
)
import torch.nn as nn
from Config.SS_config1 import config
from Dataloader.SS_dataloader import get_train_loader, get_test_loader

from Dataloader.SS_Dataset import RGBXDataset
from SS_utils.init_func import init_weight, group_weight,print_network
from SS_utils.lr_policy import WarmUpPolyLR
from SS_engine.engine import Engine
from SS_engine.logger import get_logger
from SS_utils.pyt_utils import all_reduce_tensor
from SS_engine.engine import Engine
from SS_eval import SegEvaluator
import sys
from tqdm import tqdm

from Dataloader.mixup import mixup_images as mi
from Dataloader.mixup import mixup_images2 as mi2

def SOD_train(optimizer, train_loader, save_path):
    loss_all = 0
    epoch_step = 0
    for i, (images, gts, focal) in enumerate(train_loader, start=1):            
        optimizer.zero_grad()
                    
        gts, gts1, gts2, gts3, gts4, focal, images,focal_stack = preprocess_tensors(gts, focal, images)
        images,focal_stack = mi2(images,focal_stack)
        out,out1,out2,out3,out4= model(images)
        # out= model(images)
        loss = structure_loss(out, gts)+structure_loss(out1, gts1)+structure_loss(out2, gts2)+structure_loss(out3, gts3)+structure_loss(out4, gts4)
        loss.backward()            
        clip_gradient(optimizer, opt.clip) # 梯度裁剪
        optimizer.step()
        epoch_step += 1  #这个Step是batch的意思
        loss_all += loss.data
        
        if not opt.DDP or dist.get_rank() == 0:
            logger.log_step(epoch, opt.epoch, i, Iter, loss.data, optimizer.state_dict()['param_groups'][0]['lr'])
    loss_all /= epoch_step   #每个batch的平均loss，batch loss
    if not opt.DDP or dist.get_rank() == 0:
        logger.log_epoch(epoch, opt.epoch, loss_all)
        if (epoch) % 5 == 0:
            saver.save_regular_checkpoint(epoch, model, save_path) 
    # 训练中断保留参数
        saver.save_resume_checkpoint(epoch, model, optimizer, save_path) if not opt.DDP or dist.get_rank() == 0 else None


def SS_train(engine, config, epoch):

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
        lr = engine.lr_policy.get_lr(current_idx)
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
    # if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
    #     tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
    if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
        if engine.distributed and (engine.local_rank == 0):
            current_epoch=engine.save_and_link_checkpoint(config.checkpoint_dir,
                                            config.log_dir,
                                            config.log_dir_link)
        elif not engine.distributed:
            current_epoch=engine.save_and_link_checkpoint(config.checkpoint_dir,
                                            config.log_dir,
                                            config.log_dir_link)


def train(train_loader, model, optimizer, epoch, save_path, task, engine, config):
    #global step
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model.train()
    try:
        if task=='SS':
            if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
                SS_train(engine, config, epoch)
        else:
            SOD_train(optimizer, train_loader, save_path)
  
    except KeyboardInterrupt:
        if not opt.DDP or dist.get_rank() == 0:
            saver.save_interrupt_checkpoint(epoch, model, save_path)
            
        raise


def SOD_test(test_loader, model, epoch, save_path,optimizer):
    global best_mae, best_epoch
    mae_sum = 0
    for i in range(test_loader.size):
        image, focal, gt, name = test_loader.load_data()
        gt, focal, image, focal_stack = test_process_data(gt, focal, image)

        #image torch.Size([1, 3, 256, 256])
        #focal torch.Size([12, 3, 256, 256])
        out,out1,out2,out3,out4 = model(image)
        # out= model(image)
        mae_sum += compute_mae(out, gt)

    mae = mae_sum / test_loader.size

    best_mae, best_epoch = log_and_save(epoch, mae, best_mae, best_epoch, model, save_path,optimizer)


def SS_test(test_loader, model, epoch, save_path, config):
    global segmentor
    segmentor.run(config.checkpoint_dir, epoch, config.val_log_file,
                    config.link_val_log_file, config.checkpoint_step)

def test(test_loader, model, epoch, save_path,optimizer,task,engine, config):

    model.eval()#评估模式
    with torch.no_grad():
        if task=='SS':
            SS_test(test_loader, model, epoch, save_path, config)
        else:
            SOD_test(test_loader, model, epoch, save_path,optimizer)


if __name__ == '__main__':
####################也不管
    if opt.task=='SS':
        from models.builder import EncoderDecoder as segmodel
        from Config.SS_config1 import config
        engine=Engine(parser)
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    else:
        engine=None
        from Builder.SOD_Builder import model
    logging.info("Start train...")      
    if opt.DDP == True: 
        local_rank = int(os.environ.get("LOCAL_RANK", 0))                         
        torch.cuda.set_device(local_rank)
        cudnn.benchmark = True
        dist.init_process_group(backend='nccl')
        print('opt.DDp',opt.DDP) if dist.get_rank() == 0 else None
        print("GPU available:", ",".join([str(i) for i in range(torch.cuda.device_count())])) if dist.get_rank() == 0 else None
    else:
        print('Single GPU')    
    #模型和优化器初始化
    if opt.task=='SS':
        if opt.DDP:
            BatchNorm2d = nn.SyncBatchNorm
        else:
            BatchNorm2d = nn.BatchNorm2d    
        model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
        params_list=[]
        train_loader, train_sampler = get_train_loader(engine, RGBXDataset)

        if config.optimizer == 'AdamW':
            params_list= group_weight(params_list, model, BatchNorm2d, config.lr)
            optimizer = torch.optim.AdamW(params_list, lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
        elif config.optimizer == 'SGDM':
            optimizer = torch.optim.SGD(params_list, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
        else:
            raise NotImplementedError
        total_iteration = config.nepochs * config.niters_per_epoch
        engine.lr_policy = WarmUpPolyLR(config.lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    else:
        model = model() 
        params = model.parameters()
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)


    
#############先不管
    if opt.resume == False:  
        start_epoch = 0
        load_pretrained_weights(model, opt)
          #weight_decay正则化系数
    else:
        print('Start Resume')
        checkpoint = torch.load(opt.load_resume)
        for key in checkpoint.keys():
            print(key)        
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  #weight_decay正则化系数
        start_epoch = checkpoint['epoch']
        print('Resume:  start_epoch{}'.format(start_epoch))

    model.cuda() 
    if not opt.DDP or dist.get_rank() == 0:
        model_params = print_network(model, opt.model_name)
    os.makedirs(opt.save_path, exist_ok=True)
    #模型并行
    if opt.DDP: model = DistributedDataParallel(model, find_unused_parameters=True)


    #数据加载
    if opt.DDP == True:
        print('load data...') if dist.get_rank() == 0 else None
    if opt.task=='SS':
        train_loader, train_sampler = get_train_loader(engine, RGBXDataset)
        test_loader =get_test_loader(RGBXDataset, config)
        engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
        if engine.continue_state_object:
            engine.restore_checkpoint()
        segmentor = SegEvaluator(test_loader, config.num_classes, config.norm_mean,
                        config.norm_std, model,
                        config.eval_scale_array, config.eval_flip,
                        devices=[0])
    else : 
        train_dataset = SalObjDataset(opt.rgb_root, opt.gt_root, opt.fs_root,trainsize=opt.trainsize)  

        if opt.DDP == True:
            train_sampler = DistributedSampler(train_dataset)
            train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt.batchsize, shuffle=False, pin_memory=True, sampler=train_sampler)
        else:
            train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt.batchsize, shuffle=True, pin_memory=True)
    
        test_loader = test_dataset(opt.test_rgb_root, opt.test_gt_root, opt.test_fs_root,testsize=opt.trainsize)
    Iter = len(train_loader)
    
    

    #保存和初始化日志项
    if not opt.DDP or dist.get_rank() == 0:
        setup_logging(opt.save_path, model_params, opt)
    best_mae = 1
    ax_M_iou=0
    best_epoch = 0
    saver = ModelSaver()
    logger = Logger()
    
    
    #if opt.resume == False:
    for epoch in range(start_epoch, opt.epoch+1):
        if opt.task!='SS':       
            cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch) #指数衰减      
        train(train_loader, model, optimizer, epoch, opt.checkpoints, opt.task, engine, config)
        test(test_loader, model, epoch, opt.checkpoints,optimizer, opt.task, engine, config)
   



    #else:
    #    print('Start Resume')
    #    checkpoint = torch.load(opt.load_resume)
    #    model.load_state_dict(checkpoint['model_state_dict'])
    #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #    start_epoch = checkpoint['epoch']
    #    print('Resume:  start_epoch{}'.format(start_epoch))
    #    for param in optimizer.param_groups:
    #        print('当前学习率',param['lr'])





































