
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

from Config.opt import opt
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
# 使用示例


from Dataloader.SS_Dataset import RGBXDataset
from SS_utils.init_func import init_weight, group_weight,print_network
from SS_utils.lr_policy import WarmUpPolyLR
from SS_engine.engine import Engine

from SS_utils.pyt_utils import all_reduce_tensor, load_model
import SS_utils.eval_utils as eval_utils
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


def SS_train(engine, train_loader,opt, epoch):

    if opt.DDP:
        train_sampler.set_epoch(epoch)#数据采样器，分配数据
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    # pbar = tqdm(range(opt.niters_per_epoch), file=sys.stdout,
    #             bar_format=bar_format)
    dataloader = iter(train_loader)#加载器到迭代器
    sum_loss = 0
    # for idx in pbar:
    for idx, minibatch in enumerate(tqdm(train_loader, desc="Training")):
        engine.update_iteration(epoch, idx)#记录编号
        List_Img_Name = list(minibatch.keys())            
        List_Img = list(minibatch.values())
        for i in range(len(List_Img)-2):
            List_Img[i] = List_Img[i].cuda(non_blocking=True)
        Label = List_Img[1]     
        del List_Img[1]     
        del List_Img[-1]      
        del List_Img[-1]                   
        aux_rate = 0.2    
        loss = model(List_Img,Label)
        # reduce the whole loss over multi-gpu
        if opt.DDP:
            reduce_loss = all_reduce_tensor(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_idx = (epoch- 1) * opt.niters_per_epoch + idx 
        lr = engine.lr_policy.get_lr(current_idx)
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
        if opt.DDP:
            sum_loss += reduce_loss.item()
            print_str = 'Epoch {}/{}'.format(epoch, opt.epoch) \
                    + ' Iter {}/{}:'.format(idx + 1, opt.niters_per_epoch) \
                    + ' lr=%.4e' % lr \
                    + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
        else:
            sum_loss += loss
            print_str = 'Epoch {}/{}'.format(epoch, opt.epoch) \
                    + ' Iter {}/{}:'.format(idx + 1, opt.niters_per_epoch) \
                    + ' lr=%.4e' % lr \
                    + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
        # print(print_str, end='\r')  # 使用 \r 实现覆盖式打印，即每次打印都会覆盖上一行内容
        del loss

def train(train_loader, model, optimizer, epoch, save_path, task, engine, opt):
    #global step
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model.train()
    try:
        if task=='SS':
                SS_train(engine, train_loader, opt, epoch)
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


def SS_test(test_loader, model, epoch, save_path, opt):
    global max_Miou, best_epoch

    tmp_m_Iou, result=eval_utils.eval(test_loader, model, opt)
    if max_Miou<=tmp_m_Iou:
        max_Miou=tmp_m_Iou
        best_epoch=epoch
        # torch.save(model.state_dict(), save_path + 'best.pth')
    logging.info(result)
    print("----------------------   best M_Iou:{:.3f}  best epoch{}".format(max_Miou, best_epoch))


def test(test_loader, model, epoch, save_path,optimizer,task, opt):

    model.eval()#评估模式
    with torch.no_grad():
        if task=='SS':
            SS_test(test_loader, model, epoch, save_path, opt)
        else:
            SOD_test(test_loader, model, epoch, save_path, optimizer)


if __name__ == '__main__':
####################也不管
    if torch.cuda.is_available():
        # 获取CUDA版本
        cuda_version = torch.version.cuda
        print("CUDA 版本:", cuda_version)
    else:
        print("CUDA 不可用")
    if opt.task=='SS':
        from models.builder import EncoderDecoder as segmodel

        opt.log_dir=opt.save_path
        engine=Engine(None)
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=opt.background)
    else:
        from Builder.SOD_Builder import model
        engine=None
    end_epoch=opt.epoch

    logging.info("Start %s train..."%opt.task)      
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
        model=segmodel(criterion=criterion, norm_layer=BatchNorm2d)
        params_list=[]

        if opt.optimizer == 'AdamW':
            params_list= group_weight(params_list, model, BatchNorm2d, opt.lr)
            optimizer = torch.optim.AdamW(params_list, lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
        elif opt.optimizer == 'SGDM':
            optimizer = torch.optim.SGD(params_list, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
        else:
            raise NotImplementedError
        total_iteration = opt.epoch * opt.niters_per_epoch
        engine.lr_policy = WarmUpPolyLR(opt.lr, opt.lr_power, total_iteration, opt.niters_per_epoch * opt.warm_up_epoch)

    else:
        model = model() 
        params = model.parameters()
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)


    
#############先不管
    if opt.resume == False:  
        start_epoch = 0
        if opt.task=='SOD':
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
        train_dataset = RGBXDataset(opt, "train")
        test_loader =RGBXDataset(opt, "val")
        # # train_sampler = DistributedSampler(train_dataset)
        # train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True)#, sampler=train_sampler)
        if engine.continue_state_object:
            engine.restore_checkpoint()
    else : 
        train_dataset = SalObjDataset(opt.rgb_root, opt.gt_root, opt.fs_root,trainsize=opt.train_size)  
        test_loader = test_dataset(opt.test_rgb_root, opt.test_gt_root, opt.test_fs_root,testsize=opt.train_size)

    if opt.DDP == True:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)
    else:
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    
    if opt.task=='SS':
        engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    Iter = len(train_loader)#批次数量
    
    

    #保存和初始化日志项路径
    if (not opt.DDP or dist.get_rank()== 0):
        if (opt.task =='SOD') :
            setup_logging(opt.save_path, model_params, opt)
    best_mae = 1
    saver = ModelSaver()
    logger = Logger()
    max_Miou=0
    best_epoch = 0

    
    
    #if opt.resume == False:
    for epoch in range(start_epoch, end_epoch):
        if opt.task=='SOD':       
            cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch) #指数衰减      
        train(train_loader, model, optimizer, epoch, opt.checkpoints, opt.task, engine, opt)
        test(test_loader, model, epoch, opt.checkpoints, optimizer, opt.task,  opt)
   



    #else:
    #    print('Start Resume')
    #    checkpoint = torch.load(opt.load_resume)
    #    model.load_state_dict(checkpoint['model_state_dict'])
    #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #    start_epoch = checkpoint['epoch']
    #    print('Resume:  start_epoch{}'.format(start_epoch))
    #    for param in optimizer.param_groups:
    #        print('当前学习率',param['lr'])





































