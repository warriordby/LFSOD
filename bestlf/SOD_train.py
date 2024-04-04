
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
from Builder.SOD_Builder import model
from Config.SOD_options import opt
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


from Dataloader.mixup import mixup_images as mi
from Dataloader.mixup import mixup_images2 as mi2

def train(train_loader, model, optimizer, epoch, save_path):
    #global step
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
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

            
    except KeyboardInterrupt:
        if not opt.DDP or dist.get_rank() == 0:
            saver.save_interrupt_checkpoint(epoch, model, save_path)
            
        raise


def test(test_loader, model, epoch, save_path,optimizer):
    global best_mae, best_epoch
    model.eval()#评估模式
    with torch.no_grad():
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



if __name__ == '__main__':
####################也不管
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
    best_epoch = 0
    saver = ModelSaver()
    logger = Logger()
    
    
    #if opt.resume == False:
    for epoch in range(start_epoch, opt.epoch+1):       
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)       
        train(train_loader, model, optimizer, epoch, opt.checkpoints)
        test(test_loader, model, epoch, opt.checkpoints,optimizer)
   






    #else:
    #    print('Start Resume')
    #    checkpoint = torch.load(opt.load_resume)
    #    model.load_state_dict(checkpoint['model_state_dict'])
    #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #    start_epoch = checkpoint['epoch']
    #    print('Resume:  start_epoch{}'.format(start_epoch))
    #    for param in optimizer.param_groups:
    #        print('当前学习率',param['lr'])





































