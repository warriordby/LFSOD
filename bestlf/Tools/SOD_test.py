import torch
import torch.nn.functional as F
import sys


sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from Dataloader.SODdataloader import test_dataset
from model.LFTransNet import model
from torchvision.utils import save_image


print("GPU available:", torch.cuda.is_available())


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/root/autodl-tmp/test/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = model()
checkpoint = torch.load('/root/autodl-tmp/第二次/Code/lfsod_cpts/lfsod_epoch_best.pth')
if 'module' in list(checkpoint.keys())[0]:  # 如果模型在加载时包含了'module'前缀
    model.load_state_dict({k[7:]: v for k, v in checkpoint.items()})  # 去掉'module'前缀
else:
    model.load_state_dict(checkpoint)
#model.load_state_dict(torch.load('/root/autodl-tmp/LFDataset/Code/another/LFTransNet-main/lfsod_cpts/lfsod_0.pth'))

# test
model.cuda()
model.eval()


def CAM(features, img_path, save_path):
    features.retain_grad()

    grads = features.grad

    features = features.squeeze(0)

    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
    cv2.imwrite(save_path, superimposed_img)




test_datasets = ['LFSD'] #,'HFUT','LFSD','DUTLF-FS'
for dataset in test_datasets:
    save_path = './Result/test_maps/' + dataset + '/'
    save_path1 = './Result/test_maps1/' + dataset + '/'
    save_path2 = './Result/test_maps2/' + dataset + '/'
    save_path3 = './Result/test_maps3/' + dataset + '/'
    save_path4 = './Result/test_maps4/' + dataset + '/'
    save_path_sde = './Result/sde_maps/' + dataset + '/'
    save_path_decoder = './Result/decoder_maps/' + dataset + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path1):
        os.makedirs(save_path1)

    if not os.path.exists(save_path2):
        os.makedirs(save_path2)

    if not os.path.exists(save_path3):
        os.makedirs(save_path3)

    if not os.path.exists(save_path4):
        os.makedirs(save_path4)

    if not os.path.exists(save_path_sde):
        os.makedirs(save_path_sde)

    if not os.path.exists(save_path_decoder):
        os.makedirs(save_path_decoder)

    image_root = dataset_path + dataset + '/test_images/'
    gt_root = dataset_path + dataset + '/test_masks/'
    fs_root = dataset_path + dataset + '/test_focals/'
    test_loader = test_dataset(image_root, gt_root, fs_root, opt.testsize)
    for i in range(test_loader.size):
        #todo 位置
        image, focal, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        dim, height, width = focal.size()
        basize = 1
        focal = focal.view(1, basize, dim, height, width).transpose(0, 1)  # (basize, 1, 36, 256, 256)
        focal = torch.cat(torch.chunk(focal, chunks=12, dim=2), dim=1)  # (basize, 12, 3, 256, 256)
        focal = torch.cat(torch.chunk(focal, chunks=basize, dim=0), dim=1)  # (1, basize*12, 6, 256, 256)
        focal = focal.view(-1, *focal.shape[2:])  # [basize*12, 6, 256, 256)
        focal = focal.cuda()
        image = image.cuda()
        out = model(focal, image)
        print(out.shape)
        #res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = out
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path + name, res * 255)


        img_path = image_root+name
        img_path = img_path.split('.')[0]+'.jpg'

        #CAM(sde, img_path, save_path_sde + name)
        #CAM(fuse_sal, img_path, save_path_decoder + name)

    print('Test Done!')
