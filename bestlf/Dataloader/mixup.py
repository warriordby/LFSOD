import numpy as np
import torch
import torchvision.transforms as transforms

def mixup_images(main_image, image_list, alpha=0.5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    对主要图像和图像列表中的每个图像进行 Mixup 操作。
    
    main_image: 主要图像 (张量)
    image_list: 包含要混合的图像的列表 (张量列表)
    alpha: Mixup 参数
    mean: 每个通道的均值
    std: 每个通道的标准差
    """
    # 创建标准化转换
    normalize = transforms.Normalize(mean=mean, std=std)

    # 对主图像进行标准化
    main_image = normalize(main_image)
    mixed_images = []
    for img in image_list:
        # 对列表中的图像进行标准化
        img = normalize(img)

        # 进行 Mixup
        lam = np.random.beta(alpha, alpha)
        mixed_img = lam * main_image + (1 - lam) * img
        mixed_images.append(mixed_img)

    return mixed_images


import torch

def mixup_images2(main_image, image_list, alpha=0.5):
    """
    以0.5的概率对 image_list 中的图像进行混合操作，
    同时以0.1的概率对 main_image 进行混合操作。
    返回混合后的 main_image 和 image_list。

    main_image: 主要图像 (在 CUDA 上)
    image_list: 包含要混合的图像的列表 (在 CUDA 上)
    alpha: Mixup 参数
    """
    #print('mixup_true')
    if torch.rand(1).item() < 0.1:
        # 随机选择一个图像与 main_image 混合
        img_to_mix = image_list[torch.randint(0, len(image_list), (1,)).item()]
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        main_image = lam * main_image + (1 - lam) * img_to_mix

    new_image_list = []
    for img in image_list:
        if torch.rand(1).item() < 0.5:
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
            mixed_img = lam * main_image + (1 - lam) * img
            new_image_list.append(mixed_img)
        else:
            new_image_list.append(img)

    return main_image, new_image_list

# 使用方法
# main_image, new_image_list = mixup_images_cuda(main_image, image_list)