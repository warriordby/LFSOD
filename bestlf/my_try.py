from tqdm import tqdm
import numpy as np
import cv2
import torch
from timm.models.layers import to_2tuple
from SS_utils.transforms import pad_image_to_shape, normalize
from SS_utils.metric import hist_info, compute_score
from SS_utils.visualize import print_iou
import yaml
from easydict import EasyDict as edict
from SS_utils.pyt_utils import load_model
with open('Config/yaml.yaml', 'r') as f:
    config = edict(yaml.safe_load(f))


def process_image_rgbX( List_Img,crop_size=None):
    Number_Img = len(List_Img)
    if List_Img[0].shape[2] < 3:
        im_b = List_Img[0]
        im_g = List_Img[0]
        im_r = List_Img[0]
        List_Img[0] = np.concatenate((im_b, im_g, im_r), amodal_xis=2)
        
    List_Img[0] = normalize(List_Img[0], config.norm_mean, config.norm_std)
    
    for k in range(Number_Img-2):  
        List_Img[k+2] = normalize(List_Img[k+2], config.norm_mean, config.norm_std)

    #print('问题出发点5 List_Img[0]',List_Img[0].shape)
    
    if len(List_Img[1].shape) == 2:
        List_Img[1] = normalize(List_Img[1], 0, 1)
    else:
        List_Img[1] = normalize(List_Img[1], config.norm_mean, config.norm_std)


    if crop_size is not None:
        List_Img[0], margin = pad_image_to_shape(List_Img[0], crop_size, cv2.BORDER_CONSTANT, value=0)
        List_Img[0] = List_Img[0].transpose(2, 0, 1)
        
        for k in range(Number_Img-2):
            List_Img[k+2], _ = pad_image_to_shape(List_Img[k+2], crop_size, cv2.BORDER_CONSTANT, value=0)
            List_Img[k+2] = List_Img[k+2].transpose(2, 0, 1)
        
        List_Img[1], _ = pad_image_to_shape(List_Img[1], crop_size, cv2.BORDER_CONSTANT, value=0)
    
        if len(List_Img[1].shape) == 2:
            List_Img[1] = List_Img[1][np.newaxis, ...]
        else:
            List_Img[1] = List_Img[1].transpose(2, 0, 1) # 3 H W

        

        return List_Img, margin

    
    for k in range(Number_Img-2):
        List_Img[k+2] = List_Img[k+2].transpose(2, 0, 1)
    

    List_Img[0] = List_Img[0].transpose(2, 0, 1) # 3 H W


    if len(List_Img[1].shape) == 2:
        List_Img[1] = List_Img[1][np.newaxis, ...]
    else:
        List_Img[1] = List_Img[1].transpose(2, 0, 1) # 3 H W
    return List_Img, margin



def val_func_process_rgbX(List_Img, val_func, device=None):
    List_Img = [torch.tensor(np.ascontiguousarray(item[None, :, :, :], dtype=np.float32)).cuda(device) for item in List_Img]

    score = val_func(List_Img)
    score = score[0]
    if config.eval_flip:
        for k in range(len(List_Img)):
            List_Img[k] = List_Img[k].flip(-1)
        score_flip = val_func(List_Img[0], List_Img[1])
        score_flip = score_flip[0]
        score += score_flip.flip(-1)
    score = torch.exp(score)

    return score

def scale_process_rgbX( List_Img, ori_shape, crop_size, stride_rate, val_func ,device=None):
    new_rows, new_cols, _ = List_Img[0].shape
    long_size = new_cols if new_cols > new_rows else new_rows

    if new_cols <= crop_size[1] or new_rows <= crop_size[0]:
        List_Img,margin = process_image_rgbX(List_Img, crop_size)

        score = val_func_process_rgbX(List_Img, val_func,device)
        score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]
    else:
        stride = (int(np.ceil(crop_size[0] * stride_rate)), int(np.ceil(crop_size[1] * stride_rate)))
        List_Img = [pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0) for img in List_Img]

        pad_rows = List_Img[0].shape[0]
        pad_cols = List_Img[0].shape[1]

        r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride[0])) + 1
        c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride[1])) + 1
        data_scale = torch.zeros(config.num_classes, pad_rows, pad_cols).cuda(device)

        for grid_yidx in range(r_grid):
            for grid_xidx in range(c_grid):
                s_x = grid_xidx * stride[0]
                s_y = grid_yidx * stride[1]
                e_x = min(s_x + crop_size[0], pad_cols)
                e_y = min(s_y + crop_size[1], pad_rows)
                s_x = e_x - crop_size[0]
                s_y = e_y - crop_size[1]
                List_Img[0] = List_Img[0][s_y:e_y, s_x: e_x, :]

                for k in range(len(List_Img)-2):
                    List_Img[k+2] = List_Img[k+2][s_y:e_y, s_x: e_x, :]

                if len(List_Img[1].shape) == 2:
                    List_Img[1] = List_Img[1][s_y:e_y, s_x: e_x]
                else:
                    List_Img[1] = List_Img[1][s_y:e_y, s_x: e_x,:] 

                List_Img, tmargin = process_image_rgbX(List_Img, crop_size)

                temp_score = val_func_process_rgbX(List_Img, val_func, device)

                temp_score = temp_score[:, tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                        tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                data_scale[:, s_y: e_y, s_x: e_x] += temp_score
        score = data_scale
        score = score[:, margin[0]:(score.shape[1] - margin[1]),
                margin[2]:(score.shape[2] - margin[3])]

    score = score.permute(1, 2, 0)
    data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

    return data_output

def sliding_eval_rgbX(List_Img, config, crop_size, stride_rate, val_func, device=0 ):
    crop_size = to_2tuple(crop_size)
    ori_rows, ori_cols, _ = List_Img[0].shape
    List_Number = len(List_Img)

    processed_pred = np.zeros((ori_rows, ori_cols, config.num_classes))

    for s in config.eval_scale_array: 
        List_Img[0] = cv2.resize(List_Img[0], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        if len(List_Img[1].shape) == 2:
            List_Img[1] = cv2.resize(List_Img[1], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        else:
            List_Img[1] = cv2.resize(List_Img[1], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

        for k in range(List_Number-2):
            List_Img[2+k] = cv2.resize(List_Img[2+k], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        
        new_rows, new_cols, _ = List_Img[0].shape
        processed_pred += scale_process_rgbX(List_Img, (ori_rows, ori_cols), crop_size, stride_rate, device, val_func)

    pred = processed_pred.argmax(2)
    return pred

def eval(dataset, val_func , config, device=0):
    ndata=dataset.get_length()
    all_results = []

    for idx in tqdm(range(ndata)):
        data = dataset[idx]
        List_Img = list(data.values())

        name = List_Img[-2]
        label = List_Img[1]
        del List_Img[1]
        del List_Img[-1]
        del List_Img[-1]

        pred = sliding_eval_rgbX(List_Img, config, config.eval_crop_size, config.eval_stride_rate, device, val_func)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}
        all_results.append(results_dict)

    hist = np.zeros((config.num_classes, config.num_classes))
    correct = 0
    labeled = 0
    count = 0
    for d in all_results:
        hist += d['hist']
        correct += d['correct']
        labeled += d['labeled']
        count += 1

    iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
    result_line, tmp_m_Iou = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                            dataset.class_names, show_no_back=False)
    

    return tmp_m_Iou , result_line


