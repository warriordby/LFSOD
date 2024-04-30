import tqdm
from SS_utils.transforms import pad_image_to_shape
import numpy as np
import cv2
import torch
from timm.models.layers import to_2tuple
from SS_utils.metric import hist_info, compute_score
from SS_utils.visualize import print_iou
import yaml
from easydict import EasyDict as edict
with open('Config/yaml.yaml', 'r') as f:
    config = edict(yaml.safe_load(f))
   
def val_func_process_rgbX(List_Img, device=None):
    
    #List_Img[0] = np.ascontiguousarray(List_Img[0][None, :, :, :], dtype=np.float32)
    #List_Img[0] = torch.FloatTensor(List_Img[0]).cuda(device)

    List_Img = [torch.tensor(np.ascontiguousarray(item[None, :, :, :], dtype=np.float32)).cuda(device) for item in List_Img]

    with torch.cuda.device(List_Img[0].get_device()):
        val_func.eval()
        val_func.to(List_Img[0].get_device())

        with torch.no_grad():
            score = val_func(List_Img)
            score = score[0]
            if config.is_flip:                
                for k in range(len(List_Img)):
                    List_Img[k] = List_Img[k].flip(-1)
                score_flip = self.val_func(List_Img[0],List_Img[1])
                score_flip = score_flip[0]
                score += score_flip.flip(-1)
            score = torch.exp(score)
    
    return score


def scale_process_rgbX(self,List_Img,ori_shape, crop_size, stride_rate, device=None):
    new_rows, new_cols, c = List_Img[0].shape
    long_size = new_cols if new_cols > new_rows else new_rows

    if new_cols <= crop_size[1] or new_rows <= crop_size[0]:
        List_Img,margin = self.process_image_rgbX(List_Img, crop_size)


        score = val_func_process_rgbX(List_Img,device)
        score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]
    else:
        stride = (int(np.ceil(crop_size[0] * stride_rate)), int(np.ceil(crop_size[1] * stride_rate)))
        List_Img = [pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0) for img in List_Img]

        
        pad_rows = List_Img[0].shape[0]
        pad_cols = List_Img[0].shape[1]

        r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride[0])) + 1
        c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride[1])) + 1
        data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(device)

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



                List_Img, tmargin = self.process_image_rgbX(List_Img, crop_size)

                temp_score = val_func_process_rgbX(List_Img, device)
                
                temp_score = temp_score[:, tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                        tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                data_scale[:, s_y: e_y, s_x: e_x] += temp_score
        score = data_scale
        score = score[:, margin[0]:(score.shape[1] - margin[1]),
                margin[2]:(score.shape[2] - margin[3])]

    score = score.permute(1, 2, 0)
    data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

    return data_output


def sliding_eval_rgbX(List_Img,config, crop_size, stride_rate, device=0):          
    ####rgb,x,11,22
    crop_size = to_2tuple(crop_size)
    ori_rows, ori_cols, _ = List_Img[0].shape
    List_Number = len(List_Img)

    processed_pred = np.zeros((ori_rows, ori_cols,config.num_classes))

    for s in config.eval_scale_array: 
        List_Img[0] = cv2.resize(List_Img[0], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        if len(List_Img[1].shape) == 2:
            List_Img[1] = cv2.resize(List_Img[1], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        else:
            List_Img[1] = cv2.resize(List_Img[1], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

        for k in range(List_Number-2):

            List_Img[2+k] = cv2.resize(List_Img[2+k], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        
        new_rows, new_cols, _ = List_Img[0].shape
        processed_pred += scale_process_rgbX(List_Img, (ori_rows, ori_cols),crop_size, stride_rate, device)

    pred = processed_pred.argmax(2)


def eval(dataset, epoch, checkpoints, optimizer, engine, config, device=0):
    ndata=dataset.get_length()
    all_results = []
    for idx in tqdm(range(ndata)):
        data = dataset[idx]
        List_Img = list(data.values())
        ####
        name = List_Img[-2]
        label = List_Img[1]
        del List_Img[1]
        del List_Img[-1]
        del List_Img[-1]    ####List里面包含rgb,model_x,view_img
        pred = sliding_eval_rgbX(List_Img, config.eval_crop_size, config.eval_stride_rate, device)
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
    result_line,tmp_m_Iou = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                            dataset.class_names, show_no_back=False)
    if max_Miou<=tmp_m_Iou:
        max_Miou=tmp_m_Iou
        best_epoch=epoch
    print("----------------------   best M_Iou:{:.3f}  best epoch{}".format(max_Miou, best_epoch))




# def single_process_evalutation(self):

#     logger.info('GPU %s handle %d data.' % (self.devices[0], self.ndata))
#     all_results = []
#     for idx in tqdm(range(self.ndata)):
#         dd = self.dataset[idx]
#         results_dict = self.func_per_iteration(dd,self.devices[0])
#         all_results.append(results_dict)
#     result_line = self.compute_metric(all_results)

#     return result_line 
# results.write(result_line)



# def compute_metric(self, results):

#     hist = np.zeros((config.num_classes, config.num_classes))
#     correct = 0
#     labeled = 0
#     count = 0
#     for d in results:
#         hist += d['hist']
#         correct += d['correct']
#         labeled += d['labeled']
#         count += 1

#     iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
#     result_line,tmp_m_Iou = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
#                             self.dataset.class_names, show_no_back=False)
#     if self.max_Miou<=tmp_m_Iou:
#         self.max_Miou=tmp_m_Iou
#         self.best_epoch=self.current_epoch
#     print("----------------------   best M_Iou:{:.3f}  best epoch{}".format(self.max_Miou, self.best_epoch))
#     return result_line


if __name__=='__main__':
    eval()