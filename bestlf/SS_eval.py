import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from Config.SS_config1 import config
from SS_utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from SS_utils.visualize import print_iou, show_img
from SS_engine.evaluator import Evaluator
from SS_engine.logger import get_logger
from SS_utils.metric import hist_info, compute_score
from Dataloader.SS_Dataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from Dataloader.SS_dataloader import ValPre,get_test_loader
import tqdm
logger = get_logger()
import SS_utils.eval_utils as eval_utils 
class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        List_Img = list(data.values())
        ####
        name = List_Img[-2]
        label = List_Img[1]
        del List_Img[1]
        del List_Img[-1]
        del List_Img[-1]    ####List里面包含rgb,model_x,view_img
        pred = self.sliding_eval_rgbX(List_Img, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')

            fn = name + '.png'

            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path+'_color', fn))

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):

        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line,tmp_m_Iou = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                self.dataset.class_names, show_no_back=False)
        if self.max_Miou<=tmp_m_Iou:
            self.max_Miou=tmp_m_Iou
            self.best_epoch=self.current_epoch
        print("----------------------   best M_Iou:{:.3f}  best epoch{}".format(self.max_Miou, self.best_epoch))
        return result_line


# def eval(self,model, dataset , model_path,  log_file, devices):
#     ndata=dataset.get_length
#     model = [os.path.join(model_path, 'epoch-last.pth' % self.epoch), ]

    
#     results = open(log_file, 'a')#追加模式打开
#     logger.info("Load Model: %s" % model)
#     self.val_func = load_model(self.network, model)
#     all_results = []
#     for idx in tqdm(range(ndata)):
#         data = self.dataset[idx]
#         results_dict = func_per_iteration(data,devices[0])
#         all_results.append(results_dict)
#     result_line = compute_metric(all_results)
#     return result_line     


#     results.write('Model: ' + model + '\n')
#     results.write(result_line)
#     results.write('\n')
#     results.flush()
#     results.write("----------------------   best M_Iou:{:.3f}".format(self.max_Miou))
#     results.close()
    
class Eval_per_iter(Evaluator):
    def __init__(self, dataset, network, args, config, all_dev=0):
        self.all_dev = all_dev
        self.network = network
        self.dataset = dataset
        self.config=config
        self.args=args

    def test(self, epochs):
        config=self.config
        args=self.args
        with torch.no_grad():
            segmentor = SegEvaluator(
                self.dataset, 
                config.num_classes, 
                config.norm_mean,
                config.norm_std, 
                self.network,
                config.eval_scale_array, 
                config.eval_flip,
                self.all_dev, 
                args.verbose, 
                args.save_path,
                args.show_image
            )
            segmentor.run(
                config.checkpoint_dir, 
                epochs, 
                config.val_log_file,
                config.link_val_log_file
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-e', '--epochs', default='500', type=str)
    parser.add_argument('-e', '--epochs', default='0-10', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names,
                   'view_path':config.rgb_view_path,
                   'View_path_format':config.rgb_view_format,
                   'view_list':config.view_list}
    val_pre = ValPre()#直接返回图像列表
    dataset = RGBXDataset(data_setting, 'val', val_pre)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
 
        segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file, config.checkpoint_step)
        # my_try.eval(dataset, args.epochs, network, config)
    # dataset=get_test_loader(RGBXDataset,config)
    # Eval=Eval_per_iter(dataset, segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d), args, config)
    # Eval.test(args.epochs)
