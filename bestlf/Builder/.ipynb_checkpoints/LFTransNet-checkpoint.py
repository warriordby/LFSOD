import torch
import torch.nn as nn
from thop import profile, clever_format

from bakebone.pvtv2 import pvt_v2_b2
from transformer_decoder import transfmrerDecoder
import torch.nn.functional as F
from model.MultiScaleAttention import Block
from model.MLPDecoder import DecoderHead


#from fvcore.nn import FlopCountAnalysis, parameter_count_table


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        #focal
        self.focal_encoder = pvt_v2_b2()
#        self.focal_encoder = pvt_v2_b2()

        self.decoder = DecoderHead()

    def forward(self, x,x_stack, y):#focal  torch.Size([840, 3, 256, 256]) , image  torch.Size([70, 3, 256, 256])
                    #focal,focal_stack, images
        #focal
        ba = x.size()[0]//12


        y,out1,out2,out3,out4 = self.focal_encoder(x_stack,y)  #[64] [128] [320] [512]
        rgb_out = self.decoder(y)
        rgb_out = F.interpolate(rgb_out, size=(256, 256), mode='bilinear', align_corners=False)
        

        return rgb_out,out1,out2,out3,out4


if __name__ == '__main__':
    import torchvision
    from ptflops import get_model_complexity_info
    import time

    from torchstat import stat
    # path = "../config/hrt_base.yaml"
    a = torch.randn(24, 3, 256, 256).cuda()
    b = torch.randn(2, 3, 256, 256).cuda()
    # c = torch.randn(1, 1, 352, 352).cuda()
    # config = yaml.load(open(path, "r"),yaml.SafeLoader)['MODEL']['HRT']
    # hr_pth_path = r"E:\ScientificResearch\pre_params\hrt_base.pth"
    # cnn_pth_path = r"D:\tanyacheng\Experiments\pre_trained_params\swin_base_patch4_window7_224_22k.pth"
    # cnn_pth_path = r"E:\ScientificResearch\pre_params\resnet18-5c106cde.pth"
    model = model().cuda()
    # model.initialize_weights()

    # out = model(a, b)

    # stat(model, (b, a))
    # 分析FLOPs
    # flops = FlopCountAnalysis(model, (a, b))
    # print("FLOPs: ", flops.total())
    #
    # # 分析parameters
    # print(parameter_count_table(model))

    # -- coding: utf-8 --


    # model = torchvision.pvt_v2_b2().alexnet(pretrained=False)
    # flops, params = get_model_complexity_info(model, a, as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)

    # params, flops = profile(model, inputs=(b,))
    # params, flops = clever_format([params, flops], "%.2f")
    #
    # print(params, flops)
    # print(out.shape)
    # for x in out:
    #     print(x.shape)


    ###### FPS


    # nums = 710
    # time_s = time.time()
    # for i in range(nums):
    #     _ = model(a, b, c)
    # time_e = time.time()
    # fps = nums / (time_e - time_s)
    # print("FPS: %f" % fps)