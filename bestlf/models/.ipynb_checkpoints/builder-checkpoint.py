import torch
import torch.nn as nn
import torch.nn.functional as F
from Config.opt import opt, import_model_from_directory
from SS_utils.init_func import init_weight
from SS_utils.load_utils import load_pretrain
from functools import partial

from SS_engine.logger import get_logger
logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
#         # import backbone and decoder
#         if opt.backbone == 'swin_s':
#             logger.info('Using backbone: Swin-Transformer-small')
#             from .encoders.dual_swin import swin_s as backbone
#             self.channels = [96, 192, 384, 768]
#             self.backbone = backbone(norm_fuse=norm_layer)
#         elif opt.backbone == 'swin_b':
#             logger.info('Using backbone: Swin-Transformer-Base')
#             from .encoders.dual_swin import swin_b as backbone
#             self.channels = [128, 256, 512, 1024]
#             self.backbone = backbone(norm_fuse=norm_layer)
#         elif opt.backbone == 'mit_b5':
#             logger.info('Using backbone: Segformer-B5')
#             from .encoders.dual_segformer import mit_b5 as backbone
#             self.backbone = backbone(norm_fuse=norm_layer)
#         elif opt.backbone == 'mit_b4':
#             logger.info('Using backbone: Segformer-B4')
#             from .encoders.dual_segformer import mit_b4 as backbone
#             self.backbone = backbone(norm_fuse=norm_layer)
#         elif opt.backbone == 'mit_b2':
#             logger.info('Using backbone: Segformer-B2')
#             from .encoders.dual_segformer import mit_b2 as backbone
#             self.backbone = backbone(norm_fuse=norm_layer)
#         elif opt.backbone == 'mit_b1':
#             logger.info('Using backbone: Segformer-B1')
#             from .encoders.dual_segformer import mit_b0 as backbone
#             self.backbone = backbone(norm_fuse=norm_layer)
#         elif opt.backbone == 'mit_b0':
#             logger.info('Using backbone: Segformer-B0')
#             self.channels = [32, 64, 160, 256]
#             from .encoders.dual_segformer import mit_b0 as backbone
#             self.backbone = backbone(norm_fuse=norm_layer)
# #####################################################            
#         elif opt.backbone == 'agent_swin':
#             logger.info('Using backbone: agent-Swin-Transformer')
#             from .encoders.agent_swin import agent_swin_v2_b2 as backbone
#             self.backbone = backbone()
#         elif opt.backbone=='agent_pvt':
#             logger.info('Using backbone: agent-Pvt-Transformer')
#             from .encoders.agent_pvt import agent_pvt_v2_b2 as backbone
#             self.channels = [64, 128, 320, 512]
#             self.backbone = backbone()
#         elif opt.backbone=='pvt':
#             logger.info('Using backbone: Pvt-Transformer')
#             from bakebone.pvtv2 import pvt_v2_b2 as backbone
#             self.channels = [64, 128, 320, 512]
#             self.backbone = backbone()
#             # if opt.pretrained==True:
#             #     opt.pretrained=r'/root/autodl-tmp/LFSOD/bestlf/Pretain_weight/pvt_v2_b2.pth'
#         elif opt.backbone == 'segeformer_mit_b2':
#             logger.info('Using backbone: Segformer-B2')
#             from .encoders.segformer import mit_b2 as backbone
#             self.backbone = backbone()
#         elif opt.backbone == 'mscan_b':
#             logger.info('Using backbone: MSCAN')
#             from .encoders.mscan import mascan_b as backbone
#             self.backbone = backbone()
#         elif opt.backbone=='afformer_base':
#             logger.info(" Using backbone: afformer_base")
#             from .encoders.afformer import afformer_base as backbone
#             self.backbone = backbone(pretrained=opt.pretrained)
#             self.channels = [96, 176, 216, 216]
#         elif opt.backbone=='afformer_small':
#             logger.info(" Using backbone: afformer_small")
#             from .encoders.afformer import afformer_small as backbone
#             self.backbone = backbone()
#             self.channels = [64, 176, 216, 216]
#         elif opt.backbone=='afformer_tiny':
#             logger.info(" Using backbone: afformer_tiny")
#             from .encoders.afformer import afformer_tiny as backbone
#             self.backbone = backbone()
#             self.channels = [64, 160, 216, 216]
#         elif opt.backbone=='cgnet':
#             logger.info(" Using backbone: cgnet")
#             from .encoders.cgnet import CGNet as backbone
#             self.backbone = backbone()
#             self.channels = [35 , 131 , 256, 0]
#         elif opt.backbone=='convnext_tiny':
#             logger.info(" Using backbone: convnext_tiny")
#             from .encoders.convnext import convnext_tiny as backbone
#             self.backbone = backbone()
#             self.channels = [96, 192, 384, 768]
#         else:
#             logger.info('Using backbone: Segformer-B2')
#             from .encoders.dual_segformer import mit_b2 as backbone
#             self.backbone = backbone()
        self.backbone=import_model_from_directory(opt.model_name)(pretrained=opt.pretrained)
        # if opt.model_name=='inception_transformer':
        #     from .encoders.inception_transformer import iformer_small
        #     self.backbone= iformer_small()

        self.aux_head = None

        if opt.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=opt.num_classes, norm_layer=norm_layer, embed_dim=opt.decoder_embed_dim)
        
        elif opt.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=opt.num_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], opt.num_classes, norm_layer=norm_layer)
        
        elif opt.decoder == 'deeplabv3+':
            logger.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=opt.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], opt.num_classes, norm_layer=norm_layer)

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=opt.num_classes, norm_layer=norm_layer)

        self.criterion = criterion
        # self.pretrained = self.backbone.getatrributes(load_pretrain)
        if self.criterion:
            self.init_weights(opt, pretrained= opt.pretrained)

    def init_weights(self, opt, pretrained=None):
        if pretrained:
            logger.info('#################Loading pretrained model: {}'.format(pretrained))
            print('############################################# pretrianed True')
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, opt.bn_eps, opt.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, opt.bn_eps, opt.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, List_Img):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # assert len(List_Img) == 10,"List Wrong"
        orisize = List_Img[0].shape
        x ,_,_,_,_= self.backbone(List_Img)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out

    def forward(self,List_Img, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(List_Img)
        else:
            out= self.encode_decode(List_Img)
        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return loss
        return out
