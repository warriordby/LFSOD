import torch
import torch.nn as nn
import torch.nn.functional as F
from Config.opt import opt, import_model_from_directory
from SS_utils.init_func import init_weight
from SS_utils.load_utils import load_pretrain
from functools import partial
from tsne import tsne_runner
from pca2 import pca_visual
from SS_engine.logger import get_logger
logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer

        self.backbone=import_model_from_directory(opt.model_name)(pretrained=opt.pretrained)

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
            print(out.shape)
            print(label.shape)
            pca_visual.pca(out, label)
            # tsne_runner.input2basket(out, label, 'syn')
            # tsne_runner.draw_tsne(['syn'], adding_name='visual/',plot_memory=True, clscolor=False)
            return loss

        return out
