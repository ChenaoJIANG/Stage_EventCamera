import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .normalization import *



#############


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes, kernel_size=4):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class MaskNet6(nn.Module):

    def __init__(self, nb_ref_imgs=4, output_exp=True):
        super(MaskNet6, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp
        print('nb ref imgs',nb_ref_imgs)
        conv_planes = [16, 32, 64, 128, 256, 256]#, 256, 256]
        self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)  # p = 3
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        #self.conv7 = conv(conv_planes[5], conv_planes[6])
        #self.conv8 = conv(conv_planes[6], conv_planes[7])

        #self.pose_pred = nn.Conv2d(conv_planes[7], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

        if self.output_exp:
            upconv_planes = [256, 256, 128, 64, 32, 16]
            self.deconv6 = upconv(conv_planes[5], upconv_planes[0],kernel_size=4)   # 256 256
            self.deconv5 = upconv(upconv_planes[0]+conv_planes[4], upconv_planes[1])  #256+256  256
            self.deconv4 = upconv(upconv_planes[1]+conv_planes[3], upconv_planes[2])  #256+128  125
            self.deconv3 = upconv(upconv_planes[2]+conv_planes[2], upconv_planes[3])  #128+64  64 
            self.deconv2 = upconv(upconv_planes[3]+conv_planes[1], upconv_planes[4])  #64+32  32
            self.deconv1 = upconv(upconv_planes[4]+conv_planes[0], upconv_planes[5])  #32+16  16

            self.pred_mask6 = nn.Conv2d(upconv_planes[0], self.nb_ref_imgs, kernel_size=3, padding=1)  # 256
            self.pred_mask5 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.pred_mask4 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.pred_mask3 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.pred_mask2 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.pred_mask1 = nn.Conv2d(upconv_planes[5], self.nb_ref_imgs, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def init_mask_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        for module in [self.pred_mask1, self.pred_mask2, self.pred_mask3, self.pred_mask4, self.pred_mask5, self.pred_mask6]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.xavier_uniform(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

        # for mod in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, self.pose_pred]:
        #     for fparams in mod.parameters():
        #         fparams.requires_grad = False


    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)
        input = F.pad(input,(0,0,32,0))
        
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)

        print('input size', input.size())
        print(out_conv5.size(), out_conv4.size(), out_conv3.size(), out_conv2.size(), out_conv1.size())
        print(out_conv6.size())
        #out_conv7 = self.conv7(out_conv6)
        #out_conv8 = self.conv8(out_conv7)

        #pose = self.pose_pred(out_conv8)
        #pose = pose.mean(3).mean(2)
        #pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        if self.output_exp:
            out_upconv6 = self.deconv6(out_conv6  )#[:, :, 0:out_conv5.size(2), 0:out_conv5.size(3)]
            print('out_conv6', out_conv6.size())
            print('out_upconv6', out_upconv6.size(), 'out_conv5 ', out_conv5.size())

            out_upconv5 = self.deconv5(torch.cat((out_upconv6, out_conv5), 1))#[:, :, 0:out_conv4.size(2), 0:out_conv4.size(3)]
            out_upconv4 = self.deconv4(torch.cat((out_upconv5, out_conv4), 1))#[:, :, 0:out_conv3.size(2), 0:out_conv3.size(3)]
            print('out_upconv5', out_upconv5.size(), 'out_conv4 ', out_conv4.size())
            out_upconv3 = self.deconv3(torch.cat((out_upconv4, out_conv3), 1))#[:, :, 0:out_conv2.size(2), 0:out_conv2.size(3)]
            print('out_upconv4', out_upconv4.size(), 'out_conv3 ', out_conv3.size())
            out_upconv2 = self.deconv2(torch.cat((out_upconv3, out_conv2), 1))#[:, :, 0:out_conv1.size(2), 0:out_conv1.size(3)]
            print('out_upconv3', out_upconv3.size(), 'out_conv2 ', out_conv2.size())
            out_upconv1 = self.deconv1(torch.cat((out_upconv2, out_conv1), 1))#[:, :, 0:input.size(2), 0:input.size(3)]
            print('out_upconv2', out_upconv2.size(), 'out_conv1 ', out_conv1.size())
            print('out_upconv1', out_upconv1.size())
            exp_mask6 = nn.functional.sigmoid(self.pred_mask6(out_upconv6))
            exp_mask5 = nn.functional.sigmoid(self.pred_mask5(out_upconv5))
            exp_mask4 = nn.functional.sigmoid(self.pred_mask4(out_upconv4))
            exp_mask3 = nn.functional.sigmoid(self.pred_mask3(out_upconv3))
            exp_mask2 = nn.functional.sigmoid(self.pred_mask2(out_upconv2))
            exp_mask1 = nn.functional.sigmoid(self.pred_mask1(out_upconv1))
        else:
            exp_mask6 = None
            exp_mask5 = None
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        if self.training:
            return exp_mask1, exp_mask2, exp_mask3, exp_mask4, exp_mask5, exp_mask6
        else:
            return exp_mask1



############""










class SingleConvBlock(nn.Module):
    # basic convolution block v1
    def __init__(self, in_planes, out_planes, kernel_size, padding, norm_type='gn', norm_group=16):
        super(SingleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1s = None
        if norm_type == 'bn':
            self.bn1s = nn.BatchNorm2d(in_planes,momentum=0.1)
        if norm_type == 'in':
            self.bn1s = nn.InstanceNorm2d(in_planes, affine=True)
        if norm_type == 'gn':
            self.bn1s = GroupNorm(in_planes, num_groups=norm_group)
        if norm_type == 'fd':
            self.bn1s = FeatureDecorr(in_planes, num_groups=norm_group)

    def forward(self, x):
        if self.bn1s is not None:
            x = self.bn1s(x)
        out = self.conv1(F.relu(x))
        return out



class DoubleConvBlock(nn.Module):
    # basic convolution block v2
    def __init__(self, in_planes, out_planes, kernel_size, padding, norm_type='gn', norm_group=16):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False)

        self.bn1s = None
        self.bn2s = None
        if norm_type == 'bn':
            self.bn1s = nn.BatchNorm2d(in_planes,momentum=0.1)
            self.bn2s = nn.BatchNorm2d(out_planes, momentum=0.1)
        if norm_type == 'in':
            self.bn1s = nn.InstanceNorm2d(in_planes, affine=True)
            self.bn2s = nn.InstanceNorm2d(out_planes, affine=True)
        if norm_type == 'gn':
            self.bn1s = GroupNorm(in_planes, num_groups=norm_group)
            self.bn2s = GroupNorm(out_planes, num_groups=norm_group)
        if norm_type == 'fd':
            self.bn1s = FeatureDecorr(in_planes, num_groups=norm_group)
            self.bn2s = FeatureDecorr(out_planes, num_groups=norm_group)

    def forward(self, x):
        if self.bn1s is not None:
            x = self.bn1s(x)
        out = self.conv1(F.relu(x))
        
        if self.bn2s is not None:
            out = self.bn2s(out)
        out = self.conv2(F.relu(out))
        #print('out doubleconv',np.shape(out))
        return out






def scaling(maps, scaling_factor=None, output_size=None):
    N, C, H, W = maps.shape

    if scaling_factor is not None:
        # target map size
        H_t = math.floor(H * scaling_factor)
        W_t = math.floor(W * scaling_factor)
        pool_size = int(math.floor(1. / scaling_factor))
        min_pool_size = pool_size

    if output_size is not None:
        _, _, H_t, W_t = output_size
        scaling_factor = [H_t / H, W_t / W]
        pool_size = [int(math.floor(1. / scaling_factor[0])), int(math.floor(1. / scaling_factor[1]))]
        min_pool_size = min(pool_size)

    if min_pool_size >= 2:
        maps = F.avg_pool2d(maps, pool_size, ceil_mode=True)
        N, C, H, W = maps.shape

    if H != H_t or W != W_t:
        maps = F.adaptive_avg_pool2d(maps,(H_t,W_t))
    return maps


def cascade(out, x):   # 这个函数的目的是实现对不同通道数的张量进行级联操作，以便在深度学习模型中进行特征融合或特征连接的应用。
    if out.shape[1] > x.shape[1]:
        channels_in = x.shape[1]
        out = torch.cat([out[:, :channels_in] + x, out[:, channels_in:]], dim=1)
    elif out.shape[1] == x.shape[1]:
        out = out + x
    elif out.shape[1] < x.shape[1]:
        channels_out = out.shape[1]
        out = x[:, :channels_out] + out

    return out


class CascadeLayer(nn.Module):
    """
    This function continuously samples the feature maps
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, scale_factor=None, dropout_rate=0. ,norm_type='gn', norm_group=16,n_iter=2):
        super(CascadeLayer, self).__init__()
        self.scale_factor = scale_factor
        self.ConvBlock = DoubleConvBlock(in_planes, out_planes, kernel_size, int((kernel_size - 1) / 2),
                                         norm_type=norm_type)
        #self.ConvBlock=RecurrentConvBlockC(in_planes, out_planes, kernel_size, int((kernel_size - 1) / 2),dropout_rate=dropout_rate,norm_type=norm_type,n_iter=n_iter)
    def forward(self, x, output_size=None):

        out = self.ConvBlock(x)
        out = cascade(out, x)
        if self.scale_factor is not None:  # 0.5
            out = scaling(out, self.scale_factor)
        if output_size is not None:
            out = scaling(out, output_size)

       #print('cascade out', np.shape(out))
        return out


class InvertedCascadeLayer(nn.Module):
    def __init__(self, in_planes, in_planes2, out_planes, kernel_size=3, padding=1, dropout_rate=0.,norm_type='gn', norm_group=16,n_iter=1):
        super(InvertedCascadeLayer, self).__init__()

        self.ConvBlock1 = SingleConvBlock(in_planes, out_planes, kernel_size, padding, norm_type=norm_type)
        self.ConvBlock2 = SingleConvBlock(in_planes2 + out_planes, out_planes, kernel_size, padding,
                                          norm_type=norm_type)
        #self.ConvBlock1 = RecurrentConvBlockC(in_planes, out_planes, kernel_size, int((kernel_size - 1) / 2),dropout_rate=dropout_rate,norm_type=norm_type,n_iter=n_iter)
        #self.ConvBlock2 = RecurrentConvBlockC(in_planes2 + out_planes, out_planes, kernel_size, int((kernel_size - 1) / 2),dropout_rate=dropout_rate,norm_type=norm_type,n_iter=n_iter)

    def forward(self, x, x2): 
        x = scaling(x, output_size=x2.shape)
        out = self.ConvBlock1(x)
        out = cascade(out, x)
        out = self.ConvBlock2(torch.cat([out, x2], dim=1))
        maps = cascade(out, x)
        return maps


class ECN_Disp(nn.Module):
    def __init__(self, input_size, in_planes=3, init_planes=32, scale_factor=0.5, growth_rate=32, final_map_size=1,
                 alpha=10, beta=0.01, norm_type='gn'):  # init planes = n_channel
        super(ECN_Disp, self).__init__()
        self.scale_factor = scale_factor
        self.final_map_size = final_map_size
        self.encoding_layers = nn.ModuleList()  #  创建一个空的nn.ModuleList()对象，用于存储编码器层的模块。
        self.decoding_layers = nn.ModuleList()
        self.pred_planes = 1    # 最终的预测通道数
        self.alpha = alpha
        self.beta = beta

        out_planes = init_planes  # 32
        output_size = input_size  # 260
        self.conv1 = nn.Conv2d(in_planes, init_planes, kernel_size=3, padding=1, stride=2, bias=False)  # 创建一个2D卷积层,用于作为模型的第一层编码器。in channel=3; out==32
        output_size = math.floor(output_size / 2)  # 130
        while math.floor(output_size * scale_factor) >= final_map_size:  # 1. 130*0.5=65 ; final=8
            
            new_out_planes = out_planes + growth_rate    # 1. 32+32=64 
            if len(self.encoding_layers) == 0:
                kernel_size = 3
            else:
                kernel_size = 3
            self.encoding_layers.append(    #根据指定的参数创建CascadeLayer模块，并将其添加到self.encoding_layers列表中
                CascadeLayer(in_planes=out_planes, out_planes=new_out_planes, kernel_size=kernel_size,
                             scale_factor=scale_factor, norm_type=norm_type))
            output_size = math.floor(output_size * scale_factor)
            out_planes = out_planes + growth_rate
            print('encode layer: ', out_planes)

        print(len(self.encoding_layers), ' encoding layers.') # 4
        print(out_planes, ' encoded feature maps.')  # 64, 96, 128, 160

        self.predict_maps = nn.ModuleList()

        in_planes2 = out_planes  # encoder planes 160

        planes = []
        for i in range(len(self.encoding_layers)+1):  # 0-4
        #for i in range(len(self.encoding_layers)):
            if i == len(self.encoding_layers):
                in_planes2 = in_planes  # 3,  last layer
                new_out_planes = max(in_planes, self.pred_planes)    #max(3, 1) pred_planes = 1 最终的预测通道数
            else:
                in_planes2 = in_planes2 - growth_rate  # -32
                new_out_planes = max(out_planes - growth_rate, self.pred_planes)
            self.decoding_layers.append(
                InvertedCascadeLayer(in_planes=out_planes, in_planes2=in_planes2, out_planes=new_out_planes,norm_type=norm_type))
            out_planes = new_out_planes
            planes.append(out_planes)

        print(len(self.decoding_layers), ' decoding layers.')
        print(planes, ' decoded feature maps.')  #[128,96,64,32,3]
   
        planes.reverse()  # 将列表planes中的元素顺序反转 [3,32,64,96,128]

        self.predicts=len(planes)#4
        self.predict_maps = nn.ModuleList()  # 创建一个空的nn.ModuleList()对象，用于存储预测图层的模块。
        for i in range(self.predicts):
            if False:
                self.predict_maps.append(SingleConvBlock(planes[i], 1, kernel_size=3, padding=1, norm_type=norm_type))
            else:
                self.predict_maps.append(
                nn.Sequential(SingleConvBlock(planes[i], 1, kernel_size=3, padding=1, norm_type=norm_type),
                              nn.BatchNorm2d(1,affine=True,momentum=0.1))    # 创建预测图层的模块，并将其添加到self.predict_maps列表中。预测图层通常由一个(SingleConvBlock)或卷积块加上批归一化(nn.BatchNorm2d)组成。
            )
    def init_weights(self):    # 对神经网络模型中各个层的权重（参数）进行初始化的函数

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)  # 使用 Xavier 均匀分布初始化该卷积层的权重。Xavier 均匀分布是一种常用的权重初始化方法，用于在较好地保持信息流动的同时避免梯度消失或梯度爆炸。
                # nn.init.kaiming_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)  #  将实例归一化层的权重初始化为1
                nn.init.constant_(m.bias, 0)   #  将实例归一化层的偏置项初始化为零
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):   # 如果当前模块是线性层（全连接层，
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):

        b, _, h, w = input.shape   # b, 3, 480, 640
        encode = [input]
        encode.append(self.conv1(encode[-1]))   # conv1(encode[-1]: b, 32, 240, 320)

        for i, layer in enumerate(self.encoding_layers):
            #print(i, np.shape(layer(encode[-1])))  # 0. b,64,120,160; 1. b,96,60,80; 2. b,128,30,40; 3. b,160,15,20
            encode.append(layer(encode[-1]))
        #print('len encode', len(encode))   # 6  [input, conv1, 4 encode layers]
        decode = [encode[-1]]   # b,160,15,20
        predicts = []

        for i, layer in enumerate(self.decoding_layers): # 5
            out = layer(decode[-1], encode[-2 - i])  # layer=InvertedCasxadeLayer(x, x2.shape)  b,160,15,20 ==> b,128,30,40
            #print(i, np.shape(out))  # 0. b,128,30,40; 1. b,96,60,80; 2. b,64,120,160; 3. b,32,240,320; 4. b,3,480,640
            decode.append(out)

            j = len(self.decoding_layers) - i   # 5-i
            if j <= self.predicts:  # j<= 5
                pred = self.predict_maps[j - 1](decode[-1])  # decode[-1]: b,3,480,640
                #print(i,'p',pred.min().item(),pred.mean().item(),pred.max().item())

                predicts.append(pred)

                if len(predicts) > 1:
                    predicts[-1] = predicts[-1] + scaling(predicts[-2],output_size=predicts[-1].shape)  # residual learning
                #decode[-1] = torch.cat([decode[-1][:, :self.pred_planes] + predicts[-1], decode[-1][:, self.pred_planes:]],dim=1)  # residual learning
                decode[-1] = torch.cat([decode[-1][:, :self.pred_planes] + predicts[-1], decode[-1][:, self.pred_planes:]],dim=1)  # residual learning

        predicts.reverse()
        # print(len(predicts)) 5

        #for i in range(self.predicts):
        #    print(i,predicts[i].min().item(),predicts[i].mean().item(),predicts[i].max().item())

        disp_predicts = [self.alpha * torch.sigmoid(predicts[i]) + self.beta for i in range(self.predicts)]

        if self.training:
            return disp_predicts
        else:
            return disp_predicts[0]



class ECN_PixelPose(nn.Module):
    def __init__(self, input_size, nb_ref_imgs=2, in_planes=3,init_planes=16, scale_factor=0.5, growth_rate=16,
                 final_map_size=1, output_exp=True,
                 norm_type='gn'):
        super(ECN_PixelPose, self).__init__()
        self.scale_factor = scale_factor
        self.final_map_size = final_map_size
        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()

        self.nb_ref_imgs = nb_ref_imgs
        self.pred_planes = 1+6

        in_planes = (1 + nb_ref_imgs) *in_planes

        out_planes = init_planes
        output_size = input_size
        self.conv1 = nn.Conv2d(in_planes, init_planes, kernel_size=3, padding=1, stride=2, bias=False)
        output_size = math.floor(output_size / 2)
        while math.floor(output_size * scale_factor) >= final_map_size:
            new_out_planes = out_planes + growth_rate
            if len(self.encoding_layers) == 0:
                kernel_size = 3
            else:
                kernel_size = 3
            self.encoding_layers.append(
                CascadeLayer(in_planes=out_planes, out_planes=new_out_planes, kernel_size=kernel_size,
                             scale_factor=scale_factor,norm_type=norm_type))
            output_size = math.floor(output_size * scale_factor)
            out_planes = out_planes + growth_rate

        print(len(self.encoding_layers), ' encoding layers.')
        print(out_planes, ' encoded feature maps.')
        


        self.pose_pred = SingleConvBlock(out_planes, 6 , kernel_size=1, padding=0,
                                         norm_type=norm_type)

        in_planes2 = out_planes  # encoder planes
        self.predicts = 50
        planes = []
        for i in range(len(self.encoding_layers) + 1):
            if i == len(self.encoding_layers):
                in_planes2 = in_planes  # encoder planes
                new_out_planes = max(in_planes, self.pred_planes)
            else:
                in_planes2 = in_planes2 - growth_rate  # encoder planes
                new_out_planes = max(out_planes - growth_rate, self.pred_planes)
            self.decoding_layers.append(
                InvertedCascadeLayer(in_planes=out_planes, in_planes2=in_planes2, out_planes=new_out_planes,norm_type=norm_type))
            out_planes = new_out_planes
            planes.append(out_planes)

        print(len(self.decoding_layers), ' decoding layers.')
        print(out_planes, ' decoded feature maps.')
        
        planes.reverse()

        self.predicts=len(planes)
        self.predict_maps = nn.ModuleList()
        for i in range(self.predicts):
            if False:
                self.predict_maps.append(SingleConvBlock(planes[i], self.pred_planes, kernel_size=3, padding=1, norm_type=norm_type))
            else:
                self.predict_maps.append(
                nn.Sequential(SingleConvBlock(planes[i], self.pred_planes, kernel_size=3, padding=1, norm_type=norm_type),
                              nn.BatchNorm2d(self.pred_planes,affine=True,momentum=0.1))
            )
    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_image, ref_imgs):
        assert (len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)

        b, _, h, w = input.shape
        encode = [input]
        encode.append(self.conv1(encode[-1]))

        for i, layer in enumerate(self.encoding_layers):
            encode.append(layer(encode[-1]))

        out = encode[-1]

        pose = self.pose_pred(out)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), 1, 6)

        decode = [out]
        predicts = []

        for i, layer in enumerate(self.decoding_layers):
            out = layer(decode[-1], encode[-2 - i])
            decode.append(out)

            j = len(self.decoding_layers) - i
            if j <= self.predicts:
                pred = self.predict_maps[j - 1](decode[-1])
                predicts.append(pred)
                if len(predicts) > 1:
                    predicts[-1] = predicts[-1] + scaling(predicts[-2],output_size=predicts[-1].shape)  # residual learning
                #decode[-1] = torch.cat([decode[-1][:, :self.pred_planes], decode[-1][:, self.pred_planes:]],dim=1)  # residual learning
                decode[-1] = torch.cat([decode[-1][:, :self.pred_planes] + predicts[-1], decode[-1][:, self.pred_planes:]],dim=1)  # residual learning

        predicts.reverse()
        #print('self ', self.predicts)
        #print('predicts = ',predicts[0])
        #for i in range(self.predicts):
        #    print(i,predicts[i].min().item(),predicts[i].mean().item(),predicts[i].max().item())

        exps = [torch.sigmoid(predicts[i][:, :1]) for i in range(self.predicts)]
        # for i in range(self.predicts):
        #     exps[i][exps[i]>=0.3] = 1.0
        #     exps[i][exps[i]<0.3] = 0.0
        
        #print('exps0',torch.max(exps[0]))
        res_pose = [0.001 * predicts[i][:, 1:] for i in range(self.predicts)]

        for i in range(self.predicts):
            res_pose[i][:,3:]=0.

        pixel_pose=[pose.view(-1,6,1,1)+res_pose[i] for i in range(self.predicts)]

        final_pose=[pose.view(-1,6,1,1)*(1-exps[i])+exps[i]*pixel_pose[i] for i in range(self.predicts)]
        if self.training:
            return exps, pose,pixel_pose,final_pose
        else:
            return exps[0],pose,pixel_pose[0],final_pose[0]



class ECN_Pose(nn.Module):
    def __init__(self, input_size, nb_ref_imgs=2, in_planes=3,init_planes=16, scale_factor=0.5, growth_rate=16,
                 final_map_size=1, n_motions=5,
                 norm_type='gn'):
        super(ECN_Pose, self).__init__()
        self.scale_factor = scale_factor
        self.final_map_size = final_map_size
        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()

        self.nb_ref_imgs = nb_ref_imgs
        self.pred_planes = n_motions
        self.n_motions = n_motions

        in_planes = (1 + nb_ref_imgs) *in_planes

        out_planes = init_planes
        output_size = input_size
        self.conv1 = nn.Conv2d(in_planes, init_planes, kernel_size=3, padding=1, stride=2, bias=False)
        output_size = math.floor(output_size / 2)
        while math.floor(output_size * scale_factor) >= final_map_size:
            new_out_planes = out_planes + growth_rate
            if len(self.encoding_layers) == 0:
                kernel_size = 3
            else:
                kernel_size = 3
            self.encoding_layers.append(
                CascadeLayer(in_planes=out_planes, out_planes=new_out_planes, kernel_size=kernel_size,
                             scale_factor=scale_factor,norm_type=norm_type))
            output_size = math.floor(output_size * scale_factor)
            out_planes = out_planes + growth_rate

        print(len(self.encoding_layers), ' encoding layers.')
        print(out_planes, ' encoded feature maps.')

        self.pose_pred = SingleConvBlock(out_planes, 6*n_motions, kernel_size=1, padding=0,
                                         norm_type=norm_type)

        in_planes2 = out_planes  # encoder planes
        self.predicts = 50
        planes = []
        for i in range(len(self.encoding_layers) + 1):
        #for i in range(len(self.encoding_layers)):
            if i == len(self.encoding_layers):
                in_planes2 = in_planes  # encoder planes
                new_out_planes = max(in_planes, self.pred_planes)
            else:
                in_planes2 = in_planes2 - growth_rate  # encoder planes
                new_out_planes = max(out_planes - growth_rate, self.pred_planes)
            self.decoding_layers.append(
                InvertedCascadeLayer(in_planes=out_planes, in_planes2=in_planes2, out_planes=new_out_planes,norm_type=norm_type))
            out_planes = new_out_planes
            planes.append(out_planes)

        print(len(self.decoding_layers), ' decoding layers.')
        print(out_planes, ' decoded feature maps.')
        planes.reverse()

        self.predicts=len(planes)
        self.predict_maps = nn.ModuleList()
        for i in range(self.predicts):
            if False:
                self.predict_maps.append(SingleConvBlock(planes[i], self.pred_planes, kernel_size=3, padding=1, norm_type=norm_type))
            else:
                self.predict_maps.append(
                nn.Sequential(SingleConvBlock(planes[i], self.pred_planes, kernel_size=3, padding=1, norm_type=norm_type),
                              nn.BatchNorm2d(self.pred_planes,affine=True,momentum=0.1))
            )
    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_image, ref_imgs):
        assert (len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)

        b, _, h, w = input.shape
        encode = [input]
        encode.append(self.conv1(encode[-1]))

        for i, layer in enumerate(self.encoding_layers):
            encode.append(layer(encode[-1]))

        out = encode[-1]

        pose = self.pose_pred(out)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.n_motions,1,1,6)
        if self.n_motions>1:
            pose[:, 1:,:,:,3:]=0.
            pose=torch.cat((pose[:,:1],pose[:,:1]+pose[:,1:]),dim=1)

        decode = [out]
        predicts = []

        for i, layer in enumerate(self.decoding_layers):
            out = layer(decode[-1], encode[-2 - i])
            decode.append(out)

            j = len(self.decoding_layers) - i
            if j <= self.predicts:
                pred = self.predict_maps[j - 1](decode[-1])
                predicts.append(pred)
                if len(predicts) > 1:
                    predicts[-1] = predicts[-1] + scaling(predicts[-2],output_size=predicts[-1].shape)  # residual learning
                #decode[-1] = torch.cat([decode[-1][:, :self.pred_planes], decode[-1][:, self.pred_planes:]],dim=1)  # residual learning
                decode[-1] = torch.cat([decode[-1][:, :self.pred_planes] + predicts[-1], decode[-1][:, self.pred_planes:]],dim=1)  # residual learning

        predicts.reverse()
        #print('predicts = ',np.shape(predicts[0]))   # 5* (1,4,480,640)
        #print(np.max(predicts[0]))
        # for i in range(self.predicts):
        #     print(predicts[i].min().item(),predicts[i].max().item())
        #print(predicts)
        exps = [F.softmax(predicts[i],dim=1) for i in range(self.predicts)]
        # for i in range(self.predicts):
        #     print(exps[i].min().item(),exps[i].max().item())
        #exps =  [torch.sigmoid(predicts[i][:, :self.nb_ref_imgs]) for i in range(self.predicts)]
        #print(np.shape(exps[0].cpu().data.numpy()))
        #print(exps1[0].size())


        final_pose=[torch.sum(pose*exps[i].unsqueeze(4),dim=1).permute(0,3,1,2) for i in range(self.predicts)]
        ego_pose=pose[:,0].view(-1,1,6)
        if self.training:
            return exps, ego_pose,final_pose
        else:
            return exps[0],ego_pose,final_pose[0]