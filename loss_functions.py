from __future__ import division
import torch
from torch import nn
from torch.autograd import Variable
from inverse_warp import inverse_warp

import torch.nn.functional as F


import numpy as np
from inverse_warp import *

class simple_photometric_reconstruction_loss(nn.Module):  # calculer flows, grids, ref_warped, diff(tgt - ref_warped)
    def __init__(self):
        super(simple_photometric_reconstruction_loss, self).__init__()

    def forward(self, tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose,ssim_w=0.,padding_mode='zeros'):
        def one_scale(depth,explainability_mask,pose):
            reconstruction_loss = 0
            b, _, h, w = depth.size()
            downscale = tgt_img.size(2)/h   # 1, 2, 4, 8, 16
            ego_flows_scaled=[]
            refs_warped_scaled = []
            grids=[]
            tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
            ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
            if pose.size(1)==1 or pose.size(1)==6:
                refs_warped_scaled, grids, ego_flows_scaled = multi_inverse_warp(ref_imgs_scaled, depth[:, 0], pose,
                                                                              intrinsics_scaled,
                                                                              intrinsics_scaled_inv, padding_mode)

            else:

                for i, ref_img in enumerate(ref_imgs_scaled):
                    if pose.size(1) == len(ref_imgs):
                        current_pose = pose[:, i]
                    elif pose.size(1)==len(ref_imgs)*6:
                        current_pose=pose[:,i*6:(i+1)*6]
                    ref_img_warped,grid,ego_flow = simple_inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, padding_mode)
                    refs_warped_scaled.append(ref_img_warped)
                    grids.append(grid)
                    ego_flows_scaled.append(ego_flow)

            for i in range(len(refs_warped_scaled)):
                #grid = grids[i]
                diff = (tgt_img_scaled - refs_warped_scaled[i])
                reconstruction_loss += diff.abs().view(b,-1).mean(1)
            return reconstruction_loss,refs_warped_scaled,ego_flows_scaled

        if type(explainability_mask) not in [tuple, list]:
            explainability_mask = [explainability_mask]
        if type(depth) not in [list, tuple]:
            depth = [depth]
        if type(pose) in [tuple, list]:
            assert len(pose)==len(depth)
        else:
            pose=[pose for i in range(len(depth))]
        loss = 0
        ego_flows=[]
        warped_refs=[]

        weight=0
        for d, mask, p in zip(depth, explainability_mask,pose):
            current_loss,refs_warped_scaled,ego_flows_scaled= one_scale(d, mask,p)
            _, _, h, w = d.size()
            weight+=h*w
            loss=loss+current_loss*h*w
            ego_flows.append(ego_flows_scaled)
            warped_refs.append(refs_warped_scaled)
        loss=loss/weight

        return loss,warped_refs,ego_flows



class sharpness_loss(nn.Module):
    def __init__(self):
        super(sharpness_loss, self).__init__()

    def forward(self, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose, padding_mode='zeros'):
        def one_scale(depth,explainability_mask,pose):

            sharpness_loss = 0
            b, _, h, w = depth.size()
            downscale = ref_imgs[0].size(2)/h
            ego_flows_scaled=[]
            ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

            stacked_im=0.

            ref_imgs_warped, grids, ego_flows_scaled = multi_inverse_warp(ref_imgs_scaled, depth[:, 0], pose, intrinsics_scaled,
                                                              intrinsics_scaled_inv, padding_mode)
            for i in range(len(ref_imgs)):
                ref_img=ref_imgs_scaled[i]
                ref_img_warped=ref_imgs_warped[i]
                new_grid=grids[i]
                in_bound = (new_grid[:,:,:,0]!=2).type_as(ref_img_warped).unsqueeze(1)
                #print(ref_img.min(),ref_img.mean(),ref_img.max(),ref_img_warped.min(),ref_img_warped.mean(),ref_img_warped.max())
                scaling = ref_img.view(b, 3, -1).mean(-1) / (1e-5 + ref_img_warped.view(b, 3, -1).mean(-1))
                #print(scaling.view(1,-1))
                stacked_im = stacked_im + ref_img_warped #* in_bound* scaling.view(b, 3, 1, 1)
            stacked_im=torch.pow(stacked_im.abs()+1e-4, .5)
            #if explainability_mask is not None:
            #    stacked_im = stacked_im * explainability_mask[:, 0:1]
            stacked_im=stacked_im[:,0]+stacked_im[:,2]#take the event channels
            sharpness_loss += stacked_im.view(b, -1).mean(1)

            return sharpness_loss,ref_imgs_warped,ego_flows_scaled

        if type(explainability_mask) not in [tuple, list]:
            explainability_mask = [explainability_mask]
        if type(depth) not in [list, tuple]:
            depth = [depth]
        if type(pose) in [tuple, list]:
            assert len(pose)==len(depth)
        else:
            pose=[pose for i in range(len(depth))]
        loss = 0
        ego_flows=[]
        warped_refs=[]
        weight=0

        for d, mask, p in zip(depth, explainability_mask,pose):
            current_loss,ref_imgs_warped,ego_flows_scaled= one_scale(d, mask,p)
            _, _, h, w = d.size()
            weight += h * w
            loss = loss + current_loss * h * w
            ego_flows.append(ego_flows_scaled)
            warped_refs.append(ref_imgs_warped)

        loss = loss / weight
        return loss,warped_refs,ego_flows


class explainability_loss(nn.Module):    # Motion mask loss  (abs+mean(grad(exp)) + binary_cross_entropy(exp[0]))
    def __init__(self):
        super(explainability_loss, self).__init__()
    def forward(self,mask,gt_mask):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(mask) not in [tuple, list]:
            mask = [mask]
        gt_mask=(gt_mask>0.01).type_as(mask[0])   # 阈值处理，将大于0.01的值设置为1，其余为0

        loss = 0
        weight=0
        for mask_scaled in mask:
            N,C,H,W=mask_scaled.shape
            mask_scaled=torch.clamp(mask_scaled, min=0.001, max=0.999)   # 对于每个掩码，获取其形状信息，并将其限制在0.001和0.999之间的范围内

            if min(H,W)<4:
                continue
            dx, dy = gradient(mask_scaled)
            loss += (dx.abs().view(N, -1).mean(1) + dy.abs().view(N, -1).mean(1)) * H * W  # abs+ mean of 1er gradients of weights + mean

            if C>1:   # 只考虑第一个通道
                mask_scaled=mask_scaled[:, :1]
                ones_var = (F.adaptive_avg_pool2d(gt_mask.type_as(mask_scaled), (H, W)) < 0.01).type_as(mask_scaled)#background_mask
                # 计算背景掩码ones_var, 自适应平均池化操作将真实掩码gt_mask缩放到当前掩码尺寸，并将小于0.01的值设为1，其余为0
            else:
                ones_var = (F.adaptive_avg_pool2d(gt_mask.type_as(mask_scaled),(H,W))>0.01).type_as(mask_scaled)#foreground mask
                # 如果通道数为1，计算前景掩码ones_var，通过自适应平均池化操作将真实掩码gt_mask缩放到当前掩码尺寸，并将大于0.01的值设为1，其余为0
            loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)*H*W
            weight+=H*W
        return loss/weight

class explainability_loss_new(nn.Module):
    def __init__(self):
        super(explainability_loss_new, self).__init__()
    def forward(self,mask,gt_mask):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(mask) not in [tuple, list]:
            mask = [mask]
        gt_mask=torch.round(gt_mask).long()

        loss = 0
        weight=0
        for mask_scaled in mask:
            N,C,H,W=mask_scaled.shape
            mask_scaled=torch.clamp(mask_scaled, min=0.001, max=0.999)

            if min(H,W)<4:
                continue
            dx, dy = gradient(mask_scaled)
            loss += (dx.abs().view(N, -1).mean(1) + dy.abs().view(N, -1).mean(1)) * H * W

            gt=F.adaptive_avg_pool2d(gt_mask.float(),(H, W)).long()
            w=torch.FloatTensor(C)
            for c in range(C):
                w[c]=1/(((gt==c).sum()/N+1).float())
                #根据每个类别计算权重w, 对于每个类别c，计算该类别在真实掩码中的样本数量除以总样本数的比例，并取其倒数作为权重。这样，样本数量较少的类别将具有较高的权重
            #print(nn.functional.cross_entropy(mask_scaled, gt[:,0],weight=w.type_as(mask_scaled)),nn.functional.cross_entropy(mask_scaled, gt[:,0]))
            #loss = loss+ nn.functional.cross_entropy(mask_scaled, gt[:,0],weight=w.type_as(mask_scaled))*H*W


            for c in range(C):
                ones_var = (F.adaptive_avg_pool2d((gt_mask==c).type_as(mask_scaled), (H, W)) ).type_as(mask_scaled)
                loss = loss+ nn.functional.binary_cross_entropy(mask_scaled[:, c:c+1], ones_var)*H*W
                #print(c,ones_var.sum().item())
            weight+=H*W
        return loss/weight

"""
class explainability_loss_new2(nn.Module):
    def __init__(self):
        super(explainability_loss_new2, self).__init__()

    def forward(self, mask, gt_mask):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(mask) not in [tuple, list]:
            mask = [mask]
        gt_mask = torch.round(gt_mask).long()

        loss = 0
        weight = 0
        for mask_scaled in mask:
            N, C, H, W = mask_scaled.shape
            mask_scaled = torch.clamp(mask_scaled, min=0.001, max=0.999)

            if min(H, W) < 4:
                continue
            dx, dy = gradient(mask_scaled)
            loss += (dx.abs().view(N, -1).mean(1) + dy.abs().view(N, -1).mean(1)) * H * W

            gt = F.adaptive_avg_pool2d(gt_mask.float(), (H, W)).long()
            w = H*W/((gt == 0).view(N,-1).sum(-1)+1).float()
            w=w/sum(w)

            mask_scaled=mask_scaled[:, :1]
            ones_var = (F.adaptive_avg_pool2d(gt_mask.type_as(mask_scaled), (H, W)) < 0.01).float()#background_mask

            loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var,weight=w.view(N,1,1,1))*H*W

            weight += H * W
        return loss / weight
"""


class explainability_loss_new2(nn.Module):
    def __init__(self):
        super(explainability_loss_new2, self).__init__()
    def forward(self,mask,gt_mask):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(mask) not in [tuple, list]:
            mask = [mask]
        gt_mask=torch.round(gt_mask).long()

        loss = 0
        weight=0
        for mask_scaled in mask:
            N,C,H,W=mask_scaled.shape
            mask_scaled=torch.clamp(mask_scaled, min=0.001, max=0.999)

            if min(H,W)<4:
                continue
            dx, dy = gradient(mask_scaled)
            loss += (dx.abs().view(N, -1).mean(1) + dy.abs().view(N, -1).mean(1)) * H * W

            gt=F.adaptive_avg_pool2d(gt_mask.float(),(H, W)).long()
            w=torch.FloatTensor(C)
            for c in range(C):
                w[c]=1/(((gt==c).sum()/N+1).float())
            #print('exp loss new2 w = ',w)
            #print(nn.functional.cross_entropy(mask_scaled, gt[:,0],weight=w.type_as(mask_scaled)),nn.functional.cross_entropy(mask_scaled, gt[:,0]))
            loss = loss+ nn.functional.cross_entropy(mask_scaled, gt[:,0],weight=w.type_as(mask_scaled))*H*W

            weight+=H*W
        return loss/weight



class depth_loss(nn.Module):   # Loss depth
    def __init__(self):
        super(depth_loss, self).__init__()
    def forward(self, gt, predicts,eps=1e-5):
        weight=0
        abs_rel=0.
        acc=0.
        valid_gt = ((gt > 100 / 6000) * (gt < 7000 / 6000)).type_as(gt)  #根据预测深度值 gt，生成一个有效性掩码，将深度值在 100/6000 和 7000/6000 之间的像素视为有效。
        for pred in predicts:
            N, C, H, W = pred.shape
            current_gt = F.adaptive_avg_pool2d(gt, (H, W))  # 将真实深度图 gt 通过自适应平均池化操作调整到与当前预测结果相同的尺寸
            weight += H * W
            valid = (F.adaptive_avg_pool2d(valid_gt, (H, W))>0.999).type_as(gt)  # 将有效性掩码 valid_gt 通过自适应平均池化操作调整到与当前预测结果相同的尺寸，并进行二值化处理，得到当前预测结果中的有效像素掩码
            masked_gt=current_gt*valid  # 得到只包含有效像素的深度图。
            masked_pred=pred*valid
            pred = pred * (torch.mean(masked_gt.view(N,-1),1) / (eps+torch.mean(masked_pred.view(N,-1),1))).view(N,1,1,1)  # 对预测结果进行缩放，使其与真实深度图的平均值相匹配。这样做的目的是对预测结果进行归一化，使其与真实深度图保持一致。
            thresh = torch.max((masked_gt / (eps+pred)), (pred / (eps+masked_gt)))*valid   # 计算阈值误差，选择真实深度图与预测结果的最大值比较，并乘以有效像素掩码。
            cost=(torch.abs(current_gt - pred) / current_gt)*valid   # 计算绝对相对误差，将真实深度图与预测结果之间的绝对差异除以真实深度图，并乘以有效像素掩码。
            abs_rel += cost.view(N, -1).mean(1)*H*W    # 将绝对相对误差乘以像素数量，并累加到总的绝对相对误差中。 sum abs_rel
            acc+=thresh.view(N,-1).mean(1)*H*W   # 将阈值误差乘以像素数量，并累加到总的阈值误差中  sum acc
        return (abs_rel+acc)/weight

class smooth_loss(nn.Module):  # smooth penelty on 2nd gradiant of predict
    def __init__(self):
        super(smooth_loss, self).__init__()

    def forward(self, pred_map,p=.5,eps=1e-4):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(pred_map) not in [tuple, list]:
            pred_map = [pred_map]

        loss = 0
        weight =0

        for scaled_map in pred_map:
            N,C,H,W = scaled_map.shape
            if min(H,W)<4:     # 如果当前尺度地图的高度和宽度小于4，则跳过该尺度，继续下一个尺度的处理。
                continue
            dx, dy = gradient(scaled_map)   # 计算二阶梯度（dx2、dxdy、dydx和dy2）
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            #loss += (dx2.abs().view(N,-1).mean(1) + dxdy.abs().view(N,-1).mean(1) + dydx.abs().view(N,-1).mean(1) + dy2.abs().view(N,-1).mean(1))*weight
            loss += (torch.pow(torch.clamp(dx2.abs(), min=eps),p).view(N, -1).mean(1)     # 绝对值dx2中的每个元素小于eps的值裁剪为eps，大于等于的保持不变, p=0.5,平方根计算
                     + torch.pow(torch.clamp(dxdy.abs(), min=eps),p).view(N, -1).mean(1)
                        + torch.pow(torch.clamp(dydx.abs(), min=eps),p).view(N, -1).mean(1)
                           + torch.pow(torch.clamp(dy2.abs(), min=eps),p).view(N, -1).mean(1)) * H*W

            weight += H*W

        return loss/weight





class pose_smooth_loss(nn.Module):  # 3 parts
    def __init__(self):
        super(pose_smooth_loss, self).__init__()

    def forward(self, pred_map,pose,mask):  # final_pose, pose, gt_mask
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy


        bg_mask=(mask<0.01).type_as(pose) # 生成背景掩码bg_mask，将小于0.01的值设置为1，其余为0
        M=int(mask.max().item())  # 获取掩码中的最大值M，用于后续的循环

        if type(pred_map) not in [tuple, list]:
            pred_map = [pred_map]

        loss = 0
        weight = 0
        pose = pose.view(-1, 6, 1, 1)  # ego pose

        for i,scaled_map in enumerate(pred_map): # 5 pred layers
            N, _, H, W = scaled_map.shape
            # print('scale pose shape', np.shape(scaled_map)) # b, 6, 480,640/ b,6,240,320.../b,6,30,40
            if H > 3 and W > 3:
                dx, dy = gradient(scaled_map)
                #loss += (dx.abs().view(N, -1).mean(1) + dy.abs().view(N, -1).mean(1)) * H*W
                loss += (dx.abs().reshape(N, -1).mean(1) + dy.abs().reshape(N, -1).mean(1)) * H*W    # abs+mean of 1er gradient 
                
                #loss += NonLocalSmoothnessLoss(scaled_map, p=1., eps=1e-6, R=0.2, B=3)*H*W

                #loss += 10*(((scaled_map-pose).abs()*F.adaptive_avg_pool2d(bg_mask,(H,W))).view(N, -1)).mean(1)
                # (final pose - ego pose)* bg_mask
                loss += 10*(((scaled_map-pose).abs()*F.adaptive_avg_pool2d(bg_mask,(H,W))).reshape(N, -1)).mean(1)

                for j in range(1,M):  # 对于掩码中的每个类别j（除了背景类别）
                    # print(j)
                    obj_mask=F.adaptive_avg_pool2d((mask==j).type_as(pose),(H,W))  # gt_obj_mask
                    # test = obj_mask>0.01
                    # import cv2
                    # import numpy as np
                    # import matplotlib.pyplot as plt
                    # plt.imshow(test.permute(0,2,3,1).data.cpu().numpy()[0], cmap='gray')
                    # plt.show()
                    mean_pose=(scaled_map*obj_mask).view(N, 6,-1).sum(2)/(1e-6+obj_mask.view(N, -1).sum(1).view(N,1))
                    mean_dev=((scaled_map-mean_pose.view(N,6,1,1)).abs()*obj_mask).view(N, 6,-1).sum(2)/(1e-6+obj_mask.view(N, -1).sum(1).view(N,1))
                    loss += 100 * mean_dev.mean() * H * W/M   # 可以促使预测地图在每个类别上更接近于该类别的均值姿势
                weight += H*W 

        return loss/weight







def box_filter(tensor,R):
    N,C,H,W=tensor.shape
    cumsum = torch.cumsum(tensor, dim=2)
    slidesum=torch.cat((cumsum[:, :, R:2 * R + 1, :],cumsum[:, :, 2 * R + 1:H, :] - cumsum[:, :, 0:H - 2 * R - 1, :],cumsum[:, :, -1:, :] - cumsum[:, :, H - 2 * R - 1:H - R - 1, :]),dim=2)
    cumsum = torch.cumsum(slidesum, dim=3)
    slidesum=torch.cat((cumsum[:, :, :, R:2 * R + 1],cumsum[:, :, :, 2 * R + 1:W] - cumsum[:, :, :, 0:W - 2 * R - 1],cumsum[:, :, :, -1:] - cumsum[:, :, :, W - 2 * R - 1:W - R - 1]),dim=3)
    return slidesum




def NonLocalSmoothnessLoss(I, p=1.0,eps=1e-4, R=0.1, B=10 ):

    N, C, H, W = I.shape

    R=int(min(H,W)*R)
    if H<10 or W<10 or R<2:
        return 0

    loss = 0.

    J=I

    min_J, _ = torch.min(J.view(N, C, -1), dim=2)
    max_J, _ = torch.max(J.view(N, C, -1), dim=2)
    min_J = min_J.view(N, C, 1, 1, 1)
    max_J = max_J.view(N, C, 1, 1, 1)
    Q = torch.from_numpy(np.linspace(0.0, 1.0, B + 1)).type_as(min_J).view(1, 1, 1, 1, B + 1)
    Q = Q * (max_J - min_J + 1e-5) + min_J
    min_J = min_J.view(N, C, 1, 1)
    max_J = max_J.view(N, C, 1, 1)
    Bin1 = torch.floor((J - min_J) / (max_J - min_J + 1e-5) * B).long()
    Bin2 = torch.ceil((J - min_J) / (max_J - min_J + 1e-5) * B).long()

    I_old = I#.detach()

    W1 = (torch.abs(J - Q[:, :, :, :, 0]) + eps) ** (p - 2)
    W_sum1 = box_filter(W1, R)
    WI_sum1 = box_filter(W1 * I_old, R)
    WI2_sum1 = box_filter(W1 * (I_old ** 2), R)
    loss1 = W_sum1 * (I ** 2) - 2 * I * WI_sum1 + WI2_sum1

    W_sum = 0

    for i in range(1, B + 1):

        W2 = (torch.abs(J - Q[:, :, :, :, i]) + eps) ** (p - 2)
        W_sum2 = box_filter(W2, R)
        WI_sum2 = box_filter(W2 * I_old, R)
        WI2_sum2 = box_filter(W2 * (I_old ** 2), R)
        loss2 = W_sum2 * (I ** 2) - 2 * I * WI_sum2 + WI2_sum2

        mask1 = (Bin1 == (i - 1)).float()
        mask2 = (Bin2 == i).float()


        slice_loss = (loss1 * (Q[:, :, :, :, i] - J) * mask1 + loss2 * (J - Q[:, :, :, :, i - 1]) * mask2) / (
                    Q[:, :, :, :, i] - Q[:, :, :, :, i - 1])

        loss = loss + slice_loss

        W_sum = W_sum + (
                    W_sum1 * (Q[:, :, :, :, i] - J) * mask1 + W_sum2 * (J - Q[:, :, :, :, i - 1]) * mask2) / (
                            Q[:, :, :, :, i] - Q[:, :, :, :, i - 1])

        W_sum1 = W_sum2
        loss1 = loss2

    loss = torch.mean((loss / W_sum).view(N,-1),1)

    return loss



def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[0][valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]

