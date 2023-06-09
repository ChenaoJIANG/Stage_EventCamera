import torch

#from scipy.misc import imread, imsave, imresize

from imageio.v2 import imread, imsave
import cv2
import numpy as np
from path import Path
import argparse

import models
from utils import tensor2array

from inverse_warp import *
from flowlib import *

from timeit import default_timer as timer

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--pretrained-posenet", default=None, type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=260, type=int, help="Image height")  #260 480
parser.add_argument("--img-width", default=346, type=int, help="Image width")  #346 640
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")


parser.add_argument('--arch', default='ecn', help='architecture')
parser.add_argument('--norm-type', default='fd', help='normalization type')

parser.add_argument('--n-channel', '--init-channel', default=32, type=int,
                    help='initial feature channels(32|64|128).')
parser.add_argument('--growth-rate', default=32, type=int, help='feature channel growth rate.')
parser.add_argument('--scale-factor', default=1. / 2.,
                    type=float, help='scaling factor of each layer(0.5|0.75|0.875)')
parser.add_argument('--final-map-size', default=8, type=int, help='final map size')

parser.add_argument("--pixelpose", action='store_true', help="use binary mask and pixel wise pose")
parser.add_argument('-c','--n-motions', default=4, type=int, metavar='N',
                    help='number of independent motions')

def main():
    args = parser.parse_args()

    if args.arch=='ecn':

        disp_net = models.ECN_Disp(input_size=args.img_height,init_planes=args.n_channel,scale_factor=args.scale_factor,growth_rate=args.growth_rate,final_map_size=args.final_map_size,norm_type=args.norm_type).cuda()
    else:
        disp_net = models.DispNetS().cuda()

    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'],strict=False)
    disp_net.eval()

    if args.pretrained_posenet:
        if args.arch=='ecn':
            if args.pixelpose:
                
                pose_net = models.ECN_PixelPose(input_size=args.img_height, nb_ref_imgs=args.sequence_length - 1,
                                           init_planes=args.n_channel // 2, scale_factor=args.scale_factor,
                                           growth_rate=args.growth_rate // 2, final_map_size=args.final_map_size,
                                           norm_type=args.norm_type).cuda()

            else:
                pose_net = models.ECN_Pose(input_size=args.img_height, nb_ref_imgs=args.sequence_length - 1,
                                               init_planes=args.n_channel // 2, scale_factor=args.scale_factor,
                                               growth_rate=args.growth_rate // 2, final_map_size=args.final_map_size,
                                               norm_type=args.norm_type,n_motions=args.n_motions).cuda()
        else:
            pose_net = models.PoseExpNet(nb_ref_imgs=args.sequence_length - 1).cuda()

        weights = torch.load(args.pretrained_posenet)
        pose_net.load_state_dict(weights['state_dict'],strict=False)
        pose_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    import os
    import glob
    import time

    scene=os.path.join(args.dataset_dir)

    f = open(os.path.join(dataset_dir, 'calib.txt'), 'r')
    line = f.readline()
    l = [float(num) for num in line.split()]
    intrinsics = np.asarray([[l[1], 0, l[3]], [0, l[0], l[2]], [0, 0, 1]]).astype(np.float32)
    distortion = np.asarray(l[4:]).astype(np.float32)

    #intrinsics = np.genfromtxt(dataset_dir / 'calib.txt',max_rows=3).astype(np.float32).reshape((3, 3))

    intrinsics_inv = np.linalg.inv(intrinsics)

    intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).cuda()
    intrinsics_inv = torch.from_numpy(intrinsics_inv).unsqueeze(0).cuda()
    imgs = sorted(glob.glob(os.path.join(scene ,'slices' ,'frame*.png')))
    masks = sorted(glob.glob(os.path.join(scene, 'slices', 'mask*.png')))
    depth_gt = sorted(glob.glob(os.path.join(scene, 'slices' , 'depth*.png')))
    vis = sorted(glob.glob(os.path.join(scene, 'vis' , 'frame*.png')))
    print('{} files to test'.format(len(imgs)))

    #for file in tqdm(test_files):
    
    

    class File:
        basename=None
        ext=None

    demi_length = (args.sequence_length - 1) // 2
    shifts = list(range(-demi_length, demi_length + 1))
    shifts.pop(demi_length)

    counter=0
    depth_time = 0.
    pose_time = 0.
    warp_time = 0.

    import warnings
    warnings.filterwarnings("ignore")
    T_ego = []
    for i in range(demi_length,len(imgs)-demi_length):
    #for i in range(round(len(imgs)*.9), len(imgs) - demi_length):
    #for i in range(demi_length, round(len(imgs)*0.1 - demi_length)):

        file =File()
        file.namebase=os.path.basename(imgs[i]).replace('.png','')
        file.ext='.png'

        img = imread(imgs[i]).astype(np.float32)
        vi = imread(vis[i]).astype(np.float32)

        # d_gt = imread(depth_gt[i]).astype(np.float32)
        # #m_gt = imread(masks[i]).astype(np.float32)

        # nmin = np.nanmin(d_gt)
        # nmax = np.nanmax(d_gt)
        # d_gt = (d_gt - nmin) / (nmax - nmin) * 255
        # d_gt = cv2.cvtColor(d_gt,cv2.COLOR_GRAY2BGR)

        # obj_mask = imread(masks[i]).astype(np.float32)
        # obj_mask=np.round(obj_mask/1000)
        # obj_mask = cv2.cvtColor(obj_mask,cv2.COLOR_GRAY2BGR)

        ref_imgs=[]
        for j in shifts:
            ref_imgs.append(imread(imgs[i + j]).astype(np.float32))

        h, w, _ = img.shape
        img0=img


        
        with torch.no_grad():

            img = np.transpose(img, (2, 0, 1))
            ref_imgs = [np.transpose(im, (2, 0, 1)) for im in ref_imgs]
            img = torch.from_numpy(img).unsqueeze(0)
            ref_imgs = [torch.from_numpy(im).unsqueeze(0) for im in ref_imgs]
            img = (img / 255 ).cuda()
            ref_imgs = [(im / 255 ).cuda() for im in ref_imgs]



            counter+=1
            start = timer()

            output= disp_net(img)#,raw_disp

            end = timer()
            depth_time+=end-start

            output_depth = 1 / output

            if args.pretrained_posenet is not None:

                start = timer()

                if args.pixelpose:
                    explainability_mask,pose, pixel_pose,final_pose= pose_net(img, ref_imgs)#,raw_disp
                    _, rigid_flow = get_new_grid(output_depth[0], pixel_pose[:1, :], intrinsics, intrinsics_inv)
                    np.save(output_dir / 'pixel_pose_{}{}'.format(file.namebase, '.npy'),
                            pixel_pose[0].cpu().data.numpy().transpose((1, 2, 0)))
                    np.save(output_dir / 'motion_mask_{}{}'.format(file.namebase, '.npy'),
                            explainability_mask[0, 0].cpu().data.numpy())
                    exp = (255*tensor2array(explainability_mask[0].data.cpu(), max_value=None, colormap='bone')).astype(np.uint8).transpose(1,2,0)
                
                else:
                    explainability_mask, pose, final_pose = pose_net(img, ref_imgs)
                    #print('pose shape', np.shape(pose[0]))  # 1,6
                    t_ego = pose[:,0,:3].data.cpu().numpy()
                    T_ego.append(t_ego)

                    np.save(output_dir / 'pixel_pose_{}{}'.format(file.namebase, '.npy'),
                            final_pose[0].cpu().data.numpy().transpose((1, 2, 0)))
                    np.save(output_dir / 'motion_mask_{}{}'.format(file.namebase, '.npy'),
                            explainability_mask[0].cpu().data.numpy().transpose((1,2,0)))
                    #exp = (255*tensor2array(1-explainability_mask[0,0].data.cpu(), max_value=None, colormap='bone')).astype(np.uint8).transpose(1,2,0)
                    exp = (255*tensor2array(explainability_mask[0,0:3].data.cpu(), max_value=None, colormap='bone')).astype(np.uint8).transpose(1,2,0)
                    # exp[:,:,1]=exp[:,:,0]
                    # exp[:,:,2]=exp[:,:,0]
                    # print('exp shape',exp[0,0])
                # print('pose', pose[0,0].cpu().data.numpy())
                # print('final pose', np.shape(final_pose))
                # print('mask', np.shape(exp))

                end = timer()
                pose_time += end - start




                start = timer()
                _, ego_flow = get_new_grid(output_depth[0], pose[:1,:], intrinsics, intrinsics_inv)
                _, final_flow = get_new_grid(output_depth[0], final_pose[:1, :], intrinsics, intrinsics_inv)
                
                #_, ego_flow = inverse_warp(img, output_depth[:, 0], pose[:,0], intrinsics,intrinsics_inv, 'euler', 'border')  

                #print(projected_img.shape)
                residual_pose=(final_pose[0].permute(1,2,0)-pose)
                #print(residual_pose[120,-1])
                assert residual_pose[:,:,3:].abs().max().item()<1e-4
                residual_pose=residual_pose[:,:,:3].cpu().data.numpy()
                # print(np.shape(residual_pose))
                # pose_test = final_pose[0].permute(1,2,0)
                # pose_test = pose_test[:,:,:3].cpu().data.numpy()
                # print(np.shape(pose_test))

                end = timer()
                warp_time += end - start

                final_flow=final_flow[0].data.cpu().numpy()
                ego_flow=ego_flow[0].data.cpu().numpy()

                write_flow(final_flow, output_dir / 'final_flow_{}{}'.format(file.namebase, '.flo'))
                final_flow = flow_to_image(final_flow)
                #imsave(output_dir / 'final_flow_{}{}'.format(file.namebase, file.ext), final_flow)

                write_flow(ego_flow,output_dir / 'ego_flow_{}{}'.format(file.namebase, '.flo'))
                ego_flow = flow_to_image(ego_flow)
                #imsave(output_dir / 'ego_flow_{}{}'.format(file.namebase, file.ext), ego_flow)
               
            output=output[0].cpu()
            output_depth=output_depth[0,0].cpu()

            disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8).transpose(1,2,0)
            #print('disp shape ',disp[0,0])
            #imsave(output_dir/'disp_{}{}'.format(file.namebase,file.ext), disp)
            np.save(output_dir/'depth_{}{}'.format(file.namebase,'.npy'),output_depth.data.numpy())
            np.save(output_dir/'ego_pose_{}{}'.format(file.namebase,'.npy'),pose[0,0].cpu().data.numpy())
            np.save(output_dir/'residual_3d_pose_{}{}'.format(file.namebase,'.npy'),residual_pose)
    

            # print(np.shape(exp))
            # exp1 = np.dot(exp[...,:3], [0.2989, 0.5870, 0.1140])
            # exp[:,:,0] = exp1
            # exp[:,:,1] = exp1
            # exp[:,:,2] = exp1
            # # exp = np.where(exp > 150, 0, 255)
            # # print(np.max(exp))
            # seuil = int((np.max(exp)- np.min(exp) +1 )/2) + np.min(exp)
            # #exp0 = np.where(exp > seuil, 0, 255)
            # ret, exp0 = cv2.threshold(exp1, seuil, 255, cv2.THRESH_BINARY)
            # exp[:,:,0] = exp0
            # exp[:,:,1] = exp0
            # exp[:,:,2] = exp0
            # #disp_mask = disp + exp
            # disp_mask = cv2.addWeighted(disp, 1, exp, 0.5, 0)
            # #disp_mask = cv2.bitwise_and(disp,disp,mask = exp)

            if args.pretrained_posenet is not None:
                residual_pose=(residual_pose-residual_pose.min()) / (residual_pose.max()-residual_pose.min() + 1e-6)*255
                #print(np.shape(residual_pose))
                #print(residual_pose[120,620])
                #residual_pose[120,620,:]=0
                #pose_test = (pose_test-pose_test.min()) / (pose_test.max()-pose_test.min() + 1e-6)*255
                #cat_im=np.concatenate((img0,disp,ego_flow,exp,final_flow,residual_pose),axis=1)
                #cat_im=np.concatenate((img0,d_gt,disp,obj_mask, exp,final_flow),axis=1)

                cat_im=np.concatenate((vi,img0,disp,exp,final_flow,residual_pose),axis=1)
                #cat_im=np.concatenate((vi,img0,disp,exp,final_flow,ego_flow),axis=1)



            else:
                cat_im=np.concatenate((img0,disp),axis=1)
            imsave(output_dir / 'cat_{}{}'.format(file.namebase, file.ext), cat_im)
            imsave(output_dir / 'pred_mask{}{}'.format(file.namebase, file.ext), exp)
            print('-----{}/{}-----'.format(i,len(imgs)-demi_length))


    x = np.array(T_ego)[:,0,0]
    y = np.array(T_ego)[:,0,1]
    z = np.array(T_ego)[:,0,2]
    fig, ax = plt.subplots()
    ax.plot(x, label='x')
    ax.plot(y, label='y')
    ax.plot(z, label='z')
    ax.legend()
    plt.show()
    print('dataset:',dataset_dir)
    print('output:',output_dir)
    print('inputs: ', counter,', network time: ', depth_time+pose_time, ', depth net time: ',depth_time,' pose net time: ',pose_time,', warping time: ', warp_time)

if __name__ == '__main__':
    main()
