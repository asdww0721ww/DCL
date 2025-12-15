import torch
import torch.nn as nn
import sys
import os
import time
import datetime
import argparse
# from loaders.dataset_loader import Dataset_MRCT
from loaders.data3D_loader import Dataset_MRCT
from layers.losses_ops import AverageMeter
from layers.losses import *
import torch.utils.data as data
from utils.data_utils import *
import utils
import loaders
import models
import cv2 as cv
from scipy.misc import toimage
import numpy as np
import nibabel as nib



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
load_weights_path = "/data/linqx/Models/m3m3dt/test02_model_state_epoch_14"
savepath = "/home/gsd-40160/3d_seg/3DSD/result/6ct_5_model_state_epoch_36"


def parse_arguments(args):
    usage_text = (
        "SDNet Pytorch Implementation"
        "Usage:  python test.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #model details
    parser.add_argument('-batch_size','--batch_size', type= int, default=1, help='Number of inputs per batch')
    parser.add_argument('-patch_size','--patch_size', type= int, default=64, help='patch_size')
    parser.add_argument('-spacing','--spacing', type= int, default=4, help='spacing')
    parser.add_argument('-n_labels','--n_labels', type= int, default=3, help='n_labels')
    parser.add_argument('-n','--name', type=str, default='ct_5', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-mn','--model_name', type=str, default='Muti_SDNet', help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('--load_weights_path', type=str, default=load_weights_path, help= 'Path to save model checkpoints')
    #data
    parser.add_argument("--data_path", type=str, default='/data/gsd/data_3D', help='Path to ACDC dataset')
    # parser.add_argument('--save_path', type=str, default='factors', help='Path to save the anatomy factors')

    # test details
    parser.add_argument('--dropout_p', type=float, default=0.3, help='dropout')
    parser.add_argument('-hidden_size', dest='hidden_size', default=256, type=int)
    parser.add_argument('-skip_mode', dest='skip_mode', default='concat',
                        choices=['sum', 'concat', 'mul', 'none'])

    #hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers' ,type= int, default = 0, help='Number of workers to use for dataload')
    #visdom params
    parser.add_argument('-d','--disp_iters', type=int, default=10, help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    parser.add_argument('--visdom', type=str, nargs='?', default=None, const="127.0.0.1", help = "Visdom server IP (port defaults to 8097)")
    parser.add_argument('--visdom_iters', type=int, default=10, help = "Iteration interval that results will be reported at the visdom server for visualization.")
    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, uknown = parse_arguments(sys.argv)
    #create and init device
    print('{} | Torch Version: {}'.format(datetime.datetime.now(), torch.__version__))
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    device = torch.device('cuda:{}' .format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else 'cpu')
    print('Testing {0} on {1}'.format(args.name, device))
    torch.cuda.set_device(int(args.gpu))

    #Visdom setup
    # note: no need of this one
    # visualizer = utils.visualization.NullVisualizer() if args.visdom is None\
    #     else utils.visualization.VisdomVisualizer(args.name, args.visdom, count=1)
    # if args.visdom is None:
    #     args.visdom_iters = 0

    loaded_params = torch.load(args.load_weights_path, map_location=device) #, map_location=device
    #Model selection and initialization
    model_params = {
        'width': loaded_params['width'],
        'height': loaded_params['height'],
        'ndf': loaded_params['ndf'],
        'norm': loaded_params['norm'],
        'upsample': loaded_params['upsample'],
        'num_classes': loaded_params['num_classes'],
        # 'decoder_type': loaded_params['decoder_type'],
        'anatomy_out_channels': loaded_params['anatomy_out_channels'],
        'z_length': loaded_params['z_length'],
        'num_mask_channels': loaded_params['num_mask_channels']
    }
    model = models.get_model(args, model_params)
    model.load_state_dict(loaded_params['model_state_dict'])
    model.to(device)
    model = nn.DataParallel(model,gpus)

    model.eval()
    # Dice = Dice().to(device)

    CT_all_dice_endo = []
    CT_all_dice_epi = []
    CT_all_iou_endo = []
    CT_all_iou_epi = []
    CT_all_hd_endo = []
    CT_all_hd_epi = []
    CT_all_rvd_endo = []
    CT_all_rvd_epi = []
    CT_all_rmse_endo = []
    CT_all_rmse_epi = []


    MR_all_dice_endo = []
    MR_all_dice_epi = []
    MR_all_iou_endo = []
    MR_all_iou_epi = []
    MR_all_hd_endo = []
    MR_all_hd_epi = []
    MR_all_rvd_endo = []
    MR_all_rvd_epi = []
    MR_all_rmse_endo = []
    MR_all_rmse_epi = []


    _, _, MR_test_list, _, _, CT_test_list = split_data(args.data_path)


    for CT_patients in CT_test_list:
        CT_patients_path = os.path.join(args.data_path, "img_nii", "CT", CT_patients)
        CT_label_path = os.path.join(args.data_path, "label_nii", "CT", CT_patients)


        CT_imgs_data = nib.load(CT_patients_path).get_fdata()
        CT_imgs_data = CT_imgs_data.transpose(2,0,1)
        CT_imgs_data = CT_imgs_data/255
        CT_masks_data = nib.load(CT_label_path).get_fdata()
        CT_masks_data = CT_masks_data.transpose(2,0,1)

        CT_gt_onehot = to_one_hot_3d(CT_masks_data, args.n_labels)

        d_CT, w_CT, h_CT = CT_imgs_data.shape
        pre_count_CT = np.zeros_like(CT_gt_onehot, dtype=np.float32)
        preout_CT = np.zeros_like(CT_gt_onehot, dtype=np.float32)
        predict_CT = np.zeros_like(CT_gt_onehot, dtype=np.float32)

        x_list_CT = np.squeeze(np.concatenate((np.arange(0, w_CT - args.patch_size, args.patch_size // args.spacing)[:, np.newaxis],np.array([w_CT - args.patch_size])[:, np.newaxis])).astype(np.int))
        
        y_list_CT = np.squeeze(np.concatenate((np.arange(0, h_CT - args.patch_size, args.patch_size // args.spacing)[:, np.newaxis],np.array([h_CT - args.patch_size])[:, np.newaxis])).astype(np.int))

        if d_CT <= args.patch_size:
            z_list_CT = [0]
        else:
            z_list_CT = np.squeeze(np.concatenate((np.arange(0, d_CT - args.patch_size, args.patch_size // 4)[:, np.newaxis],np.array([d_CT - args.patch_size])[:, np.newaxis])).astype(np.int))
        start_time = time.time()

        for z in z_list_CT:
            for x in x_list_CT:
                for y in y_list_CT:
                    image_patch = CT_imgs_data[ z:z + args.patch_size, x:x + args.patch_size, y:y + args.patch_size].astype(np.float32)
                    patch_tensor = torch.from_numpy(image_patch[np.newaxis, ...]).cuda()
                    preout_CT = model.a_encoder2(patch_tensor, 'test')
                    predict_CT[:, z:z + args.patch_size, x:x + args.patch_size, y:y + args.patch_size] += model.segmentor(
                        preout_CT, 'test').squeeze().cpu().data.numpy()

                    pre_count_CT[:, z:z + args.patch_size, x:x + args.patch_size, y:y + args.patch_size] += 1

        predict_CT /= pre_count_CT


        predict_CT = np.squeeze(predict_CT)
        CT_imgs_data = np.squeeze(CT_imgs_data)

        predict_CT[predict_CT > 0.5] = 1
        predict_CT[predict_CT < 0.5] = 0

        predict_3D_CT = one_hot2_3d(predict_CT)

        dice = dice_score_muti_labels2(predict_CT, CT_gt_onehot)
        iou = iou_score_muti_labels2(predict_CT, CT_gt_onehot)
        hd = hd_score_muti_labels2(predict_CT, CT_gt_onehot)
        rmse = rmse_score_muti_labels2(predict_CT, CT_gt_onehot)
        rvd = rvd_score_muti_labels2(predict_CT, CT_gt_onehot)

        CT_imgs_data = CT_imgs_data.transpose(1, 2, 0)
        predict_3D_CT = predict_3D_CT.transpose(1, 2, 0)

        image_nii_CT = nib.Nifti1Image(CT_imgs_data, affine=None)
        predict_nii_CT = nib.Nifti1Image(predict_3D_CT, affine=None)

        # mod: temp not save nii
        # check_dir(savepath)
        # check_dir(os.path.join(savepath, CT_patients.split('.')[0]))

        # nib.save(image_nii_CT, os.path.join(savepath, CT_patients.split('.')[0],'image.nii.gz'))
        
        # nib.save(predict_nii_CT,os.path.join(savepath, CT_patients.split('.')[0], 'predict.nii.gz'))
        
        print("{} Val_dice_endo  {:.3f}, Val_dice_epi  {:.3f},  Dice: {:.3f}".format(CT_patients, dice[1], dice[2], (dice[1] + dice[2]) / 2))
        
        print(" Val_iou_endo  {:.3f}, Val_iou_epi  {:.3f},  IOU: {:.3f}".format( iou[1], iou[2],(iou[1] + iou[2]) / 2))

        print(" Val_hd_endo  {:.3f}, Val_hd_epi  {:.3f},  HD: {:.3f}".format( hd[1], hd[2],(hd[1] + hd[2]) / 2))

        print(" Val_rmse_endo  {:.3f}, Val_rmse_epi  {:.3f},  RMSE: {:.3f}".format(rmse[1], rmse[2],(rmse[1] + rmse[2]) / 2))

        print(" Val_rvd_endo  {:.3f}, Val_rvd_epi  {:.3f},  RVD: {:.3f}".format(rvd[1], rvd[2], (rvd[1] + rvd[2]) / 2))



        CT_all_dice_endo.append(dice[1])
        CT_all_dice_epi.append(dice[2])
        CT_all_iou_endo.append(iou[1])
        CT_all_iou_epi.append(iou[2])
        CT_all_hd_endo.append(hd[1])
        CT_all_hd_epi.append(hd[2])
        CT_all_rmse_endo.append(rmse[1])
        CT_all_rmse_epi.append(rmse[2])
        CT_all_rvd_endo.append(rvd[1])
        CT_all_rvd_epi.append(rvd[2])


        # print("[{}] Testing Finished, Cost {:.2f}s, ".format(idx, time.time()-start_time))
    print("ALL CT :dice_endo  {:.3f}  {:.6f}, Val_dice_epi  {:.3f}  {:.6f}".format(np.average(CT_all_dice_endo),np.var(CT_all_dice_endo),np.average(CT_all_dice_epi),np.var(CT_all_dice_epi)))
    print("iou_endo  {:.3f}  {:.6f}, Val_iou_epi  {:.3f}  {:.6f}".format(np.average(CT_all_iou_endo),np.var(CT_all_iou_endo),np.average(CT_all_iou_epi),np.var(CT_all_iou_epi)))

    print("hd_endo  {:.3f}  {:.6f}, Val_hd_epi  {:.3f}  {:.6f}".format(np.average(CT_all_hd_endo),np.var(CT_all_hd_endo),np.average(CT_all_hd_epi),np.var(CT_all_hd_epi)))

    print("rmse_endo  {:.3f}  {:.6f}, Val_rmse_epi  {:.3f}  {:.6f}".format(np.average(CT_all_rmse_endo),np.var(CT_all_rmse_endo),np.average(CT_all_rmse_epi),np.var(CT_all_rmse_epi)))
    print("rvd_endo  {:.3f}  {:.6f}, Val_rvd_epi  {:.3f}  {:.6f}".format(np.average(CT_all_rvd_endo),np.var(CT_all_rvd_endo),np.average(CT_all_rvd_epi),np.var(CT_all_rvd_epi)))

    for MR_patients in MR_test_list:
        MR_patients_path = os.path.join(args.data_path, "img_nii", "MR_90", MR_patients)
        MR_label_path = os.path.join(args.data_path, "label_nii", "MR_90", MR_patients)


        MR_imgs_data = nib.load(MR_patients_path).get_fdata()
        MR_imgs_data = MR_imgs_data.transpose(2,0,1)
        MR_imgs_data = MR_imgs_data/255
        MR_masks_data = nib.load(MR_label_path).get_fdata()
        MR_masks_data = MR_masks_data.transpose(2,0,1)

        MR_gt_onehot = to_one_hot_3d(MR_masks_data, args.n_labels)

        d_MR, w_MR, h_MR = MR_imgs_data.shape
        pre_count_MR = np.zeros_like(MR_gt_onehot, dtype=np.float32)
        preout_MR = np.zeros_like(MR_gt_onehot, dtype=np.float32)
        prediMR_MR = np.zeros_like(MR_gt_onehot, dtype=np.float32)

        x_list_MR = np.squeeze(np.concatenate((np.arange(0, w_MR - args.patch_size, args.patch_size // args.spacing)[:, np.newaxis],np.array([w_MR - args.patch_size])[:, np.newaxis])).astype(np.int))

        y_list_MR = np.squeeze(np.concatenate((np.arange(0, h_MR - args.patch_size, args.patch_size // args.spacing)[:, np.newaxis],np.array([h_MR - args.patch_size])[:, np.newaxis])).astype(np.int))

        if d_MR <= args.patch_size:
            z_list_MR = [0]
        else:
            z_list_MR = np.squeeze(np.concatenate((np.arange(0, d_MR - args.patch_size, args.patch_size // 4)[:, np.newaxis],np.array([d_MR - args.patch_size])[:, np.newaxis])).astype(np.int))
        start_time = time.time()

        for z in z_list_MR:
            for x in x_list_MR:
                for y in y_list_MR:
                    image_patch = MR_imgs_data[ z:z + args.patch_size, x:x + args.patch_size, y:y + args.patch_size].astype(np.float32)
                    patch_tensor = torch.from_numpy(image_patch[np.newaxis, ...]).cuda()
                    preout_MR = model.a_encoder1(patch_tensor, 'test')
                    
                    prediMR_MR[:, z:z + args.patch_size, x:x + args.patch_size, y:y + args.patch_size] += model.segmentor(preout_MR, 'test').squeeze().cpu().data.numpy()

                    pre_count_MR[:, z:z + args.patch_size, x:x + args.patch_size, y:y + args.patch_size] += 1

        prediMR_MR /= pre_count_MR

        prediMR_MR = np.squeeze(prediMR_MR)
        MR_imgs_data = np.squeeze(MR_imgs_data)

        prediMR_MR[prediMR_MR > 0.5] = 1
        prediMR_MR[prediMR_MR < 0.5] = 0

        prediMR_3D_MR = one_hot2_3d(prediMR_MR)

        dice = dice_score_muti_labels2(prediMR_MR, MR_gt_onehot)
        iou = iou_score_muti_labels2(prediMR_MR, MR_gt_onehot)
        hd = hd_score_muti_labels2(prediMR_MR, MR_gt_onehot)
        rmse = rmse_score_muti_labels2(prediMR_MR, MR_gt_onehot)
        rvd = rvd_score_muti_labels2(prediMR_MR, MR_gt_onehot)


        MR_imgs_data = MR_imgs_data.transpose(1, 2, 0)
        prediMR_3D_MR = prediMR_3D_MR.transpose(1, 2, 0)

        image_nii_MR = nib.Nifti1Image(MR_imgs_data, affine=None)
        prediMR_nii_MR = nib.Nifti1Image(prediMR_3D_MR, affine=None)


        # mod:temp ignored saving
        # check_dir(savepath)
        # check_dir(os.path.join(savepath, MR_patients.split('.')[0]))
        # nib.save(image_nii_MR, os.path.join(savepath, MR_patients.split('.')[0],'image.nii.gz'))
        
        # nib.save(prediMR_nii_MR,
        #          os.path.join(savepath, MR_patients.split('.')[0], 'prediMR.nii.gz'))
        
        print("{} Val_dice_endo  {:.3f}, Val_dice_epi  {:.3f},  Dice: {:.3f}".format(MR_patients, dice[1], dice[2],(dice[1] + dice[2]) / 2))

        print(" Val_iou_endo  {:.3f}, Val_iou_epi  {:.3f},  IOU: {:.3f}".format( iou[1], iou[2],(iou[1] + iou[2]) / 2))
        
        print(" Val_hd_endo  {:.3f}, Val_hd_epi  {:.3f},  HD: {:.3f}".format( hd[1], hd[2], (hd[1] + hd[2]) / 2))

        print(" Val_rmse_endo  {:.3f}, Val_rmse_epi  {:.3f},  RMSE: {:.3f}".format( rmse[1], rmse[2],(rmse[1] + rmse[2]) / 2))

        print(" Val_rvd_endo  {:.3f}, Val_rvd_epi  {:.3f},  RVD: {:.3f}".format( rvd[1], rvd[2],(rvd[1] + rvd[2]) / 2))

        MR_all_dice_endo.append(dice[1])
        MR_all_dice_epi.append(dice[2])
        MR_all_iou_endo.append(iou[1])
        MR_all_iou_epi.append(iou[2])
        MR_all_hd_endo.append(hd[1])
        MR_all_hd_epi.append(hd[2])
        MR_all_rmse_endo.append(rmse[1])
        MR_all_rmse_epi.append(rmse[2])
        MR_all_rvd_endo.append(rvd[1])
        MR_all_rvd_epi.append(rvd[2])


        # print("[{}] Testing Finished, Cost {:.2f}s, ".format(idx, time.time()-start_time))
    print("ALL MR :dice_endo  {:.3f}  {:.6f}, Val_dice_epi  {:.3f}  {:.6f}".format(np.average(MR_all_dice_endo), np.var(MR_all_dice_endo),np.average(MR_all_dice_epi),np.var(MR_all_dice_epi)))
    print("iou_endo  {:.3f}  {:.6f}, Val_iou_epi  {:.3f}  {:.6f}".format(np.average(MR_all_iou_endo), np.var(MR_all_iou_endo),np.average(MR_all_iou_epi),np.var(MR_all_iou_epi)))
    print("hd_endo  {:.3f}  {:.6f}, Val_hd_epi  {:.3f}  {:.6f}".format(np.average(MR_all_hd_endo), np.var(MR_all_hd_endo),np.average(MR_all_hd_epi),np.var(MR_all_hd_epi)))
    print("rmse_endo  {:.3f}  {:.6f}, Val_rmse_epi  {:.3f}  {:.6f}".format(np.average(MR_all_rmse_endo), np.var(MR_all_rmse_endo),np.average(MR_all_rmse_epi),np.var(MR_all_rmse_epi)))

    print("rvd_endo  {:.3f}  {:.6f}, Val_rvd_epi  {:.3f}  {:.6f}".format(np.average(MR_all_rvd_endo), np.var(MR_all_rvd_endo),np.average(MR_all_rvd_epi),np.var(MR_all_rvd_epi)))
