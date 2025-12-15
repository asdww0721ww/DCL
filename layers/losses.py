import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# from numpy.core.umath_tests import inner1d
def inner1d(a, b):
    return (a * b).sum(axis=-1)
import math
# import sys
# sys.path.append("/home/gsd-40160/3d_seg/3DSD/surface-distance/")
# from surface-distance.surcdface_distance import metrics
# from surface_distance import metrics
# import surface_distance as surfdist


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0) * target.size(1)
        smooth = 0.00001

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        dice = 2 * (intersection.sum(1) + smooth) / ((input_flat * input_flat).sum(1) + (target_flat * target_flat).sum(1) + smooth)

        dice = torch.mean(dice)
        return 1 - dice


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, input, target):
        N = target.size(0) * target.size(1)
        smooth = 0.00001

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        dice = 2 * (intersection.sum(1) + smooth) / ((input_flat * input_flat).sum(1) + (target_flat * target_flat).sum(1) + smooth)

        dice = torch.mean(dice)
        return dice


class Dice3D(nn.Module):
    def __init__(self):
        super(Dice3D, self).__init__()

    def forward(self, input, target):
        N = target.size(0) * target.size(1) * target.size(2)
        smooth = 0.00001

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        dice = 2 * (intersection.sum(1) + smooth) / ((input_flat * input_flat).sum(1) + (target_flat * target_flat).sum(1) + smooth)

        dice = torch.mean(dice)
        return dice


def ring2cycle_wrong(mask_ring):
    [a, b, h] = np.shape(mask_ring)  # (80,80, patch_size)
    new_mask = np.zeros_like(mask_ring)
    for t in range(h):
        img_tmp = mask_ring[:,:,t]
        M = np.where(img_tmp != 0)
        c = 0
        nonzeros_num = np.zeros(len(M[0]))
        save_cycle_path1 = ""
        for i in range(a):
            li = []
            for j in range(b):
                if img_tmp[i, j] != 0:
                    nonzeros_num[c] = img_tmp[i, j]
                    v = nonzeros_num[c]
                    c = c + 1
                    li.append(j)
                    # print("c:", c, "v:", v, "coor:", [i, j])
            myli = np.array(li)
            # print("myli:", myli)
            if len(myli) != 0:
                first = myli[0]
                last = myli[len(myli) - 1]
                for j in range(b):
                    if (j > first and j < last):
                        img_tmp[i, j] = 1
        new_mask[:,:,t] = img_tmp
    return new_mask

def ring2cycle(mask_ring):
    [h, a, b] = np.shape(mask_ring)  # (patch_size,80,80, )
    new_mask = np.zeros_like(mask_ring)
    for t in range(h):
        img_tmp = mask_ring[t,:,:]
        M = np.where(img_tmp != 0)
        c = 0
        nonzeros_num = np.zeros(len(M[0]))
        save_cycle_path1 = ""
        for i in range(a):
            li = []
            for j in range(b):
                if img_tmp[i, j] != 0:
                    nonzeros_num[c] = img_tmp[i, j]
                    v = nonzeros_num[c]
                    c = c + 1
                    li.append(j)
                    # print("c:", c, "v:", v, "coor:", [i, j])
            myli = np.array(li)
            # print("myli:", myli)
            if len(myli) != 0:
                first = myli[0]
                last = myli[len(myli) - 1]
                for j in range(b):
                    if (j > first and j < last):
                        img_tmp[i, j] = 1
        new_mask[t, :,:] = img_tmp
    return new_mask


def dice_score_muti_labels2(logits, targets, ratio=0.5):
    """
    function to calculate the dice score
    """
    logits[logits > 0.5] = 1
    logits[logits < 0.5] = 0
    smooth = 1e-5

    dices = []
    for class_index in range(targets.shape[0]):
        if class_index == 2:
            logits[class_index, :, :, :] = ring2cycle(logits[class_index, :, :, :])
            targets[class_index, :, :, :] = ring2cycle(targets[class_index, :, :, :])


        inter = (logits[class_index, :, :, :] * targets[class_index, :, :, :]).sum()
        union = (logits[class_index, :, :, :]).sum() + (targets[class_index, :, :, :]).sum()
        dice = (2. * inter + smooth) / (union + smooth)
        dices.append(dice.item())
    return np.asarray(dices)


def iou_score_muti_labels2(logits, targets, ratio=0.5):
    """
    function to calculate the dice score
    """
    logits[logits > 0.5] = 1
    logits[logits < 0.5] = 0
    smooth = 1e-5

    ious = []
    for class_index in range(targets.shape[0]):
        if class_index == 2:
            logits[class_index, :, :, :] = ring2cycle(logits[class_index, :, :, :])
            targets[class_index, :, :, :] = ring2cycle(targets[class_index, :, :, :])


        inter = (logits[class_index, :, :, :] * targets[class_index, :, :, :]).sum()
        union = (logits[class_index, :, :, :]).sum() + (targets[class_index, :, :, :]).sum() - inter
        iou = ( inter + smooth) / (union + smooth)
        ious.append(iou.item())
    return np.asarray(ious)

def hd_score_muti_labels2(logits, targets, ratio=0.5):
    logits[logits > 0.5] = 1
    logits[logits < 0.5] = 0
    smooth = 1e-5
    hds = []
    dHs = []
    
    for class_index in range(targets.shape[0]):
        if class_index == 2:
            logits[class_index, :, :, :] = ring2cycle(logits[class_index, :, :, :])
            targets[class_index, :, :, :] = ring2cycle(targets[class_index, :, :, :])

        for slice_index in range(targets.shape[1]):

            D_mat = np.sqrt(inner1d(logits[class_index, slice_index, :, :] ,logits[class_index, slice_index, :, :] )[np.newaxis].T
                        + inner1d(targets[class_index, slice_index, :, :],targets[class_index, slice_index, :, :])
                        -2*(np.dot(logits[class_index, slice_index, :, :],targets[class_index, slice_index, :, :].T)))
    # Find DH
            dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
            dHs.append(dH)
        dH_avg = np.mean(dHs)
        hds.append(dH_avg.item())
    return np.asarray(hds)


# def hd_score_muti_labels2(logits, targets, ratio=0.5):
#     logits[logits > 0.5] = 1
#     logits[logits < 0.5] = 0
#     smooth = 1e-5
#     hds = []
#
#     for class_index in range(targets.shape[0]):
#         if class_index == 2:
#             logits[class_index, :, :, :] = ring2cycle(logits[class_index, :, :, :])
#             targets[class_index, :, :, :] = ring2cycle(targets[class_index, :, :, :])
#
#
#         surface_distances = surfdist.compute_surface_distances(
#             targets[class_index, :, :, :] , logits[class_index, :, :, :], spacing_mm=(1.0, 1.0, 1.0))
#         hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
#
#         hds.append(hd_dist_95.item())
#     return np.asarray(hds)



def rmse_score_muti_labels2(logits, targets, ratio=0.5):
    """
    function to calculate the dice score
    """
    logits[logits > 0.5] = 1
    logits[logits < 0.5] = 0
    smooth = 1e-5

    rmses = []
    mses = []
    for class_index in range(targets.shape[0]):
        if class_index == 2:
            logits[class_index, :, :, :] = ring2cycle(logits[class_index, :, :, :])
            targets[class_index, :, :, :] = ring2cycle(targets[class_index, :, :, :])

        for slice_index in range(targets.shape[1]):
            mse = ((targets[class_index, slice_index, :, :]-logits[class_index, slice_index, :, :])**2 ).sum()/float(targets.shape[2] * targets.shape[3])
            mses.append(mse)

        mse_avg = np.mean(mses)

        rmse = np.sqrt(mse_avg)
        rmses.append(rmse.item())
    return np.asarray(rmses)



def rvd_score_muti_labels2(logits, targets, ratio=0.5):
    """
    function to calculate the dice score
    """
    logits[logits > 0.5] = 1
    logits[logits < 0.5] = 0
    smooth = 1e-5

    rvds = []
    for class_index in range(targets.shape[0]):
        if class_index == 2:
            logits[class_index, :, :, :] = ring2cycle(logits[class_index, :, :, :])
            targets[class_index, :, :, :] = ring2cycle(targets[class_index, :, :, :])

        l_v = (logits[class_index, :, :, :]).sum()
        t_v = (targets[class_index, :, :, :]).sum()

        rvd = (abs( t_v - l_v) + smooth)/ (abs(l_v)+ smooth)
        rvds.append(rvd.item())
    return np.asarray(rvds)



