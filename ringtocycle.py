#!usr/bin/env python 
# -*- coding:utf-8-*-

import numpy as np
import cv2
import os
import numpy
import scipy.misc


def ring2cycle(img_ring):
    # 对每个环形mask进行操作
    # img_ring = cv2.imread(img_ring_path, cv2.COLOR_BGR2GRAY)
    mask_ring = np.array(img_ring)
    [a, b] = np.shape(mask_ring)  # (80,80)
    M = np.where(mask_ring != 0)
    c = 0
    nonzeros_num = np.zeros(len(M[0]))
    save_cycle_path1 = ""
    for i in range(a):
        li = []
        for j in range(b):
            if mask_ring[i, j] != 0:
                nonzeros_num[c] = mask_ring[i, j]
                v = nonzeros_num[c]
                c = c + 1
                li.append(j)
                # print("c:", c, "v:", v, "coor:", [i, j])
        myli = numpy.array(li)
        # print("myli:", myli)
        if len(myli) != 0:
            first = myli[0]
            last = myli[len(myli) - 1]
            for j in range(b):
                if (j > first and j < last):
                    mask_ring[i, j] = 255
    final_image = mask_ring
    return final_image

if __name__ == '__main__':

    img_ring_path = "G:/PHD/code/result/unet_1024_04/"
    save_cycle_path = "G:/PHD/code/result/unet_1024_05/"

    #循环找到每个环形mask
    patient_list = os.listdir(img_ring_path)
    for patient_name in patient_list:
        patient_name_path = img_ring_path + patient_name + "/"
        patient_img_list = os.listdir(patient_name_path)
        for img_file_name in patient_img_list:
            # img_path = patient_name_path + patient_name + "/" + img_file_name
            img_name_list = img_file_name.split("_")
            if img_name_list[2] =="01.png":
                img_ring = patient_name_path+img_file_name
                print(img_ring)

                #对每个环形mask进行操作
                img_ring = cv2.imread(img_ring, cv2.COLOR_BGR2GRAY)
                mask_ring = np.array(img_ring)
                [a, b] = np.shape(mask_ring)  # (80,80)
                M = np.where(mask_ring != 0)
                c = 0
                nonzeros_num = np.zeros(len(M[0]))
                save_cycle_path1 = ""
                for i in range(a):
                    li = []
                    for j in range(b):
                        if mask_ring[i, j] != 0:
                            nonzeros_num[c] = mask_ring[i, j]
                            v = nonzeros_num[c]
                            c = c + 1
                            li.append(j)
                            print("c:", c, "v:", v, "coor:", [i, j])
                    myli = numpy.array(li)
                    print("myli:", myli)
                    if len(myli) != 0:
                        first = myli[0]
                        last = myli[len(myli) - 1]
                        for j in range(b):
                            if (j > first and j < last):
                                mask_ring[i, j] = 255
                final_image = mask_ring
                print(final_image)
                # img_cycle = cv2.cvtColor(numpy.asarray(final_image), cv2.COLOR_BGR2GRAY)

                save_cycle_path1 = save_cycle_path + patient_name + "/" + img_name_list[0] + "_instance_02.png"
                # cv2.imwrite(save_cycle_path,final_image)
                scipy.misc.imsave(save_cycle_path1, final_image)

            print("lv")















