import os
import torch.utils.data as data
from utils import *
import cv2
import random


class Dataset_MRCT(data.Dataset):
    def __init__(self, data_dir, MR_list, CT_list):
        super(Dataset_MRCT, self).__init__()
        self.data_dir = data_dir
        self.MR_list = MR_list
        self.CT_list = CT_list
        # self.MR_data = {}
        # self.CT_data = {}
        #
        # self.read_MR_img_mask()
        # self.read_CT_img_mask()


    # def read_MR_img_mask(self):
    #     for file in self.MR_list:
    #         patients_path = os.path.join(self.data_dir, "img", "MR", file)
    #         masks_path = os.path.join(self.data_dir, "label", "MR", file)
    #         imgs = os.listdir(patients_path)
    #         imgs_data = []
    #         masks_data = []
    #         for i in imgs:
    #             img = cv2.imread(patients_path+"/"+i, -1)
    #             mask = cv2.imread(masks_path+"/"+i, -1)
    #             imgs_data.append(img//255)
    #             masks_data.append(mask)
    #         self.MR_data[file] = (np.array(imgs_data), np.array(masks_data))
    #         del imgs_data, masks_data
    #
    #
    # def read_CT_img_mask(self):
    #     for file in self.CT_list:
    #         patients_path = os.path.join(self.data_dir, "img", "CT", file)
    #         masks_path = os.path.join(self.data_dir, "label", "CT", file)
    #         imgs = os.listdir(patients_path)
    #         imgs_data = []
    #         masks_data = []
    #         for i in imgs:
    #             img = cv2.imread(patients_path+"/"+i, -1)
    #             mask = cv2.imread(masks_path+"/"+i, -1)
    #             imgs_data.append(img//255)
    #             masks_data.append(mask)
    #         self.CT_data[file] = (np.array(imgs_data), np.array(masks_data))
    #         del imgs_data, masks_data

    # def __getitem__(self, item):
    #     MR_imgs, MR_masks = self.MR_data[self.MR_list[item]]
    #     ct_num = random.randint(0, len(self.CT_list)-1)
    #     CT_imgs, CT_masks = self.CT_data[self.CT_list[ct_num]]
    #     return self.MR_list[item], MR_imgs, MR_masks, self.CT_list[ct_num], CT_imgs, CT_masks


    def get_imgs_masks(self, type, name):
        patients_path = os.path.join(self.data_dir, "img", type, name)
        masks_path = os.path.join(self.data_dir, "label", type, name)
        imgs = os.listdir(patients_path)
        imgs_data = []
        masks_data = []
        for i in imgs:
            img = cv2.imread(patients_path+"/"+i, -1)
            mask = cv2.imread(masks_path+"/"+i, -1)
            imgs_data.append(img//255)
            masks_data.append(mask)
        return np.array(imgs_data), np.array(masks_data)

    def __getitem__(self, item):
        MR_name = self.MR_list[item]
        MR_imgs, MR_masks = self.get_imgs_masks("MR", MR_name)

        ct_num = random.randint(0, len(self.CT_list)-1)
        CT_name = self.CT_list[ct_num]
        CT_imgs, CT_masks = self.get_imgs_masks("CT", CT_name)

        return MR_name, MR_imgs, MR_masks, self.CT_list[ct_num], CT_imgs, CT_masks


    def __len__(self):
        return len(self.MR_list)