import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel as nib
import torchvision.utils as vutils
from scipy import ndimage
import random


class niiDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nib.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path)

    def __len__(self):
        return len(self.database)

class niiDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = directory
        self.transform = transform

        self.test_flag = test_flag
        if test_flag:
            self.datalistpath = directory + '/mr_s_test_list'
        else:
            self.datalistpath = directory + '/mr_s_train_list2'
        
        with open(self.datalistpath,'r') as f:
            self.datalist = f.readlines()
    
        for idx,i in enumerate(self.datalist):
            self.datalist[idx] = self.datalist[idx].strip('\n')
        
    
    def __len__(self):
        return len(self.datalist) 

    def get_imgs_masks_3d(self, type, name, length=10):

        data_dir = r'/data/linqx/Multictmrus/'
        patients_path = os.path.join(data_dir, "img_nii", type, name)
        masks_path = os.path.join(data_dir, "label_nii", type, name)

        imgs_data = nib.load(patients_path).get_fdata()
        # print(imgs_data.shape)
        random_limit = imgs_data.shape[2]

        imgs_data = imgs_data.transpose(2,0,1)
        imgs_data = imgs_data/255
        masks_data = nib.load(masks_path).get_fdata()
        masks_data = masks_data.transpose(2,0,1)

        # resize from 80 to 64 for h w
        def __resize_data__(data, inputsize):
            """
            Resize the data to the input size
            """ 
            [input_H,input_W,input_D] = inputsize
            [height, width, depth] = data.shape
            scale = [input_H*1.0/height, input_W*1.0/width, input_D*1.0/depth]  
            data = ndimage.interpolation.zoom(data, scale, order=0)
            return data
        
        inputsize = [128, 256, 256]

        imgs_data = __resize_data__(imgs_data,inputsize)
        masks_data = __resize_data__(masks_data,inputsize)

        slices = random.randint(0,127)

        image_patch = imgs_data[slices,:,:]
        label_patch = masks_data[slices,:,:]

        # print(image_patch.shape)
        image_patch = np.expand_dims(image_patch,0)
        label_patch = np.expand_dims(label_patch,0)

        image_patch = image_patch.astype(np.float32)
        label_patch = label_patch.astype(np.float32)

        # print(image_patch.shape)
        # print(label_patch.shape)

        return image_patch, label_patch, patients_path ,masks_path, slices

    # def get_imgs_masks_2d(self, type, name, length=10):]



    def __getitem__(self, x):
        
        filedict = self.datalist[x]
        imgs, masks, patients_path, masks_path, slices = self.get_imgs_masks_3d('MR_90',filedict)
        
        # print(imgs.shape)
        # print(masks.shape)
        if self.test_flag == True:
            return (imgs, imgs, patients_path.split('.nii.gz')[0] + "_slice" + str(slices) + ".nii.gz") # virtual path
        else:
            return (imgs, masks, patients_path.split('.nii.gz')[0] + "_slice" + str(slices) + ".nii.gz") # virtual path


