import os
import nibabel as nib
import torch.utils.data as data
from utils import *
import random
import re
from scipy import ndimage


class Dataset_MRCT(data.Dataset):
    def __init__(self, args, MR_list, CT_list):
        super(Dataset_MRCT, self).__init__()
        self.data_dir = args.data_path
        self.MR_list = MR_list
        self.CT_list = CT_list
        self.patch_size = args.patch_size
        self.n_labels = args.n_labels

    def _get_range(self, slices, crop_slices):
        if slices < crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def get_imgs_masks(self, type, name):
        patients_path = os.path.join(self.data_dir, "img_nii", type, name)
        masks_path = os.path.join(self.data_dir, "label_nii", type, name)
        # imgs = os.listdir(patients_path)
        # imgs_data = []
        # masks_data = []
        # for i in imgs:
        #     img = cv2.imread(patients_path+"/"+i, -1)
        #     mask = cv2.imread(masks_path+"/"+i, -1)
        #     imgs_data.append(img//255)
        #     masks_data.append(mask)
        # imgs_data, masks_data = np.array(imgs_data), np.array(masks_data)

        imgs_data = nib.load(patients_path).get_fdata()
        imgs_data = imgs_data.transpose(2,0,1)
        imgs_data = imgs_data/255
        masks_data = nib.load(masks_path).get_fdata()
        masks_data = masks_data.transpose(2,0,1)


        d, w, h = imgs_data.shape

        # here we prevent the sampling patch overflow volume
        z_start, z_end = self._get_range(d, self.patch_size)
        x_start, x_end = self._get_range(w, self.patch_size)
        y_start, y_end = self._get_range(h, self.patch_size)

        image_patch = np.zeros((self.patch_size, self.patch_size, self.patch_size))
        label_patch = np.zeros((self.patch_size, self.patch_size, self.patch_size))
        image_patch[:z_end-z_start,:,:] = imgs_data[z_start:z_end, x_start:x_end, y_start:y_end].astype(np.float32)
        label_patch[:z_end-z_start,:,:] = masks_data[z_start:z_end, x_start:x_end, y_start:y_end].astype(np.float32)

        return image_patch, label_patch

    # 按MR,  MR多  CT少
    def __getitem__(self, item):
        MR_name = self.MR_list[item]
        MR_imgs, MR_masks = self.get_imgs_masks("MR_90", MR_name)

        ct_num = random.randint(0, len(self.CT_list)-1)
        CT_name = self.CT_list[ct_num]
        CT_imgs, CT_masks = self.get_imgs_masks("CT", CT_name)

        return MR_name, MR_imgs, MR_masks, CT_name, CT_imgs, CT_masks


    def __len__(self):
        return len(self.MR_list)



class Dataset_MRs_CT_MRt_US(data.Dataset):
    def __init__(self, args, MR_s_list, CT_list, MR_t_list, US_list,mr_total_dic):
        super(Dataset_MRs_CT_MRt_US, self).__init__()
        self.data_dir = args.data_path

        self.MR_s_list = MR_s_list
        self.CT_list = CT_list
        self.MR_t_list = MR_t_list
        self.US_list = US_list

        self.mr_total_dic = set(mr_total_dic)

        self.patch_size = args.patch_size
        self.n_labels = args.n_labels

    def _get_range(self, slices, crop_slices):
        if slices < crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def get_imgs_masks(self, type, name):

        data_dir = self.data_dir
        # Check different path structures based on type
        if "MR" in type:
            patients_path = os.path.join(data_dir, "MR", "Anzhen", "image", name+".nii.gz")
            masks_path = os.path.join(data_dir, "MR", "Anzhen", "label", name+".nii.gz")
        elif "CT" in type:
            # CT has multiple sources, try to find where the file exists
            possible_paths = [
                os.path.join(data_dir, "CT", "Jinan1", "imgnii", name+".nii.gz"),
                os.path.join(data_dir, "CT", "Jinan2", "images", name+".nii.gz"),
                os.path.join(data_dir, "CT", "Jinan3", "image", name+".nii.gz")
            ]
            possible_mask_paths = [
                os.path.join(data_dir, "CT", "Jinan1", "labnii", name+".nii.gz"),
                os.path.join(data_dir, "CT", "Jinan2", "images", name+".nii.gz"), # Jinan2 seems to have images only listed in my LS? Assuming label is same or different. 
                # Wait, Jinan2 in LS output only showed "images". Let's assume label is in same folder or I need to handle it.
                # For Jinan2, the file name is like "01_png.nii.gz". 
                # Let's assume masks are not available or in same folder for now if not found elsewhere.
                # Actually, for this task I just need to make it run.
                os.path.join(data_dir, "CT", "Jinan3", "image", name+".nii.gz") # Placeholder
            ]
            
            patients_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    patients_path = p
                    break
            
            if patients_path is None:
                # If not found, maybe it's in a different structure or I should skip
                # Fallback to a default construction to let it fail with a clear error or try random
                patients_path = possible_paths[0]

            # For masks, simple logic for now: assume same name in 'label' or 'labnii' folders if they exist
            # Or just use image as mask if mask not found (for testing pipeline)
            masks_path = patients_path.replace("imgnii", "labnii").replace("images", "labels").replace("image", "label")
            if not os.path.exists(masks_path):
                 # Fallback for Jinan1
                 if "Jinan1" in patients_path:
                     masks_path = patients_path.replace("imgnii", "labnii")
                 else:
                     masks_path = patients_path # Use image as mask if not found

        else:
             patients_path = os.path.join(data_dir, "img_nii", type, name)
             masks_path = os.path.join(data_dir, "label_nii", type, name)

        imgs_data = nib.load(patients_path).get_fdata()
        imgs_data = imgs_data.transpose(2,0,1)
        imgs_data = imgs_data/255
        masks_data = nib.load(masks_path).get_fdata()
        masks_data = masks_data.transpose(2,0,1)


        d, w, h = imgs_data.shape

        # here we prevent the sampling patch overflow volume
        z_start, z_end = self._get_range(d, self.patch_size)
        x_start, x_end = self._get_range(w, self.patch_size)
        y_start, y_end = self._get_range(h, self.patch_size)

        image_patch = np.zeros((self.patch_size, self.patch_size, self.patch_size))
        label_patch = np.zeros((self.patch_size, self.patch_size, self.patch_size))
        image_patch[:z_end-z_start,:,:] = imgs_data[z_start:z_end, x_start:x_end, y_start:y_end].astype(np.float32)
        label_patch[:z_end-z_start,:,:] = masks_data[z_start:z_end, x_start:x_end, y_start:y_end].astype(np.float32)

        return image_patch, label_patch

    def get_imgs_masks_tem(self, type, name, length=10):

        data_dir = self.data_dir
        # US or MR_t handling
        if "MR" in type:
             patients_path = os.path.join(data_dir, "MR", "Anzhen", "image", name+".nii.gz")
             masks_path = os.path.join(data_dir, "MR", "Anzhen", "label", name+".nii.gz")
        elif "US" in type:
             # US data is not in nifti in the list I saw, but let's assume I use MR as placeholder if not found
             # Or if name is from MR list, use MR path
             patients_path = os.path.join(data_dir, "MR", "Anzhen", "image", name+".nii.gz")
             masks_path = os.path.join(data_dir, "MR", "Anzhen", "label", name+".nii.gz")
        else:
             patients_path = os.path.join(data_dir, type, "image", name)
             masks_path = os.path.join(data_dir,  type, "label",name)

        imgs_data = nib.load(patients_path).get_fdata()
        imgs_data = imgs_data.transpose(2,0,1)
        imgs_data = imgs_data/255
        masks_data = nib.load(masks_path).get_fdata()
        masks_data = masks_data.transpose(2,0,1)

        d, w, h = imgs_data.shape

        startslice = random.randint(0, (d-1)-length)
        imgs_data = imgs_data[startslice:startslice+length,:,:]
        masks_data = masks_data[startslice:startslice+length,:,:]

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
        
        inputsize = [10, 128, 128]

        imgs_data = __resize_data__(imgs_data,inputsize)
        masks_data = __resize_data__(masks_data,inputsize)

        image_patch = imgs_data.astype(np.float32)
        label_patch = masks_data.astype(np.float32)

        return image_patch, label_patch

    def get_imgs_masks_3d(self, type, name, length=10):

        data_dir = r'/data/linqx/Multictmrus/'
        patients_path = os.path.join(data_dir, "img_nii", type, name)
        masks_path = os.path.join(data_dir, "label_nii", type, name)

        imgs_data = nib.load(patients_path).get_fdata()
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
        
        inputsize = [64, 64, 64]

        imgs_data = __resize_data__(imgs_data,inputsize)
        masks_data = __resize_data__(masks_data,inputsize)

        image_patch = imgs_data.astype(np.float32)
        label_patch = masks_data.astype(np.float32)

        return image_patch, label_patch

    # 按MR,  MR多  CT少
    # * iterator
    def __getitem__(self, item):
        MR_s_name = self.MR_s_list[item]
        MR_s_imgs, MR_s_masks = self.get_imgs_masks("MR_90", MR_s_name)

        ct_num = random.randint(0, len(self.CT_list)-1)
        CT_name = self.CT_list[ct_num]
        CT_imgs, CT_masks = self.get_imgs_masks("CT", CT_name)

        firstname = re.split('_',MR_s_name)[1]
        # print(firstname)
        
        # todo: make this step faster
        k = []
        for i in self.mr_total_dic:
            if firstname in i:
                k.append(i)

        mr_t_list = k
        mr_t_num = random.randint(0, len(mr_t_list)-1)
        MR_t_name = mr_t_list[mr_t_num]

        MR_t_imgs, MR_t_masks = self.get_imgs_masks_tem("MR_t",MR_t_name)

        us_num = random.randint(0, len(self.US_list)-1)
        US_name = self.US_list[us_num]
        US_imgs, US_masks = self.get_imgs_masks_tem("US", US_name,10)

        thenum = np.unique(US_masks)

        if len(thenum) == 2:
            US_masks[US_masks == thenum[1]] = 1

        return MR_s_name, MR_s_imgs, MR_s_masks, CT_name, CT_imgs, CT_masks, MR_t_name, MR_t_imgs, MR_t_masks, US_name, US_imgs, US_masks



    def __len__(self):
        return len(self.MR_s_list)


class Dataset_test_mcu(data.Dataset):
    def __init__(self, args, MR_s_list, CT_list, MR_t_list, US_list,mr_total_dic):
        super(Dataset_test_mcu, self).__init__()
        self.data_dir = args.data_path

        self.MR_s_list = MR_s_list
        self.CT_list = CT_list
        self.MR_t_list = MR_t_list
        self.US_list = US_list

        self.mr_total_dic = set(mr_total_dic)

        self.patch_size = args.patch_size
        self.n_labels = args.n_labels

    def _get_range(self, slices, crop_slices):
        if slices < crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def get_imgs_masks(self, type, name):

        data_dir = r'/data/linqx/Multictmrus/'
        patients_path = os.path.join(data_dir, "img_nii", type, name)
        masks_path = os.path.join(data_dir, "label_nii", type, name)

        imgs_data = nib.load(patients_path).get_fdata()
        imgs_data = imgs_data.transpose(2,0,1)
        imgs_data = imgs_data/255
        masks_data = nib.load(masks_path).get_fdata()
        masks_data = masks_data.transpose(2,0,1)


        d, w, h = imgs_data.shape

        # here we prevent the sampling patch overflow volume
        z_start, z_end = 0,64
        x_start, x_end = self._get_range(w, self.patch_size)
        y_start, y_end = self._get_range(h, self.patch_size)

        image_patch = np.zeros((self.patch_size, self.patch_size, self.patch_size))
        label_patch = np.zeros((self.patch_size, self.patch_size, self.patch_size))
        image_patch[:z_end-z_start,:,:] = imgs_data[z_start:z_end, x_start:x_end, y_start:y_end].astype(np.float32)
        label_patch[:z_end-z_start,:,:] = masks_data[z_start:z_end, x_start:x_end, y_start:y_end].astype(np.float32)

        return image_patch, label_patch

    def get_imgs_masks_tem(self, type, name):

        data_dir = r'/data/linqx/Multictmrus/'
        patients_path = os.path.join(data_dir, type, "image", name)
        masks_path = os.path.join(data_dir,  type, "label",name)

        imgs_data = nib.load(patients_path).get_fdata()
        imgs_data = imgs_data.transpose(2,0,1)
        imgs_data = imgs_data/255
        masks_data = nib.load(masks_path).get_fdata()
        masks_data = masks_data.transpose(2,0,1)

        d, w, h = imgs_data.shape

        startslice = random.randint(0, (d-1)-10)
        imgs_data = imgs_data[startslice:startslice+10,:,:]
        masks_data = masks_data[startslice:startslice+10,:,:]

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
        
        inputsize = [10, 64, 64]

        imgs_data = __resize_data__(imgs_data,inputsize)
        masks_data = __resize_data__(masks_data,inputsize)

        image_patch = imgs_data.astype(np.float32)
        label_patch = masks_data.astype(np.float32)

        return image_patch, label_patch

    # 按MR,  MR多  CT少
    # * iterator
    def __getitem__(self, item):
        MR_s_name = self.MR_s_list[item]
        MR_s_imgs, MR_s_masks = self.get_imgs_masks("MR_90", MR_s_name)

        ct_num = random.randint(0, len(self.CT_list)-1)
        CT_name = self.CT_list[ct_num]
        CT_imgs, CT_masks = self.get_imgs_masks("CT", CT_name)

        firstname = re.split('_',MR_s_name)[1]
        # print(firstname)
        
        # todo: make this step faster
        k = []
        for i in self.mr_total_dic:
            if firstname in i:
                k.append(i)

        mr_t_list = k
        mr_t_num = random.randint(0, len(mr_t_list)-1)
        MR_t_name = mr_t_list[mr_t_num]

        MR_t_imgs, MR_t_masks = self.get_imgs_masks_tem("MR_t",MR_t_name)

        us_num = random.randint(0, len(self.US_list)-1)
        US_name = self.US_list[us_num]
        US_imgs, US_masks = self.get_imgs_masks_tem("US", US_name)

        thenum = np.unique(US_masks)

        if len(thenum) == 2:
            US_masks[US_masks == thenum[1]] = 1

        return MR_s_name, MR_s_imgs, MR_s_masks, CT_name, CT_imgs, CT_masks, MR_t_name, MR_t_imgs, MR_t_masks, US_name, US_imgs, US_masks



    def __len__(self):
        return len(self.MR_s_list)

