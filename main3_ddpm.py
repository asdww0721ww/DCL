import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional
import torch.optim as optim
import sys,random
import os
import datetime
import argparse
import torch.utils.data as data
import utils
import numpy as np
# from loaders.dataset_loader import Dataset_MRCT
from loaders.data3D_loader import Dataset_MRCT,Dataset_MRs_CT_MRt_US
from layers.losses_ops import AverageMeter
from layers.losses import *
from utils.data_utils import split_data
import models
from torch.nn import functional as F
from tqdm import tqdm
import monai

#hard-wire the gpu id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def parse_arguments(args):
    usage_text = (
        "SDNet Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-e','--epochs', type= int, default=100, help='Number of epochs')
    parser.add_argument('-batch_size','--batch_size', type= int, default=1, help='Number of inputs per batch')
    parser.add_argument('-patch_size','--patch_size', type= int, default=64, help='patch_size')
    parser.add_argument('-n_labels','--n_labels', type= int, default=3, help='n_labels')
    parser.add_argument('-n','--name', type=str, default='mrct_5', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-mn','--model_name', type=str, default='Muti_SDNet', help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help= 'Path to save model checkpoints')
    parser.add_argument('--dropout_p', type=float, default = 0.3, help='dropout')
    parser.add_argument('-hidden_size', dest='hidden_size', default=256, type=int)
    parser.add_argument('-skip_mode', dest='skip_mode', default = 'concat',
                        choices=['sum','concat','mul','none'])
    parser.add_argument("--anatomy_factors", type=int, default=3, help = 'Number of anatomy factors to encode')
    parser.add_argument("--modality_factors", type=int, default=3, help = 'Number of modality factors to encode')
    parser.add_argument("--charbonnier", type=int, default=0, help = 'Choose Charbonnier penalty for the reconstruction loss')
    # note: change
    parser.add_argument("--data_path", type=str, default='/data/gsd/data_3D', help='Path to ACDC dataset')
    #regularizers weights
    parser.add_argument("--kl_w", type=float, default=0.01, help = 'KL divergence loss weight')
    parser.add_argument("--regress_w", type=float, default=1.0, help = 'Regression loss weight')
    parser.add_argument("--focal_w", type=float, default=0.0, help = 'Focal loss weight')
    parser.add_argument("--dice_w", type=float, default=10.0, help = 'Dice loss weight')
    parser.add_argument("--reco_w", type=float, default=1.0, help = 'Reconstruction loss weight')
    #hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers' ,type= int, default = 0, help='Number of workers to use for dataload')
    #visdom params
    parser.add_argument('-d','--disp_iters', type=int, default=1, help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    

    return parser.parse_known_args(args)

if __name__ == "__main__":

    args, uknown = parse_arguments(sys.argv)
    #create and init device
    print('{} | Torch Version: {}'.format(datetime.datetime.now(), torch.__version__))
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    device = torch.device('cuda:{}' .format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else 'cpu')
    print('Training {0} for {1} epochs using a batch size of {2} on {3}'.format(args.name, args.epochs, args.batch_size, device))

    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.set_device(int(args.gpu))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

    #Model selection and initialization
    model_params = {
        'width': 80,
        'height': 80,
        'ndf': 64,
        'norm': "batchnorm",
        'upsample': "nearest",
        'num_classes': 3, #background as extra class
        'anatomy_out_channels': args.anatomy_factors,
        'z_length': args.modality_factors,
        'num_mask_channels': 8,

    }
    model = models.get_model(args, model_params)
    num_params = utils.count_parameters(model)
    print('Model Parameters: ', num_params)
    models.initialize_weights(model, args.weight_init)
    model.to(device)
    # model = nn.DataParallel(model,gpus)

    # note: get the training data
    # todo: rewrite to direct reading mode
    # MR_train_list, MR_val_list, _, CT_train_list, CT_val_list, _ = split_data(args.data_path)

    with open(r'd:\CodeProjects\HDiffiuCanet\datalist\new_mr_s_list.txt','r') as f:
        MR_train_list = f.readlines()
    
    for idx,i in enumerate(MR_train_list):
        MR_train_list[idx] = MR_train_list[idx].strip('\n')

    with open(r'd:\CodeProjects\HDiffiuCanet\datalist\new_ct_list.txt','r') as f:
        CT_train_list = f.readlines()
    
    for idx,i in enumerate(CT_train_list):
        CT_train_list[idx] = CT_train_list[idx].strip('\n')

    with open(r'd:\CodeProjects\HDiffiuCanet\datalist\new_mr_s_list.txt','r') as f:
        MR_val_list = f.readlines()
    
    for idx,i in enumerate(MR_val_list):
        MR_val_list[idx] = MR_val_list[idx].strip('\n')

    with open(r'd:\CodeProjects\HDiffiuCanet\datalist\new_ct_list.txt','r') as f:
        CT_val_list = f.readlines()
    
    for idx,i in enumerate(CT_val_list):
        CT_val_list[idx] = CT_val_list[idx].strip('\n')

    with open(r'd:\CodeProjects\HDiffiuCanet\datalist\new_us_list.txt','r') as f:
        US_train_list = f.readlines()
    
    for idx,i in enumerate(US_train_list):
        US_train_list[idx] = US_train_list[idx].strip('\n')

    with open(r'd:\CodeProjects\HDiffiuCanet\datalist\new_us_list.txt','r') as f:
        US_val_list = f.readlines()
    
    for idx,i in enumerate(US_val_list):
        US_val_list[idx] = US_val_list[idx].strip('\n')
    
    with open(r'd:\CodeProjects\HDiffiuCanet\datalist\new_mr_t_list.txt','r') as f:
        MR_t_list = f.readlines()
    
    for idx,i in enumerate(MR_t_list):
        MR_t_list[idx] = MR_t_list[idx].strip('\n')

    #
    # Overwrite data_path in args to point to local dataset root for loader
    args.data_path = r'd:\CodeProjects\HDiffiuCanet\Dataset'
    
    data_train = Dataset_MRs_CT_MRt_US(args, MR_train_list, CT_train_list,MR_t_list, US_train_list, MR_t_list)
    dataset_train = data.DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)

    data_val = Dataset_MRs_CT_MRt_US(args, MR_val_list, CT_val_list, MR_t_list, US_val_list, MR_t_list)
    dataset_val = data.DataLoader(dataset=data_val, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)


    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.Tensor

    #loss initialization
    # todo : add loss for temporal pipline
    L1Loss = nn.L1Loss().to(device)
    MSELoss = nn.MSELoss().to(device)
    DiceLoss = DiceLoss().to(device)
    Dice = Dice().to(device)
    Dice3D = Dice3D().to(device)
    CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
    mdiceloss = monai.losses.DiceLoss()
    # BinaryCrossEntropyLoss = F.binary_cross_entropy_with_logits

    #optimizer initialization
    # optimizer = optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=args.learning_rate)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,weight_decay=1e-4,momentum=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

    #loss initialization
    total_loss = AverageMeter()
    running_reco1_loss = AverageMeter()
    running_reco2_loss = AverageMeter()
    running_dice1_loss = AverageMeter()
    running_dice2_loss = AverageMeter()
    running_dice3_loss = AverageMeter()
    running_dice4_loss = AverageMeter()

    #validation loss initialization
    val_running_dice = AverageMeter()

    global_iterations = 0
    val_dice_best = 0.0

    #train/val process
    for epoch in range(args.epochs):
        model.train()
        # mod: rewrite output of iterator
        for MR_s_name, MR_s_imgs, MR_s_masks, CT_name, CT_imgs, CT_masks, MR_t_name, MR_t_imgs, MR_t_masks, US_name, US_imgs, US_masks in tqdm(dataset_train):

            # mod : add new to device
            MR_s_imgs = MR_s_imgs.to(device)
            MR_s_masks = MR_s_masks.to(device)
            CT_imgs = CT_imgs.to(device)
            CT_masks = CT_masks.to(device)
            MR_t_imgs = MR_t_imgs.to(device)
            MR_t_masks = MR_t_masks.to(device)
            US_imgs = US_imgs.to(device)
            US_masks = US_masks.to(device)

            optimizer.zero_grad()

            # print(US_imgs.shape)
            thi = random.randint(0,MR_s_imgs.shape[1]-1)
            thi2 = random.randint(0,MR_t_imgs.shape[1]-1)
            # forward pass
            # mod: rewrite in & out
            a_out_all1, a_out_all2, seg_pred1, seg_pred2, a_out_all3, a_out_all4, seg_pred3, seg_pred4, a_out1,a_out2, a_out3, a_out4, z_out1, mu_out1, logvar_out1, z_out2, mu_out2, logvar_out2, loss1t2_mix, loss2t1_mix, fake_x1, fake_x2 , loss1_mix, loss2_mix,fake_x1t2 ,fake_x2t1,loss3_mix,loss4_mix,adv_x1,adv_x2,adv_fake_x1,adv_fake_x2,adv_fake_x1_m2,adv_fake_x2_m1,adv_x3,adv_x4,adv_fake_x3,adv_fake_x4,fake_x3,fake_x4 = model(MR_s_imgs, CT_imgs,MR_t_imgs, US_imgs,MR_s_masks,CT_masks,MR_t_masks,US_masks,thi,thi2,'training')
            
            #loss computation

            # cycle_consistency_loss
            cycle_consistency_loss_1 = L1Loss(
                fake_x1.view(-1, MR_s_imgs.shape[-2], MR_s_imgs.shape[-1]),MR_s_imgs.view(-1, MR_s_imgs.shape[-2], MR_s_imgs.shape[-1]).float()
                )

            cycle_consistency_loss_2 = L1Loss(
                fake_x2.view(-1, CT_imgs.shape[-2], CT_imgs.shape[-1]),CT_imgs.view(-1, CT_imgs.shape[-2], CT_imgs.shape[-1]).float()
                )

            # mod: con loss
            cycle_consistency_loss_3 = L1Loss(
                fake_x3.view(-1, MR_t_imgs.shape[-2], MR_t_imgs.shape[-1]),MR_t_imgs.view(-1, MR_t_imgs.shape[-2], MR_t_imgs.shape[-1]).float()
                )

            cycle_consistency_loss_4 = L1Loss(
                fake_x4.view(-1, US_imgs.shape[-2], US_imgs.shape[-1]),US_imgs.view(-1, US_imgs.shape[-2], US_imgs.shape[-1]).float()
                )

            # gan loss
            T1 = MR_s_imgs.shape[0] * MR_s_imgs.shape[1]
            T2 = CT_imgs.shape[0] * CT_imgs.shape[1]
            gan_loss_1 = MSELoss(adv_fake_x1_m2, Variable(Tensor(T1).fill_(0.0), requires_grad=False))
            gan_loss_2 = MSELoss(adv_fake_x2_m1, Variable(Tensor(T2).fill_(0.0), requires_grad=False))

            gan_loss_1_0 = MSELoss(adv_x1, Variable(Tensor(T1).fill_(1.0), requires_grad=False))
            gan_loss_2_0 = MSELoss(adv_x2, Variable(Tensor(T2).fill_(1.0), requires_grad=False))

            gan_loss_3_0 = MSELoss(adv_x3, Variable(Tensor(T1).fill_(1.0), requires_grad=False))
            gan_loss_4_0 = MSELoss(adv_x4, Variable(Tensor(T2).fill_(1.0), requires_grad=False))

            # seg loss for mr_t and us
            # note: use dice loss
            # note: the shape has been turned to 
            target_seg1_1 = seg_pred3[:,1,:,:,:]
            target_seg1_2 = seg_pred3[:,2,:,:,:]

            registered_seg1_1 = MR_t_masks.clone()
            registered_seg1_2 = MR_t_masks.clone()
            registered_seg1_1[MR_t_masks != 1] = 0
            registered_seg1_2[MR_t_masks != 2] = 0
            registered_seg1_2[MR_t_masks == 2] = 1

            target_seg2_1 = seg_pred4[:,1,:,:,:]
            target_seg2_2 = seg_pred4[:,2,:,:,:]

            registered_seg2_1 = US_masks.clone()
            registered_seg2_2 = US_masks.clone()        
            registered_seg2_1[US_masks != 1] = 0
            registered_seg2_2[US_masks != 2] = 0
            registered_seg2_2[US_masks == 2] = 1

            smooth = 1e-5

            inter1_1 = (registered_seg1_1*target_seg1_1).sum()
            union1_1 = torch.sum(registered_seg1_1)+torch.sum(target_seg1_1)

            dice1_1 = 2. * (inter1_1 + smooth) / (union1_1 + smooth)

            inter1_2 = (registered_seg1_2*target_seg1_2).sum()
            union1_2 = torch.sum(registered_seg1_2)+torch.sum(target_seg1_2)

            dice1_2 = 2. * (inter1_2 + smooth) / (union1_2 + smooth)

            # note: use 1- to get the loss; the origin is the index 

            seg_loss_3 = 1 - ((dice1_1 + dice1_2)/2)

            inter2_1 = (registered_seg2_1*target_seg2_1).sum()
            union2_1 = torch.sum(registered_seg2_1)+torch.sum(target_seg2_1)

            dice2_1 = 2. * (inter2_1 + smooth) / (union2_1 + smooth)
            
            inter2_2 = (registered_seg2_2*target_seg2_2).sum()
            union2_2 = torch.sum(registered_seg2_2)+torch.sum(target_seg2_2)

            dice2_2 = 2. * (inter2_2 + smooth) / (union2_2 + smooth)

            seg_loss_4 = 1 - ((dice2_1 + dice2_2)/2)
            # seg_loss_4 = 1 - ((dice2_1 + dice2_2 + dice2_3)/3)
            # mod to 1 label
            # seg_loss_4 = 1 - dice2_1

            # note: dont change this part
            # seg  DiceLoss
            MR_gt_onehot = MR_s_masks.unsqueeze(1)
            MR_gt_onehot = torch.zeros((MR_s_masks.shape[0], model_params['num_classes'], MR_s_masks.shape[-3], MR_s_masks.shape[-2], MR_s_masks.shape[-1])).to(device)
            MR_gt_onehot[:,0,:,:,:][MR_s_masks==0] = 1
            MR_gt_onehot[:,1,:,:,:][MR_s_masks==1] = 1
            MR_gt_onehot[:,2,:,:,:][MR_s_masks==2] = 1

            CT_gt_onehot = CT_masks.unsqueeze(1)
            CT_gt_onehot = torch.zeros((CT_masks.shape[0], model_params['num_classes'], CT_masks.shape[-3], CT_masks.shape[-2], CT_masks.shape[-1])).to(device)
            CT_gt_onehot[:,0,:,:,:][CT_masks==0] = 1
            CT_gt_onehot[:,1,:,:,:][CT_masks==1] = 1
            CT_gt_onehot[:,2,:,:,:][CT_masks==2] = 1
        

            if np.where(MR_gt_onehot.cpu().numpy() == 1)[0].shape[0] == 0:
                weight = 1
            else:
                weight = args.batch_size * args.patch_size * args.patch_size * args.patch_size / np.where(MR_gt_onehot.cpu().numpy() == 1)[0].shape[0]

            weight = torch.FloatTensor([weight]).to(device)

            loss1_MR = F.binary_cross_entropy_with_logits(seg_pred1, MR_gt_onehot, pos_weight=weight)
            loss2_MR = F.binary_cross_entropy_with_logits(a_out_all1[1], MR_gt_onehot, pos_weight=weight)
            loss3_MR = F.binary_cross_entropy_with_logits(a_out_all1[2], MR_gt_onehot, pos_weight=weight)
            loss4_MR = F.binary_cross_entropy_with_logits(a_out_all1[3], MR_gt_onehot, pos_weight=weight)
            loss5_MR = F.binary_cross_entropy_with_logits(a_out_all1[4], MR_gt_onehot, pos_weight=weight)
            loss6_MR = F.binary_cross_entropy_with_logits(a_out_all1[5], MR_gt_onehot, pos_weight=weight)
            loss7_MR = F.binary_cross_entropy_with_logits(a_out_all1[6], MR_gt_onehot, pos_weight=weight)
            loss8_MR = F.binary_cross_entropy_with_logits(a_out_all1[7], MR_gt_onehot, pos_weight=weight)
            loss9_MR = F.binary_cross_entropy_with_logits(a_out_all1[8], MR_gt_onehot, pos_weight=weight)
            
            seg_loss_1 = loss1_MR + 0.8 * loss2_MR + 0.7 * loss3_MR + 0.6 * loss4_MR + 0.5 * loss5_MR +  0.8 * loss6_MR + 0.7 * loss7_MR + 0.6 * loss8_MR + 0.5 * loss9_MR

            if np.where(CT_gt_onehot.cpu().numpy() == 1)[0].shape[0] == 0:
                weight = 1
            else:
                weight = args.batch_size * args.patch_size * args.patch_size * args.patch_size / np.where(CT_gt_onehot.cpu().numpy() == 1)[0].shape[0]

            weight = torch.FloatTensor([weight]).to(device)

            loss1_CT = F.binary_cross_entropy_with_logits(seg_pred2, CT_gt_onehot, pos_weight=weight)
            loss2_CT = F.binary_cross_entropy_with_logits(a_out_all2[1], CT_gt_onehot, pos_weight=weight)
            loss3_CT = F.binary_cross_entropy_with_logits(a_out_all2[2], CT_gt_onehot, pos_weight=weight)
            loss4_CT = F.binary_cross_entropy_with_logits(a_out_all2[3], CT_gt_onehot, pos_weight=weight)
            loss5_CT = F.binary_cross_entropy_with_logits(a_out_all2[4], CT_gt_onehot, pos_weight=weight)
            loss6_CT = F.binary_cross_entropy_with_logits(a_out_all2[5], CT_gt_onehot, pos_weight=weight)
            loss7_CT = F.binary_cross_entropy_with_logits(a_out_all2[6], CT_gt_onehot, pos_weight=weight)
            loss8_CT = F.binary_cross_entropy_with_logits(a_out_all2[7], CT_gt_onehot, pos_weight=weight)
            loss9_CT = F.binary_cross_entropy_with_logits(a_out_all2[8], CT_gt_onehot, pos_weight=weight)
            
            seg_loss_2 = loss1_CT + 0.8 * loss2_CT + 0.7 * loss3_CT + 0.6 * loss4_CT + 0.5 * loss5_CT + 0.8 * loss6_CT + 0.7 * loss7_CT + 0.6 * loss8_CT + 0.5 * loss9_CT


            diff_loss_1 = torch.log((loss1_mix.mean()/1000)+1)
            diff_loss_2 = torch.log((loss2_mix.mean()/1000)+1)
            diff_loss_1_0 = torch.log((loss1t2_mix.mean()/1000)+1)
            diff_loss_2_0 = torch.log((loss2t1_mix.mean()/1000)+1)

            diff_loss_3 = torch.log((loss3_mix.mean()/1000)+1)
            diff_loss_4 = torch.log((loss4_mix.mean()/1000)+1)

            if epoch >= 15:
                diff_para = 1e-04
            else:
                diff_para = 1e-07

            # print(diff_loss_1)
            # print(cycle_consistency_loss_1)
            # * total loss
            # mod : add temporal loss
            batch_loss = cycle_consistency_loss_1 + cycle_consistency_loss_2 + diff_para * diff_loss_1 + diff_para * diff_loss_2 + diff_para * diff_loss_2_0 + diff_para * diff_loss_1_0 + 15*seg_loss_1 + 15*seg_loss_2 + 4*15*seg_loss_3 + 6*15*seg_loss_4 + gan_loss_1+gan_loss_2+ gan_loss_1_0 + gan_loss_2_0 + diff_para * diff_loss_3+ diff_para * diff_loss_4 + gan_loss_3_0 + gan_loss_4_0 + cycle_consistency_loss_3 + cycle_consistency_loss_4
            
            #backprop and optimizer update
            batch_loss.backward()
            optimizer.step()

            # #logging
            # todo: add new logging item
            total_loss.update(batch_loss.item())
            running_reco1_loss.update(cycle_consistency_loss_1.item())
            running_reco2_loss.update(cycle_consistency_loss_2.item())
            running_dice1_loss.update(seg_loss_1.item())
            running_dice2_loss.update(seg_loss_2.item())
            running_dice3_loss.update(seg_loss_3.item())
            running_dice4_loss.update(seg_loss_4.item())

            if (epoch + 1) % args.disp_iters <= args.batch_size:
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']

                tqdm.write(f'Epoch: {epoch}, LR: {lr}\nReco_MR: {running_reco1_loss.avg}\nReco_CT: {running_reco2_loss.avg}\nDice_MR_s: {running_dice1_loss.avg}\nDice_CT: {running_dice2_loss.avg}\nDice_MR_t: {running_dice3_loss.avg}\nDice_US: {running_dice4_loss.avg}\nTotal average loss: {total_loss.avg}\n\n')

                total_loss.reset()
                running_reco1_loss.reset()
                running_reco1_loss.reset()
                running_dice1_loss.reset()
                running_dice2_loss.reset()
                running_dice3_loss.reset()
                running_dice4_loss.reset()

            #释放显存
            # note: this maybe shall not be used 
            # torch.cuda.empty_cache()

            global_iterations += args.batch_size

        # validation
        with torch.no_grad():
            model.eval()
            for MR_s_name, MR_s_imgs, MR_s_masks, CT_name, CT_imgs, CT_masks, MR_t_name, MR_t_imgs, MR_t_masks, US_name, US_imgs, US_masks in tqdm(dataset_val):


                MR_s_imgs = MR_s_imgs.to(device)
                MR_s_masks = MR_s_masks.to(device)
                CT_imgs = CT_imgs.to(device)
                CT_masks = CT_masks.to(device)
                MR_t_imgs = MR_t_imgs.to(device)
                MR_t_masks = MR_t_masks.to(device)
                US_imgs = US_imgs.to(device)
                US_masks = US_masks.to(device)

                a_out_all1, a_out_all2, seg_pred1, seg_pred2, a_out_all3, a_out_all4, seg_pred3, seg_pred4, a_out1,a_out2, a_out3, a_out4, z_out1, mu_out1, logvar_out1, z_out2, mu_out2, logvar_out2, x1t2_d, x2t1_d, a_out_all1_d,a_out_all2_d, a_out_all1t2_d, a_out_all2t1_d, z_out1_d, mu_out1_d, logvar_out1_d, z_out2_d, mu_out2_d, logvar_out2_d, z_out1t2_d, mu_out1t2_d, logvar_out1t2_d , z_out2t1_d, mu_out2t1_d, logvar_out2t1_d,z_out3_d, mu_out3_d, logvar_out3_d, z_out4_d, mu_out4_d, logvar_out4_d = model(MR_s_imgs, CT_imgs,MR_t_imgs, US_imgs,MR_s_masks,CT_masks,MR_t_masks,US_masks,thi, thi2,'val')

                #dice loss of t data
                target_seg1_1 = seg_pred3[:,1,:,:,:]
                target_seg1_2 = seg_pred3[:,2,:,:,:]
                #target_seg1_3 = seg_pred3[:,1,:,:,:]
                #target_seg1_3[seg_pred3[:,2,:,:,:] == 1] = 1
                
                registered_seg1_1 = MR_t_masks.clone()
                registered_seg1_2 = MR_t_masks.clone()
                registered_seg1_3 = MR_t_masks.clone()
                registered_seg1_1[MR_t_masks != 1] = 0
                registered_seg1_2[MR_t_masks != 2] = 0
                registered_seg1_2[MR_t_masks == 2] = 1
                #registered_seg1_3[MR_t_masks == 2] = 1

                target_seg2_1 = seg_pred4[:,1,:,:,:]
                target_seg2_2 = seg_pred4[:,2,:,:,:]
                #target_seg2_3 = seg_pred4[:,1,:,:,:]
                #target_seg2_3[seg_pred4[:,2,:,:,:] == 1] = 1

                registered_seg2_1 = US_masks.clone()
                registered_seg2_2 = US_masks.clone()
                registered_seg2_3 = US_masks.clone()
                registered_seg2_1[US_masks != 1] = 0
                registered_seg2_2[US_masks != 2] = 0
                registered_seg2_2[US_masks == 2] = 1
                #registered_seg2_3[US_masks == 2] = 1

                smooth = 1e-5

                inter1_1 = (registered_seg1_1*target_seg1_1).sum()
                union1_1 = torch.sum(registered_seg1_1)+torch.sum(target_seg1_1)

                dice1_1 = 2. * (inter1_1 + smooth) / (union1_1 + smooth)

                #print(dice1_1)

                inter1_2 = (registered_seg1_2*target_seg1_2).sum()
                union1_2 = torch.sum(registered_seg1_2)+torch.sum(target_seg1_2)

                dice1_2 = 2. * (inter1_2 + smooth) / (union1_2 + smooth)

                #inter1_3 = (registered_seg1_3*target_seg1_3).sum()
                #union1_3 = torch.sum(registered_seg1_3)+torch.sum(target_seg1_3)

                #dice1_3 = 2. * (inter1_3 + smooth) / (union1_3 + smooth)

                # note : not use 1- to get the value
                #seg_loss_3 = (dice1_1 + dice1_2 + dice1_3)/3
                seg_loss_3 = dice1_1 + dice1_2


                inter2_1 = (registered_seg2_1*target_seg2_1).sum()
                union2_1 = torch.sum(registered_seg2_1)+torch.sum(target_seg2_1)

                dice2_1 = 2. * (inter2_1 + smooth) / (union2_1 + smooth)
                
                inter2_2 = (registered_seg2_2*target_seg2_2).sum()
                union2_2 = torch.sum(registered_seg2_2)+torch.sum(target_seg2_2)

                dice2_2 = 2. * (inter2_2 + smooth) / (union2_2 + smooth)

                #inter2_3 = (registered_seg2_3*target_seg2_3).sum()
                #union2_3 = torch.sum(registered_seg2_3)+torch.sum(target_seg2_3)

                #dice2_3 = 2. * (inter2_3 + smooth) / (union2_3 + smooth)

                #seg_loss_4 = (dice2_1 + dice2_2 + dice2_3)/3
                seg_loss_4 = dice2_1 + dice2_2

                #dice score computation
                MR_gt_onehot = torch.zeros((MR_s_masks.shape[0], model_params['num_classes'], MR_s_masks.shape[-3],MR_s_masks.shape[-2], MR_s_masks.shape[-1])).to(device)
                MR_gt_onehot[:, 0, :, :, :][MR_s_masks == 0] = 1
                MR_gt_onehot[:, 1, :, :, :][MR_s_masks == 1] = 1
                MR_gt_onehot[:, 2, :, :, :][MR_s_masks == 2] = 1

                seg_1 = Dice3D(seg_pred1, MR_gt_onehot).to('cpu')

                CT_gt_onehot = torch.zeros((CT_masks.shape[0], model_params['num_classes'], CT_masks.shape[-3],CT_masks.shape[-2], CT_masks.shape[-1])).to(device)
                CT_gt_onehot[:, 0, :, :, :][CT_masks == 0] = 1
                CT_gt_onehot[:, 1, :, :, :][CT_masks == 1] = 1
                CT_gt_onehot[:, 2, :, :, :][CT_masks == 2] = 1
                
                seg_2 = Dice3D(seg_pred2, CT_gt_onehot).to('cpu')

                # terminal output 

                tqdm.write(f'Epoch: {epoch},TEST Dice1: {seg_1}, TEST Dice2: {seg_2}, TEST Dice3: {seg_loss_3}, TEST Dice4: {seg_loss_4}')

                #logging
                val_running_dice.update(seg_1.item() + seg_2.item() + seg_loss_3.to('cpu') + seg_loss_4.to('cpu'))

                # note: release gpu memory; temporaly ignored
                # torch.cuda.empty_cache()

            
            print("Epoch: {},\nDice: {}\n".format(epoch, val_running_dice.avg))


            #check for plateau
            val_dice_curr = val_running_dice.avg.item()
            scheduler.step(val_dice_curr)

            #save checkpoint for the best validation accuracy
            if val_dice_curr > val_dice_best:
                val_dice_best = val_dice_curr
                print("Epoch checkpoint")
                current_dir = os.getcwd()
                final_dir = args.save_path
                utils.save_network_state(model, model_params['width'], model_params['height'], model_params['ndf'], model_params['norm'], model_params['upsample'], model_params['num_classes'], model_params['anatomy_out_channels'], model_params['z_length'],model_params['num_mask_channels'], optimizer, epoch, args.name + "_model_state_epoch_" + str(epoch), final_dir)
            val_running_dice.reset()



