import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms as TF
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloader.dataset import BaseDataSets, RandomGenerator, Cutmix
from network.unet import UNet
from util.inference_double import test_single_volume, test_single_volume_ensemble
from util.pixel_contra_loss import PixelContrastLoss
from util import ramps

from network.unet_tf import UNetTF



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../dataset/ACDC', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='contra_tuned', help='experiment_name')
    parser.add_argument('--fold', type=str,
                        default='fold1', help='cross validation')
    parser.add_argument('--sup_type', type=str,
                        default='scribble', help='supervision type')
    parser.add_argument('--model', type=str,
                        default='swin_unet', help='model_name')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=60000, help='maximum epoch number to train')
    parser.add_argument('--save_fig_interval', type=int, default=1000, help='save fig interval')
    parser.add_argument('--val_interval', type=int, default=1000, help='val interval') # equals to the save_model_interval
    # parser.add_argument('--save_model_interval', type=int, default=1000, help='save model interval')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer (SGD or Adam)')
    parser.add_argument('--base_lr', type=float, default=0.03,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list, default=[224, 224],
                        help='patch size of network input')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--lamda_1', type=float, default=1, help='hyperparameter')
    parser.add_argument('--lamda_2', type=float, default=1, help='hyperparameter')
    parser.add_argument('--contrast_weight', type=float, default=0.1, help='hyperparameter')
    parser.add_argument('--contrast_iter', type=int, default=1000, help='hyperparameter')
    parser.add_argument('--tf_decoder_weight', type=float, default=0.4, help='hyperparameter')
    parser.add_argument('--het_weight', type=float, default=0.1, help='hyperparameter')
    parser.add_argument('--update_lr', action='store_false', help='whether update learning rate')
    args = parser.parse_args()

    return args


def _random_rotate(image, scribble, label):
    angle = float(torch.empty(1).uniform_(-20., 20.).item())
    image = TF.rotate(image, angle)
    scribble = TF.rotate(scribble, angle)
    label = TF.rotate(label, angle)
    return image, scribble, label

def get_current_het_weight(epoch, max_iter, het_weight):
        # het ramp-up from https://arxiv.org/abs/1610.02242
        return het_weight * ramps.sigmoid_rampup(epoch, max_iter // 100)

def main():
    # parse arguments
    args = parser_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

    return


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    memory = True
    pixel_update_freq = 10
    pixel_classes = 4
    memory_size = 500
    cls_weight = 1
    cls_criterion = torch.nn.CrossEntropyLoss()
    contrast_type = 'ugpcl'
    dim=256
    contrast_criterion = PixelContrastLoss()
    if memory:
        segment_queue_1 = torch.randn(pixel_classes, memory_size, dim)
        segment_queue_1 = nn.functional.normalize(segment_queue_1, p=2, dim=2)
        segment_queue_ptr_1 = torch.zeros(pixel_classes, dtype=torch.long)
        pixel_queue_1 = torch.zeros(pixel_classes, memory_size, dim)
        pixel_queue_1 = nn.functional.normalize(pixel_queue_1, p=2, dim=2)
        pixel_queue_ptr_1 = torch.zeros(pixel_classes, dtype=torch.long)

        segment_queue_2 = torch.randn(pixel_classes, memory_size, dim)
        segment_queue_2 = nn.functional.normalize(segment_queue_2, p=2, dim=2)
        segment_queue_ptr_2 = torch.zeros(pixel_classes, dtype=torch.long)
        pixel_queue_2 = torch.zeros(pixel_classes, memory_size, dim)
        pixel_queue_2 = nn.functional.normalize(pixel_queue_2, p=2, dim=2)
        pixel_queue_ptr_2 = torch.zeros(pixel_classes, dtype=torch.long)

    net_1 = UNetTF(in_channels=1, classes=num_classes).cuda()
    net_2 = UNetTF(in_channels=1, classes=num_classes).cuda()

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=TF.Compose([
        RandomGenerator(args.patch_size), Cutmix(prop_range=0.2)
    ]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path,
                          fold=args.fold, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_dataloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(db_val, batch_size=1, shuffle=False,
                                num_workers=1)
    net_1.train()
    net_2.train()

    if args.optimizer == 'SGD':
        optimizer_1 = optim.SGD(net_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
        optimizer_2 = optim.SGD(net_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    elif args.optimizer == 'Adam':
        optimizer_1 = optim.Adam(net_1.parameters(), lr=base_lr, weight_decay=0.0001)
        optimizer_2 = optim.Adam(net_2.parameters(), lr=base_lr, weight_decay=0.0001)

    partial_ce_loss = CrossEntropyLoss(ignore_index=4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(train_dataloader)))

    iter_num = 0
    max_epoch = max_iterations // len(train_dataloader) + 1

    save_fig = True
    update_model_for_dice = False
    update_model_for_hd95 = False
    best_dice = -1e8
    best_hd95 = 1e8

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_dataloader):

            volume_batch, scribble, cutmix_mask, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch[
                'mask'], sampled_batch['idx']
            volume_batch, scribble, cutmix_mask = volume_batch.cuda(), scribble.cuda(), cutmix_mask.cuda()
            volume_batch = TF.Resize([224,224])(volume_batch)
            scribble = TF.Resize([224,224])(scribble)
            cutmix_mask = TF.Resize([224,224])(cutmix_mask)

            ### first forward
            # net_1, cnn, supervised loss, no mix
            scribble_mask = (scribble != 4)
            outputs_1 = net_1(volume_batch)
            output_cnn_1 = outputs_1['seg']
            loss_sup_no_mix_cnn_1 = partial_ce_loss(output_cnn_1, scribble[:, 0].long())
            # net_1, tf, supervised loss, no mix
            output_tf_1 = outputs_1['seg_tf']
            loss_sup_no_mix_tf_1 = partial_ce_loss(output_tf_1, scribble[:, 0].long())

            # net_2, cnn, supervised loss, no mix
            scribble_mask = (scribble != 4)
            outputs_2 = net_2(volume_batch)
            output_cnn_2 = outputs_2['seg']
            loss_sup_no_mix_cnn_2 = partial_ce_loss(output_cnn_2, scribble[:, 0].long())
            # net_2, tf, supervised loss, no mix
            output_tf_2 = outputs_2['seg_tf']
            loss_sup_no_mix_tf_2 = partial_ce_loss(output_tf_2, scribble[:, 0].long())

            # perform cutmix
            rand_index_1 = torch.randperm(volume_batch.size()[0]).cuda()
            rand_index_2 = torch.randperm(volume_batch.size()[0]).cuda()

            # shuffle
            image_shuffle_1 = volume_batch[rand_index_1]
            scribble_shuffle_1 = scribble[rand_index_1]
            cutmix_mask_shuffle_1 = cutmix_mask[rand_index_1]

            image_shuffle_2 = volume_batch[rand_index_2]
            scribble_shuffle_2 = scribble[rand_index_2]
            cutmix_mask_shuffle_2 = cutmix_mask[rand_index_2]

            with torch.no_grad():
                output_unsup_cnn_1 = output_cnn_1[rand_index_1]
                output_unsup_tf_1 = output_tf_1[rand_index_1]
                output_unsup_cnn_2 = output_cnn_2[rand_index_2]
                output_unsup_tf_2 = output_tf_2[rand_index_2]

            output_unsup_soft_cnn_1 = F.softmax(output_unsup_cnn_1, dim=1)
            output_unsup_soft_tf_1 = F.softmax(output_unsup_tf_1, dim=1)
            output_unsup_soft_cnn_2 = F.softmax(output_unsup_cnn_2, dim=1)
            output_unsup_soft_tf_2 = F.softmax(output_unsup_tf_2, dim=1)

            ### generate strong augmented image, pseudo label and mixed scribble annotation.
            # augmented image for net_1
            img_mixed_1 = cutmix_mask_shuffle_1 * image_shuffle_1 + \
                (1 - cutmix_mask_shuffle_1) * image_shuffle_2
            pseudo_label_cnn_1 = cutmix_mask_shuffle_1 * output_unsup_soft_cnn_1 + \
                (1 - cutmix_mask_shuffle_1) * output_unsup_soft_cnn_2
            pseudo_label_tf_1 = cutmix_mask_shuffle_1 * output_unsup_soft_tf_1 + \
                (1 - cutmix_mask_shuffle_1) * output_unsup_soft_tf_2
            scribble_mixed_1 = cutmix_mask_shuffle_1 * scribble_shuffle_1 + \
                (1 - cutmix_mask_shuffle_1) * scribble_shuffle_2

            # augmented image for net_2
            img_mixed_2 = cutmix_mask_shuffle_2 * image_shuffle_2 + \
                (1 - cutmix_mask_shuffle_2) * image_shuffle_1
            pseudo_label_cnn_2 = cutmix_mask_shuffle_2 * output_unsup_soft_cnn_2 + \
                (1 - cutmix_mask_shuffle_2) * output_unsup_soft_cnn_1
            pseudo_label_tf_2 = cutmix_mask_shuffle_2 * output_unsup_soft_tf_2 + \
                (1 - cutmix_mask_shuffle_2) * output_unsup_soft_tf_1
            scribble_mixed_2 = cutmix_mask_shuffle_2 * scribble_shuffle_2 + \
                (1 - cutmix_mask_shuffle_2) * scribble_shuffle_1


            output_soft_cnn_1 = torch.softmax(output_cnn_1, dim=1)
            output_soft_tf_1 = torch.softmax(output_tf_1, dim=1)
            output_soft_cnn_2 = torch.softmax(output_cnn_2, dim=1)
            output_soft_tf_2 = torch.softmax(output_tf_2, dim=1)
            

            #=================== contrast loss (below) ========================
            
            # contrast for net_1
            if iter_num > args.contrast_iter and args.contrast_weight > 0.:

                # queue = torch.cat((segment_queue, pixel_queue), dim=1) if memory else None
                queue_1 = segment_queue_1 if memory else None
                seg_mean = torch.mean(torch.stack([output_soft_cnn_1, output_soft_tf_1]), dim=0)
                uncertainty = -1.0 * torch.sum(seg_mean * torch.log(seg_mean + 1e-6), dim=1, keepdim=True)
                threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
                uncertainty_mask = (uncertainty > threshold)
                mean_preds = torch.argmax(F.softmax(seg_mean, dim=1).detach(), dim=1, keepdim=True).float()
                certainty_pseudo = mean_preds.clone()
                certainty_pseudo[uncertainty_mask] = -1
                preds = torch.argmax(output_soft_cnn_1, dim=1, keepdim=True).to(torch.float)
                certainty_pseudo[scribble_mask] = scribble[scribble_mask].to(torch.float)

                loss_contra_1 = contrast_criterion(outputs_1['embed'], certainty_pseudo, preds, queue=queue_1)

                writer.add_scalar('train/uncertainty_rate', torch.sum(uncertainty_mask == True) / \
                                                (torch.sum(uncertainty_mask == True) + torch.sum(
                                                    uncertainty_mask == False)), iter_num)

                if memory: # update the queue
                    _keys, _labels = outputs_1['embed'].detach(), certainty_pseudo.detach()
                    batch_sz, feat_dim = _keys.shape[0], _keys.shape[1]
                    _labels = torch.nn.functional.interpolate(_labels, (_keys.shape[2], _keys.shape[3]), mode='nearest')

                    for bs in range(batch_sz):
                        this_feat = _keys[bs].contiguous().view(feat_dim, -1)
                        this_label = _labels[bs].contiguous().view(-1)
                        this_label_ids = torch.unique(this_label)
                        this_label_ids = [x for x in this_label_ids if x > 0]
                        for lb in this_label_ids:
                            idxs = (this_label == lb).nonzero()
                            lb = int(lb.item())
                            # segment enqueue and dequeue
                            feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                            ptr = int(segment_queue_ptr_1[lb])
                            segment_queue_1[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                            segment_queue_ptr_1[lb] = (segment_queue_ptr_1[lb] + 1) % memory_size

                            # pixel enqueue and dequeue
                            num_pixel = idxs.shape[0]
                            perm = torch.randperm(num_pixel)
                            K = min(num_pixel, pixel_update_freq)
                            feat = this_feat[:, perm[:K]]
                            feat = torch.transpose(feat, 0, 1)
                            ptr = int(pixel_queue_ptr_1[lb])

                            if ptr + K >= memory_size:
                                pixel_queue_1[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                                pixel_queue_ptr_1[lb] = 0
                            else:
                                pixel_queue_1[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                                pixel_queue_ptr_1[lb] = (pixel_queue_ptr_1[lb] + 1) % memory_size

                if iter_num % args.save_fig_interval == 0 and save_fig:
                    grid_image = make_grid(mean_preds * 50., 4, normalize=False)
                    writer.add_image('train/mean_preds_1', grid_image, iter_num)
                    grid_image = make_grid(certainty_pseudo * 50., 4, normalize=False)
                    writer.add_image('train/certainty_pseudo_1', grid_image, iter_num)
                    grid_image = make_grid(uncertainty, 4, normalize=False)
                    writer.add_image('train/uncertainty_1', grid_image, iter_num)
                    grid_image = make_grid(uncertainty_mask.float(), 4, normalize=False)
                    writer.add_image('train/uncertainty_mask_1', grid_image, iter_num)                    
            
            else:
                loss_contra_1 = 0.

            # contrast for net_2
            if iter_num > args.contrast_iter and args.contrast_weight > 0.:

                # queue = torch.cat((segment_queue, pixel_queue), dim=1) if memory else None
                queue_2 = segment_queue_2 if memory else None
                seg_mean = torch.mean(torch.stack([output_soft_cnn_2, output_soft_tf_2]), dim=0)
                uncertainty = -1.0 * torch.sum(seg_mean * torch.log(seg_mean + 1e-6), dim=1, keepdim=True)
                threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
                uncertainty_mask = (uncertainty > threshold)
                mean_preds = torch.argmax(F.softmax(seg_mean, dim=1).detach(), dim=1, keepdim=True).float()
                certainty_pseudo = mean_preds.clone()
                certainty_pseudo[uncertainty_mask] = -1
                preds = torch.argmax(output_soft_cnn_2, dim=1, keepdim=True).to(torch.float)
                certainty_pseudo[scribble_mask] = scribble[scribble_mask].to(torch.float)

                loss_contra_2 = contrast_criterion(outputs_2['embed'], certainty_pseudo, preds, queue=queue_2)

                writer.add_scalar('train/uncertainty_rate', torch.sum(uncertainty_mask == True) / \
                                                (torch.sum(uncertainty_mask == True) + torch.sum(
                                                    uncertainty_mask == False)), iter_num)

                if memory: # update the queue
                    _keys, _labels = outputs_2['embed'].detach(), certainty_pseudo.detach()
                    batch_sz, feat_dim = _keys.shape[0], _keys.shape[1]
                    _labels = torch.nn.functional.interpolate(_labels, (_keys.shape[2], _keys.shape[3]), mode='nearest')

                    for bs in range(batch_sz):
                        this_feat = _keys[bs].contiguous().view(feat_dim, -1)
                        this_label = _labels[bs].contiguous().view(-1)
                        this_label_ids = torch.unique(this_label)
                        this_label_ids = [x for x in this_label_ids if x > 0]
                        for lb in this_label_ids:
                            idxs = (this_label == lb).nonzero()
                            lb = int(lb.item())
                            # segment enqueue and dequeue
                            feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                            ptr = int(segment_queue_ptr_2[lb])
                            segment_queue_2[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                            segment_queue_ptr_2[lb] = (segment_queue_ptr_2[lb] + 1) % memory_size

                            # pixel enqueue and dequeue
                            num_pixel = idxs.shape[0]
                            perm = torch.randperm(num_pixel)
                            K = min(num_pixel, pixel_update_freq)
                            feat = this_feat[:, perm[:K]]
                            feat = torch.transpose(feat, 0, 1)
                            ptr = int(pixel_queue_ptr_2[lb])

                            if ptr + K >= memory_size:
                                pixel_queue_2[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                                pixel_queue_ptr_2[lb] = 0
                            else:
                                pixel_queue_2[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                                pixel_queue_ptr_2[lb] = (pixel_queue_ptr_2[lb] + 1) % memory_size

                if iter_num % args.save_fig_interval == 0 and save_fig:
                    grid_image = make_grid(mean_preds * 50., 4, normalize=False)
                    writer.add_image('train/mean_preds_2', grid_image, iter_num)
                    grid_image = make_grid(certainty_pseudo * 50., 4, normalize=False)
                    writer.add_image('train/certainty_pseudo_2', grid_image, iter_num)
                    grid_image = make_grid(uncertainty, 4, normalize=False)
                    writer.add_image('train/uncertainty_2', grid_image, iter_num)
                    grid_image = make_grid(uncertainty_mask.float(), 4, normalize=False)
                    writer.add_image('train/uncertainty_mask_2', grid_image, iter_num)                    
            
            else:
                loss_contra_2 = 0.

            #=================== contrast loss (above) ========================


            ### second forward

            # net_1
            pseudo_label_cnn_1 = torch.argmax(pseudo_label_cnn_1, dim=1)
            assert pseudo_label_cnn_1.requires_grad == False
            pseudo_label_tf_1 = torch.argmax(pseudo_label_tf_1, dim=1)
            assert pseudo_label_tf_1.requires_grad == False

            output_mixed_cnn_1 = net_1(img_mixed_1)['seg']
            assert output_mixed_cnn_1.requires_grad == True
            output_mixed_tf_1 = net_1(img_mixed_1)['seg_tf']
            assert output_mixed_tf_1.requires_grad == True

            # cross entropy with pseudo_label
            loss_unsup_mixed_cnn_1 = F.cross_entropy(output_mixed_cnn_1, pseudo_label_cnn_1)
            loss_unsup_mixed_tf_1 = F.cross_entropy(output_mixed_tf_1, pseudo_label_tf_1)
            # cross entropy with mixed scribble
            loss_sup_mixed_cnn_1 = partial_ce_loss(output_mixed_cnn_1, scribble_mixed_1[:, 0].long())
            loss_sup_mixed_tf_1 = partial_ce_loss(output_mixed_tf_1, scribble_mixed_1[:, 0].long())

            # net_2
            pseudo_label_cnn_2 = torch.argmax(pseudo_label_cnn_2, dim=1)
            assert pseudo_label_cnn_2.requires_grad == False
            pseudo_label_tf_2 = torch.argmax(pseudo_label_tf_2, dim=1)
            assert pseudo_label_tf_2.requires_grad == False

            output_mixed_cnn_2 = net_2(img_mixed_2)['seg']
            assert output_mixed_cnn_2.requires_grad == True
            output_mixed_tf_2 = net_2(img_mixed_2)['seg_tf']
            assert output_mixed_tf_2.requires_grad == True

            # cross entropy with pseudo_label
            loss_unsup_mixed_cnn_2 = F.cross_entropy(output_mixed_cnn_2, pseudo_label_cnn_2)
            loss_unsup_mixed_tf_2 = F.cross_entropy(output_mixed_tf_2, pseudo_label_tf_2)
            # cross entropy with mixed scribble
            loss_sup_mixed_cnn_2 = partial_ce_loss(output_mixed_cnn_2, scribble_mixed_2[:, 0].long())
            loss_sup_mixed_tf_2 = partial_ce_loss(output_mixed_tf_2, scribble_mixed_2[:, 0].long())

            ### loss_1 and loss_2
            het_weight = get_current_het_weight(epoch_num, max_iterations, args.het_weight)

            loss_sup_1 = loss_sup_mixed_cnn_1 + loss_sup_no_mix_cnn_1 + args.tf_decoder_weight * (loss_sup_mixed_tf_1 + loss_sup_no_mix_tf_1)
            loss_sup_2 = loss_sup_mixed_cnn_2 + loss_sup_no_mix_cnn_2 + args.tf_decoder_weight * (loss_sup_mixed_tf_2 + loss_sup_no_mix_tf_2)
            
            loss_mix_1 = loss_unsup_mixed_cnn_1 + args.tf_decoder_weight * loss_unsup_mixed_tf_1
            loss_mix_2 = loss_unsup_mixed_cnn_2 + args.tf_decoder_weight * loss_unsup_mixed_tf_2

            loss_het_1 = torch.mean((output_cnn_1 - output_tf_1) ** 2) + torch.mean((output_mixed_cnn_1 - output_mixed_tf_1) ** 2)
            loss_het_2 = torch.mean((output_cnn_2 - output_tf_2) ** 2) + torch.mean((output_mixed_cnn_2 - output_mixed_tf_2) ** 2)

            loss_1 = loss_sup_1 + loss_mix_1 + het_weight * loss_het_1 + args.contrast_weight * loss_contra_1
            loss_2 = loss_sup_2 + loss_mix_2 + het_weight * loss_het_2 + args.contrast_weight * loss_contra_2

            optimizer_1.zero_grad()
            loss_1.backward()
            optimizer_1.step()

            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()

            if args.update_lr:
                # change learning rate
                lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer_1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer_2.param_groups:
                    param_group['lr'] = lr_
            else:
                lr_ = args.base_lr

            iter_num = iter_num + 1

            # print and write
            logging.info('iter {}, net_1: lr {:.8f}, loss_1 {:.6f}, loss_sup_1 {:.6f}, loss_mix_1 {:.6f}, loss_het_1 {:.6f}, loss_contra_1 {:.6f}'.\
                         format(iter_num, lr_, loss_1.item(),loss_sup_1.item(), loss_mix_1.item(), loss_het_1.item(), loss_contra_1))

            logging.info('iter {}, net_2: lr {:.8f}, loss_2 {:.6f}, loss_sup_2 {:.6f}, loss_mix_2 {:.6f}, loss_het_2 {:.6f}, loss_contra_2 {:.6f}'.\
                         format(iter_num, lr_, loss_2.item(),loss_sup_2.item(), loss_mix_2.item(), loss_het_2.item(), loss_contra_2))
            
            writer.add_scalar('train/loss_1', loss_1.item(), iter_num)
            writer.add_scalar('train/loss_sup_1', loss_sup_1.item(), iter_num)
            writer.add_scalar('train/loss_mix_1', loss_mix_1.item(), iter_num)
            writer.add_scalar('train/loss_het_1', loss_het_1.item(), iter_num)
            writer.add_scalar('train/loss_contra_1', loss_contra_1, iter_num)
            
            writer.add_scalar('train/loss_2', loss_2.item(), iter_num)
            writer.add_scalar('train/loss_sup_2', loss_sup_2.item(), iter_num)
            writer.add_scalar('train/loss_mix_2', loss_mix_2.item(), iter_num)
            writer.add_scalar('train/loss_het_2', loss_het_2.item(), iter_num)
            writer.add_scalar('train/loss_contra_2', loss_contra_2, iter_num)

            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/het_weight', het_weight, iter_num)
            

            # save fig
            if iter_num % args.save_fig_interval == 0 and save_fig:
                # net_1
                image = make_grid(image_shuffle_1, image_shuffle_1.shape[0], normalize=True)
                writer.add_image('train/image_shuffle_1', image, iter_num)

                image = make_grid(cutmix_mask_shuffle_1.type_as(img_mixed_1), cutmix_mask_shuffle_1.shape[0],
                                  normalize=False)
                writer.add_image('train/cutmix_mask_shuffle_1', image, iter_num)

                image = make_grid(img_mixed_1, img_mixed_1.shape[0], normalize=True)
                writer.add_image('train/img_mixed_1', image, iter_num)

                image = make_grid(pseudo_label_cnn_1.unsqueeze(1), pseudo_label_cnn_1.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_cnn_1', image * 50, iter_num)
                image = make_grid(pseudo_label_tf_1.unsqueeze(1), pseudo_label_tf_1.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_tf_1', image * 50, iter_num)
                image = make_grid(pseudo_label_cnn_2.unsqueeze(1), pseudo_label_cnn_2.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_cnn_2', image * 50, iter_num)
                image = make_grid(pseudo_label_tf_2.unsqueeze(1), pseudo_label_tf_2.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_tf_2', image * 50, iter_num)

                image = make_grid(scribble_shuffle_1, scribble_shuffle_1.shape[0], normalize=False)
                writer.add_image('train/scribble_shuffle_1', image * 50, iter_num)

                image = make_grid(scribble_mixed_1, scribble_mixed_1.shape[0], normalize=False)
                writer.add_image('train/scribble_mixed_1', image * 50, iter_num)

                # net_2
                image = make_grid(image_shuffle_2, image_shuffle_2.shape[0], normalize=True)
                writer.add_image('train/image_shuffle_2', image, iter_num)

                image = make_grid(cutmix_mask_shuffle_2.type_as(img_mixed_2), cutmix_mask_shuffle_2.shape[0],
                                  normalize=False)
                writer.add_image('train/cutmix_mask_shuffle_2', image, iter_num)

                image = make_grid(img_mixed_2, img_mixed_2.shape[0], normalize=True)
                writer.add_image('train/img_mixed_2', image, iter_num)

                image = make_grid(pseudo_label_cnn_2.unsqueeze(1), pseudo_label_cnn_2.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_cnn_2', image * 50, iter_num)
                image = make_grid(pseudo_label_tf_2.unsqueeze(1), pseudo_label_tf_2.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_tf_2', image * 50, iter_num)
                image = make_grid(pseudo_label_cnn_2.unsqueeze(1), pseudo_label_cnn_2.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_cnn_2', image * 50, iter_num)
                image = make_grid(pseudo_label_tf_2.unsqueeze(1), pseudo_label_tf_2.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_tf_2', image * 50, iter_num)

                image = make_grid(scribble_shuffle_2, scribble_shuffle_2.shape[0], normalize=False)
                writer.add_image('train/scribble_shuffle_2', image * 50, iter_num)

                image = make_grid(scribble_mixed_2, scribble_mixed_2.shape[0], normalize=False)
                writer.add_image('train/scribble_mixed_2', image * 50, iter_num)

            # validation
            if iter_num % args.val_interval == 0:
                # net_1
                net_1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_dataloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], net_1, 'seg', classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info_net_1/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info_net_1/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info_net_1/val_mean_dice', performance, iter_num)
                writer.add_scalar('info_net_1/val_mean_hd95', mean_hd95, iter_num)

                logging.info(
                    'net_1 : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

                # net_2
                net_2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_dataloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], net_2, 'seg', classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info_net_2/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info_net_2/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info_net_2/val_mean_dice', performance, iter_num)
                writer.add_scalar('info_net_2/val_mean_hd95', mean_hd95, iter_num)

                logging.info(
                    'net_2 : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                
                # ensemble
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_dataloader):
                    metric_i = test_single_volume_ensemble(
                        sampled_batch["image"], sampled_batch["label"], net_1, net_2, 'seg', classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info_ensemble/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info_ensemble/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info_ensemble/val_mean_dice', performance, iter_num)
                writer.add_scalar('info_ensemble/val_mean_hd95', mean_hd95, iter_num)

                # update the model if got a better model
                if performance > best_dice:
                    best_dice = performance
                    update_model_for_dice = True
                if mean_hd95 < best_hd95:
                    best_hd95 = mean_hd95
                    update_model_for_hd95 = True
                
                logging.info(
                    'ensemble : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

                net_1.train()
                net_2.train()

                if update_model_for_dice:
                    save_mode_path = os.path.join(snapshot_path, 'best_net_1_dice.pth')
                    torch.save(net_1.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                    save_mode_path = os.path.join(snapshot_path, 'best_net_2_dice.pth')
                    torch.save(net_2.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                    update_model_for_dice = False
                
                if update_model_for_hd95:
                    save_mode_path = os.path.join(snapshot_path, 'best_net_1_hd95.pth')
                    torch.save(net_1.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                    save_mode_path = os.path.join(snapshot_path, 'best_net_2_hd95.pth')
                    torch.save(net_2.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                    update_model_for_hd95 = False

            if iter_num >= max_iterations:
                break
            
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    main()
