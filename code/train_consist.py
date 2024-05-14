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
from util.inference import test_single_volume, test_single_volume_ensemble
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
    parser.add_argument('--base_lr', type=float, default=0.03,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list, default=[224, 224],
                        help='patch size of network input')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU to use')
    parser.add_argument('--sup_weight', type=float, default=1, help='hyperparameter')
    parser.add_argument('--het_weight', type=float, default=1, help='hyperparameter')
    parser.add_argument('--mix_weight', type=float, default=1, help='hyperparameter')
    parser.add_argument('--contrast_weight', type=float, default=0.1, help='hyperparameter')
    parser.add_argument('--tf_decoder_weight', type=float, default=0.4, help='hyperparameter')
    parser.add_argument('--update_lr', action='store_false', help='whether update learning rate')
    parser.add_argument('--contrast_epoch', type=int, default=1000, help='when to start contrast')
    args = parser.parse_args()

    return args


def _random_rotate(image, scribble, label):
    angle = float(torch.empty(1).uniform_(-20., 20.).item())
    image = TF.rotate(image, angle)
    scribble = TF.rotate(scribble, angle)
    label = TF.rotate(label, angle)
    return image, scribble, label

def nce_loss(input1, input2):
    return 1 - F.cosine_similarity(input1, input2, dim=1).mean()

def get_current_het_weight(epoch, max_iter):
        # het ramp-up from https://arxiv.org/abs/1610.02242
        return 1.0 * ramps.sigmoid_rampup(epoch, max_iter // 100)

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
        segment_queue = torch.randn(pixel_classes, memory_size, dim)
        segment_queue = nn.functional.normalize(segment_queue, p=2, dim=2)
        segment_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)
        pixel_queue = torch.zeros(pixel_classes, memory_size, dim)
        pixel_queue = nn.functional.normalize(pixel_queue, p=2, dim=2)
        pixel_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)

    net = UNetTF(in_channels=1, classes=num_classes).cuda() 

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
    net.train()

    optimizer = optim.SGD(net.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=0.0001)

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

            ### first forward
            # cnn, supervised loss, no mix
            volume_batch = TF.Resize([224,224])(volume_batch)
            scribble = TF.Resize([224,224])(scribble)
            cutmix_mask = TF.Resize([224,224])(cutmix_mask)
            scribble_mask =  scribble != 4
            outputs = net(volume_batch)
            output_cnn = outputs['seg']
            loss_sup_no_mix_cnn = partial_ce_loss(output_cnn, scribble[:, 0].long())

            # tf, supervised loss, no mix
            output_tf = outputs['seg_tf']
            loss_sup_no_mix_tf = partial_ce_loss(output_tf, scribble[:, 0].long())


            # perform cutmix
            rand_index_cnn = torch.randperm(volume_batch.size()[0]).cuda()
            rand_index_tf = torch.randperm(volume_batch.size()[0]).cuda()

            # shuffle
            image_shuffle_cnn = volume_batch[rand_index_cnn]
            scribble_shuffle_cnn = scribble[rand_index_cnn]
            cutmix_mask_shuffle_cnn = cutmix_mask[rand_index_cnn]

            image_shuffle_tf = volume_batch[rand_index_tf]
            scribble_shuffle_tf = scribble[rand_index_tf]
            cutmix_mask_shuffle_tf = cutmix_mask[rand_index_tf]

            with torch.no_grad():
                output_unsup_cnn = output_cnn[rand_index_cnn]
                output_unsup_tf = output_tf[rand_index_tf]

            output_unsup_soft_cnn = F.softmax(output_unsup_cnn, dim=1)
            output_unsup_soft_tf = F.softmax(output_unsup_tf, dim=1)

            # throw dice to generate strong augmented image, pseudo label and mixed scribble annotation.
            if random.random() > 0.5:
                img_mixed = cutmix_mask_shuffle_cnn * image_shuffle_cnn + \
                    (1 - cutmix_mask_shuffle_cnn) * image_shuffle_tf
                pseudo_label = cutmix_mask_shuffle_cnn * output_unsup_soft_cnn + \
                    (1 - cutmix_mask_shuffle_cnn) * output_unsup_soft_tf
                scribble_mixed = cutmix_mask_shuffle_cnn * scribble_shuffle_cnn + \
                    (1 - cutmix_mask_shuffle_cnn) * scribble_shuffle_tf

            else:
                img_mixed = cutmix_mask_shuffle_tf * image_shuffle_tf + \
                    (1 - cutmix_mask_shuffle_tf) * image_shuffle_cnn
                pseudo_label = cutmix_mask_shuffle_tf * output_unsup_soft_tf + \
                    (1 - cutmix_mask_shuffle_tf) * output_unsup_soft_cnn
                scribble_mixed = cutmix_mask_shuffle_tf * scribble_shuffle_tf + \
                    (1 - cutmix_mask_shuffle_tf) * scribble_shuffle_cnn


            #=================== contrast loss (below) ========================
            
            if iter_num > args.contrast_epoch and args.contrast_weight > 0.:

                # queue = torch.cat((segment_queue, pixel_queue), dim=1) if memory else None
                queue = segment_queue if memory else None
                seg_mean = torch.mean(torch.stack([F.softmax(output_cnn, dim=1), F.softmax(output_tf, dim=1)]), dim=0)
                uncertainty = -1.0 * torch.sum(seg_mean * torch.log(seg_mean + 1e-6), dim=1, keepdim=True)
                threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
                uncertainty_mask = (uncertainty > threshold)
                mean_preds = torch.argmax(F.softmax(seg_mean, dim=1).detach(), dim=1, keepdim=True).float()
                certainty_pseudo = mean_preds.clone()
                certainty_pseudo[uncertainty_mask] = -1
                preds = torch.argmax(output_cnn, dim=1, keepdim=True).to(torch.float)
                certainty_pseudo[scribble_mask] = scribble[scribble_mask].to(torch.float)

                loss_contra = contrast_criterion(outputs['embed'], certainty_pseudo, preds, queue=queue)

                writer.add_scalar('train/uncertainty_rate', torch.sum(uncertainty_mask == True) / \
                                                (torch.sum(uncertainty_mask == True) + torch.sum(
                                                    uncertainty_mask == False)), iter_num)

                if memory: # update the queue
                    _keys, _labels = outputs['embed'].detach(), certainty_pseudo.detach()
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
                            ptr = int(segment_queue_ptr[lb])
                            segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                            segment_queue_ptr[lb] = (segment_queue_ptr[lb] + 1) % memory_size

                            # pixel enqueue and dequeue
                            num_pixel = idxs.shape[0]
                            perm = torch.randperm(num_pixel)
                            K = min(num_pixel, pixel_update_freq)
                            feat = this_feat[:, perm[:K]]
                            feat = torch.transpose(feat, 0, 1)
                            ptr = int(pixel_queue_ptr[lb])

                            if ptr + K >= memory_size:
                                pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                                pixel_queue_ptr[lb] = 0
                            else:
                                pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                                pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % memory_size

                if iter_num % args.save_fig_interval == 0 and save_fig:
                    grid_image = make_grid(mean_preds * 50., 4, normalize=False)
                    writer.add_image('train/mean_preds', grid_image, iter_num)
                    grid_image = make_grid(certainty_pseudo * 50., 4, normalize=False)
                    writer.add_image('train/certainty_pseudo', grid_image, iter_num)
                    grid_image = make_grid(uncertainty, 4, normalize=False)
                    writer.add_image('train/uncertainty', grid_image, iter_num)
                    grid_image = make_grid(uncertainty_mask.float(), 4, normalize=False)
                    writer.add_image('train/uncertainty_mask', grid_image, iter_num)                    
            
            else:
                loss_contra = 0.

            #=================== contrast loss (above) ========================


            ### second forward
            pseudo_label = torch.argmax(pseudo_label, dim=1)
            assert pseudo_label.requires_grad == False

            # cnn
            output_mixed_cnn = net(img_mixed)['seg']
            assert output_mixed_cnn.requires_grad == True
            
            # cross entropy with pseudo_label
            loss_unsup_mixed_cnn = F.cross_entropy(output_mixed_cnn, pseudo_label)
            # cross entropy with mixed scribble
            loss_sup_mixed_cnn = partial_ce_loss(output_mixed_cnn, scribble_mixed[:, 0].long())

            # tf
            output_mixed_tf = net(img_mixed)['seg_tf']
            assert output_mixed_tf.requires_grad == True

            # cross entropy with pseudo_label
            loss_unsup_mixed_tf = F.cross_entropy(output_mixed_tf, pseudo_label)
            # cross entropy with mixed scribble
            loss_sup_mixed_tf = partial_ce_loss(output_mixed_tf, scribble_mixed[:, 0].long())

            # sum up
            loss_sup = loss_sup_no_mix_cnn + loss_sup_mixed_cnn + \
                args.tf_decoder_weight * (loss_sup_no_mix_tf + loss_sup_mixed_tf)
            loss_het = torch.mean((output_cnn - output_tf) ** 2) + torch.mean((output_mixed_cnn - output_mixed_tf) ** 2)
            loss_mix = loss_unsup_mixed_cnn + args.tf_decoder_weight * loss_unsup_mixed_tf
            
            het_weight = get_current_het_weight(epoch_num, max_iterations)
            loss = args.sup_weight * loss_sup + het_weight * loss_het + \
                args.mix_weight * loss_mix + args.contrast_weight * loss_contra

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.update_lr:
                # change learning rate
                lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                lr_ = args.base_lr

            iter_num = iter_num + 1

            # print and write
            logging.info('iter {}, lr {:.8f}, loss {:.6f}, loss_sup {:.6f}, loss_het {:.6f}, loss_mix {:.6}, loss_contra {:.6f}'.\
                         format(iter_num, lr_, loss.item(),loss_sup.item(), loss_het.item(), loss_mix.item(), loss_contra))

            writer.add_scalar('train/loss_total', loss.item(), iter_num)
            writer.add_scalar('train/loss_sup', loss_sup.item(), iter_num)
            writer.add_scalar('train/loss_het', loss_het.item(), iter_num)
            writer.add_scalar('train/loss_mix', loss_mix.item(), iter_num)
            writer.add_scalar('train/loss_contra', loss_contra, iter_num)

            writer.add_scalar('train/lr', lr_, iter_num)
            

            # save fig
            if iter_num % args.save_fig_interval == 0 and save_fig:
                image = make_grid(image_shuffle_cnn, image_shuffle_cnn.shape[0], normalize=True)
                writer.add_image('train/image_shuffle_cnn', image, iter_num)

                image = make_grid(cutmix_mask_shuffle_cnn.type_as(img_mixed), cutmix_mask_shuffle_cnn.shape[0],
                                  normalize=False)
                writer.add_image('train/cutmix_mask_shuffle_cnn', image, iter_num)

                image = make_grid(img_mixed, img_mixed.shape[0], normalize=True)
                writer.add_image('train/img_mixed', image, iter_num)

                image = make_grid(pseudo_label.unsqueeze(1), pseudo_label.shape[0], normalize=False)
                writer.add_image('train/pseudo_label', image * 50, iter_num)

                image = make_grid(scribble_shuffle_cnn, scribble_shuffle_cnn.shape[0], normalize=False)
                writer.add_image('train/scribble_shuffle_cnn', image * 50, iter_num)

                image = make_grid(scribble_mixed, scribble_mixed.shape[0], normalize=False)
                writer.add_image('train/scribble_mixed', image * 50, iter_num)

                image = make_grid(image_shuffle_tf, image_shuffle_tf.shape[0], normalize=True)
                writer.add_image('train/image_shuffle_tf', image, iter_num)

                image = make_grid(cutmix_mask_shuffle_tf.type_as(img_mixed), cutmix_mask_shuffle_tf.shape[0],
                                  normalize=False)
                writer.add_image('train/cutmix_mask_shuffle_tf', image, iter_num)

                image = make_grid(scribble_shuffle_tf, scribble_shuffle_tf.shape[0], normalize=False)
                writer.add_image('train/scribble_shuffle_tf', image * 50, iter_num)

            # validation
            if iter_num % args.val_interval == 0:
                net.eval()
                # cnn decoder
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_dataloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], net, 'seg', classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info_cnn/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info_cnn/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info_cnn/val_mean_dice', performance, iter_num)
                writer.add_scalar('info_cnn/val_mean_hd95', mean_hd95, iter_num)

                # update the model if got a better model
                if performance > best_dice:
                    best_dice = performance
                    update_model_for_dice = True
                
                if mean_hd95 < best_hd95:
                    best_hd95 = mean_hd95
                    update_model_for_hd95 = True

                logging.info(
                    'cnn : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

                # tf decoder
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_dataloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], net, 'seg_tf', classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info_tf/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info_tf/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info_tf/val_mean_dice', performance, iter_num)
                writer.add_scalar('info_tf/val_mean_hd95', mean_hd95, iter_num)

                logging.info(
                    'tf : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                

                # ensemble
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_dataloader):
                    metric_i = test_single_volume_ensemble(
                        sampled_batch["image"], sampled_batch["label"], net, classes=num_classes)
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

                logging.info(
                    'ensemble : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

                net.train()

                if update_model_for_dice:
                    save_mode_path = os.path.join(snapshot_path, 'best_net_dice.pth')
                    torch.save(net.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                    update_model_for_dice = False
                
                if update_model_for_hd95:
                    save_mode_path = os.path.join(snapshot_path, 'best_net_hd95.pth')
                    torch.save(net.state_dict(), save_mode_path)
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
