import argparse
import re
import shutil
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from network.unet import UNet
from network.unet_tf import UNetTF



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../dataset/ACDC', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='contra_tuned', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='swin_unet', help='model_name')
    parser.add_argument('--fold', type=str,
                        default='fold1', help='fold')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='output channel of network')
    parser.add_argument('--sup_type', type=str, default="scribble",
                        help='label')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    # parser.add_argument('--device', type=str, default='cuda"0', help='device')
    parser.add_argument('--save_model_interval', type=int, default=10000, help='save model interval')
    # parser.add_argument('--best_model_iter', type=int, default=10000, help='save model interval')
    

    args = parser.parse_args()

    return args


def get_fold_ids(fold):
    all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
    fold1_testing_set = [
        "patient{:0>3}".format(i) for i in range(1, 21)]
    fold1_training_set = [
        i for i in all_cases_set if i not in fold1_testing_set]

    fold2_testing_set = [
        "patient{:0>3}".format(i) for i in range(21, 41)]
    fold2_training_set = [
        i for i in all_cases_set if i not in fold2_testing_set]

    fold3_testing_set = [
        "patient{:0>3}".format(i) for i in range(41, 61)]
    fold3_training_set = [
        i for i in all_cases_set if i not in fold3_testing_set]

    fold4_testing_set = [
        "patient{:0>3}".format(i) for i in range(61, 81)]
    fold4_training_set = [
        i for i in all_cases_set if i not in fold4_testing_set]

    fold5_testing_set = [
        "patient{:0>3}".format(i) for i in range(81, 101)]
    fold5_training_set = [
        i for i in all_cases_set if i not in fold5_testing_set]
    if fold == "fold1":
        return [fold1_training_set, fold1_testing_set]
    elif fold == "fold2":
        return [fold2_training_set, fold2_testing_set]
    elif fold == "fold3":
        return [fold3_training_set, fold3_testing_set]
    elif fold == "fold4":
        return [fold4_training_set, fold4_testing_set]
    elif fold == "fold5":
        return [fold5_training_set, fold5_testing_set]
    else:
        return "ERROR KEY"


def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
        return dice, hd95
    else:
        return 0, 50


def test_single_volume(case, net, decoer_name, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +"/ACDC_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)[decoer_name]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred
    case = case.replace(".h5", "")
    org_img_path = "../data/ACDC/ACDC_training/{}.nii.gz".format(case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.CopyInformation(org_img_itk)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def test_single_volume_ensemble_old(case, net1, net2, net3, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/ACDC_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net1.eval()
        net2.eval()
        net3.eval()
        with torch.no_grad():
            out_main1 = net1(input)
            out_main2 = net2(input)
            out_main3 = net3(input)
            out = torch.argmax(torch.softmax(
                (out_main1 + out_main2 + out_main3) / 3, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred
    case = case.replace(".h5", "")
    org_img_path = "../data/ACDC/ACDC_training/{}.nii.gz".format(case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.CopyInformation(org_img_itk)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def test_single_volume_ensemble(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/ACDC_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()

        net.eval()
        with torch.no_grad():
            out_main1 = torch.softmax(net(input)['seg'], dim=1)
            out_main2 = torch.softmax(net(input)['seg_tf'], dim=1)
            out = torch.argmax((out_main1 + out_main2) / 2, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred
    case = case.replace(".h5", "")
    org_img_path = "../data/ACDC/ACDC_training/{}.nii.gz".format(case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.CopyInformation(org_img_itk)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    train_ids, test_ids = get_fold_ids(FLAGS.fold)
    all_volumes = os.listdir(
        FLAGS.root_path + "/ACDC_training_volumes")
    image_list = []
    for ids in test_ids:
        new_data_list = list(filter(lambda x: re.match(
            '{}.*'.format(ids), x) != None, all_volumes))
        image_list.extend(new_data_list)
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type)

    test_save_path_1 = snapshot_path+"/{}_predictions_cnn/".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type, FLAGS.model)
    if os.path.exists(test_save_path_1):
        shutil.rmtree(test_save_path_1)
    os.makedirs(test_save_path_1)

    test_save_path_2 = snapshot_path+"/{}_predictions_tf/".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type, FLAGS.model)
    if os.path.exists(test_save_path_2):
        shutil.rmtree(test_save_path_2)
    os.makedirs(test_save_path_2)

    test_save_path_3 = snapshot_path+"/{}_predictions_ensemble/".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type, FLAGS.model)
    if os.path.exists(test_save_path_3):
        shutil.rmtree(test_save_path_3)
    os.makedirs(test_save_path_3)

    net = UNetTF(in_channels=1, classes=FLAGS.num_classes).cuda()

    save_mode_path = os.path.join(
        snapshot_path, f'best_net_dice.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_class_total = 0.0
    second_class_total = 0.0
    third_class_total = 0.0
    for case in tqdm(image_list):
        print(case)
        first_metric, second_metric, third_metric = test_single_volume(
            case, net,'seg', test_save_path_1, FLAGS)
        first_class_total += np.asarray(first_metric)
        second_class_total += np.asarray(second_metric)
        third_class_total += np.asarray(third_metric)
    avg_metric = [first_class_total / len(image_list), second_class_total /
                  len(image_list), third_class_total / len(image_list)]

    logging.info('inference from cnn.')
    logging.info(avg_metric)
    logging.info((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)

    ret_avg_metric = avg_metric

    first_class_total = 0.0
    second_class_total = 0.0
    third_class_total = 0.0
    for case in tqdm(image_list):
        print(case)
        first_metric, second_metric, third_metric = test_single_volume(
            case, net,'seg_tf', test_save_path_2, FLAGS)
        first_class_total += np.asarray(first_metric)
        second_class_total += np.asarray(second_metric)
        third_class_total += np.asarray(third_metric)
    avg_metric = [first_class_total / len(image_list), second_class_total /
                  len(image_list), third_class_total / len(image_list)]
    
    logging.info('inference from tf.')
    logging.info(avg_metric)
    logging.info((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)

    first_class_total = 0.0
    second_class_total = 0.0
    third_class_total = 0.0
    for case in tqdm(image_list):
        print(case)
        first_metric, second_metric, third_metric = test_single_volume_ensemble(
            case, net, test_save_path_3, FLAGS)
        first_class_total += np.asarray(first_metric)
        second_class_total += np.asarray(second_metric)
        third_class_total += np.asarray(third_metric)
    avg_metric = [first_class_total / len(image_list), second_class_total /
                  len(image_list), third_class_total / len(image_list)]
    logging.info('inference from ensemble.')
    logging.info(avg_metric)
    logging.info((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)

    return ret_avg_metric

def main():
    FLAGS = parser_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logging.basicConfig(filename="../model/" + FLAGS.exp + "_result.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    metric_all = [0.0, 0.0, 0.0]
    for i in [1, 2, 3, 4, 5]:
        FLAGS.fold = "fold{}".format(i)
        logging.info("Inference fold{}".format(i))
        metric_fold = Inference(FLAGS)
        metric_all = [i+j for i,j in zip(metric_all, metric_fold)]

    metric_all = [m/5.0 for m in metric_all]
    metric_mean = np.mean(metric_all, axis=0)
    logging.info("Final result from all the 5 folds")
    logging.info(f"metric for all classes: {metric_all}")
    logging.info(f"metric for mean: {metric_mean}")

if __name__ == '__main__':
    main()
