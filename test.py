from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import WIDERFace_ROOT, WIDERFace_CLASSES as labelmap
from PIL import Image
from data import WIDERFaceDetection, WIDERFaceAnnotationTransform, WIDERFace_CLASSES, WIDERFace_ROOT, BaseTransform, \
    TestBaseTransform
from data import *
import torch.utils.data as data
from light_face_ssd import build_ssd
# from resnet50_ssd import build_sfd
import pdb
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/light_DSFD.pth',
                    type=str, help='Trained state_dict file path to open')

parser.add_argument('--save_folder', default='eval_tools/light_DSFD/', type=str,
                    help='mosaiced img folder ')
parser.add_argument('--visual_threshold', default=0.9, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--video_folder', default='', type=str,
                    help='origin video folder')
parser.add_argument('--widerface_root', default=WIDERFace_ROOT, help='Location of VOC root directory')
parser.add_argument('--video_output', default='/home/rvlab/Desktop/', type=str,
                    help='processed video folder ')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[0:750, :]
    return dets


def write_to_txt(f, det, event, im_name):
    f.write('{:s}\n'.format(event + '/' + im_name))
    f.write('{:d}\n'.format(det.shape[0]))
    for i in range(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        score = det[i][4]
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))


def infer(net, img, transform, thresh, cuda, shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    with torch.no_grad():
        x = Variable(x.unsqueeze(0))
        if cuda:
            x = x.cuda()
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1] / shrink, img.shape[0] / shrink,
                              img.shape[1] / shrink, img.shape[0] / shrink])
        det = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                score = detections[0, i, j, 0]
                # label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                det.append([pt[0], pt[1], pt[2], pt[3], score])
                j += 1
        if (len(det)) == 0:
            det = [[0.1, 0.1, 0.2, 0.2, 0.01]]
        det = np.array(det)

        keep_index = np.where(det[:, 4] >= 0)[0]
        det = det[keep_index, :]
        return det


def save_mosaiced_img(im, dets, save_folder,image_name, frame_id,thresh=0.5):
    """Draw detected bounding boxes."""
    class_name = 'face'
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    print(len(inds))
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        do_mosaic(im, int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))

        '''
        ax.text(bbox[0], bbox[1] - 5,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=10, color='white')
        '''

    # cv2.imshow('test', im)
    # cv2.waitKey()
    cv2.imwrite(save_folder+image_name+'_'+'{:09d}.jpg'.format(frame_id),im)


def do_mosaic(frame, x, y, w, h, neighbor=9):
    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        return
    for i in range(0, h - neighbor, neighbor):
        for j in range(0, w - neighbor, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            color = frame[i + y][j + x].tolist()
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)
            cv2.rectangle(frame, left_up, right_down, color, -1)
def make_video_from_images(save_folder,img_paths, outvid_path, fps=25, size=None,
               is_color=True, format="H264"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    fourcc = cv2.VideoWriter_fourcc(*format)
    vid = None
    for ct, img_path in enumerate(img_paths):
        img = cv2.imread(save_folder+img_path)
        if img is None:
            print(img_path)
            continue
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = cv2.VideoWriter(outvid_path, fourcc, float(fps), size, is_color)

        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = cv2.resize(img, size)
        vid.write(img)
    if vid is not None:
        vid.release()
    return vid
def delete_imgs(img_folder):
    img_list=os.listdir(img_folder)
    for img in img_list:
        os.remove(img_folder+img)
    return print('imgs of last video have been removed')
def main():
    # load net
    shrink = 1
    cfg = widerface_640
    num_classes = len(WIDERFace_CLASSES) + 1  # +1 background
    net = build_ssd('test', cfg['min_dim'], num_classes)  # initialize SSD
    # net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.trained_model))
    cuda = args.cuda
    if cuda:
        net.cuda()
    else:
        net.cpu()
    net.eval()
    print('Finished loading model!')

    # evaluation

    transform = TestBaseTransform((104, 117, 123))
    thresh = cfg['conf_thresh']

    save_folder=args.save_folder

    video_folder_list=os.listdir(args.video_folder)
    for video in video_folder_list:
        frame_id=0
        video_name=os.path.splitext(os.path.split(video)[1])[0]
        cap=cv2.VideoCapture(args.video_folder+video)
        while True:
            _, frame = cap.read()
            if not _:
                break
            frame_id+=1
            det = infer(net, frame, transform, thresh, cuda, shrink)
            print('prossing:',frame_id)
            if det[0][0]==0.1:
                cv2.imwrite(save_folder + video_name + '_' + '{:09d}.jpg'.format(frame_id), frame)
            save_mosaiced_img(frame, det, save_folder,video_name, frame_id,0.9)
        save_folder_list=os.listdir(save_folder)
        save_folder_list.sort()
        make_video_from_images(save_folder,save_folder_list,
                           os.path.join(args.video_output, video_name+'.mp4'),
                           fps=60)
        delete_imgs(save_folder)

if __name__ == '__main__':
    main()
    # light_test_widerface()
