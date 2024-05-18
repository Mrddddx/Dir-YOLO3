# coding='utf-8'
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil
import cv2
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import torch
import torch.nn as nn


MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.utils import non_max_suppression, bbox_iou

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

# coding='utf-8'
import os
import sys
import numpy as np
import logging
import cv2
import random
import torch
import torch.nn as nn
import importlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.utils import non_max_suppression, bbox_iou

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

def load_model(config):
    is_training = False
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)
    net = nn.DataParallel(net)
    net = net.cuda()
    
    if config["pretrain_snapshot"]:
        logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    return net

def prepare_yolo_losses(config):
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i], config["yolo"]["classes"], (config["img_w"], config["img_h"])))
    return yolo_losses

def detect_image(image, net, yolo_losses, config):
    image_origin = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config["img_w"], config["img_h"]), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = net(image)
        output_list = [yolo_losses[i](outputs[i]) for i in range(3)]
        output = torch.cat(output_list, 1)
        detections = non_max_suppression(output, config["yolo"]["classes"], conf_thres=config["confidence_threshold"], nms_thres=0.45)
    
    return detections[0], image_origin

def draw_detections(image, detections, classes, config):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            ori_h, ori_w = image.shape[:2]
            pre_h, pre_w = config["img_h"], config["img_w"]
            box_h = ((y2 - y1).cpu().numpy() / pre_h) * ori_h
            box_w = ((x2 - x1).cpu().numpy() / pre_w) * ori_w
            y1 = (y1.cpu().numpy() / pre_h) * ori_h
            x1 = (x1.cpu().numpy() / pre_w) * ori_w
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top', bbox={'color': color, 'pad': 0})
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    return fig, ax  # Return fig and ax to show it later in the main loop

def main():
    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s %(filename)s] %(message)s")

    if len(sys.argv) != 2:
        logging.error("Usage: python camery.py params.py")
        sys.exit()
    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    net = load_model(config)
    yolo_losses = prepare_yolo_losses(config)

    classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open camera")
        sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read frame")
            break

        detections, image_origin = detect_image(frame, net, yolo_losses, config)
        fig, ax = draw_detections(image_origin, detections, classes, config)  # 传递 config 参数

        # Convert matplotlib figure to array and display it
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cv2.imshow('YOLOv3 Detection', img_array)
        plt.close(fig)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()