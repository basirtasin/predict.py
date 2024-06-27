# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import argparse
import time
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import pandas as pd
torch.cuda.set_device(0)
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np

global bbox

#import pafy
import csv

'''This Code is for writing and appending the data in a csv file'''
def create_and_write_initial_data(filename, data):
    """Creates a CSV file and writes the initial data."""
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data)


def append_data_iteration(filename, data):
    """Appends data to the existing CSV file."""
    with open(filename, 'a',newline='' ) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data)

# Main execution example
filename = f'csv/my_data{np.random.randint(1000)}.csv'
initial_data = ['Time (sec) ', 'u (kt or km/h) for each vessel', 'k (veh/NM) for a given time', 'ql (veh/h)', 'qe (veh/h)']  # Header row

# Create the file and write the initial data
create_and_write_initial_data(filename, initial_data)

'''ABOVE DOCUMENTATION ENDS HERE '''
cap = None
current_frame = 0
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

object_counter = {}

object_counter1 = {}

line1 = [(1253, 581), (226, 388)]
line2 = [(1250, 600), (176, 397)]
boats_crossed_ids_line1 = set()
boats_crossed_ids_line2 = set()
def WarpImage_TPS(source, target):
    tps = cv2.createThinPlateSplineShapeTransformer()

    source = source.reshape(-1, len(source), 2)
    target = target.reshape(-1, len(target), 2)

    matches = list()
    for i in range(len(source[0])):
        matches.append(cv2.DMatch(i, i, 0))

    tps.estimateTransformation(target, source, matches)  # note it is target --> source

    #new_img = tps.warpImage(img)

    # get the warp kps in for source and target
    tps.estimateTransformation(source, target, matches)  # note it is source --> target
    f32_pts = np.zeros(source.shape, dtype=np.float32)
    f32_pts[:] = source[:]
    transform_cost, new_pts1 = tps.applyTransformation(f32_pts)  # e.g., 1 x 4 x 2
    f32_pts = np.zeros(target.shape, dtype=np.float32)
    f32_pts[:] = target[:]
    transform_cost, new_pts2 = tps.applyTransformation(f32_pts)  # e.g., 1 x 4 x 2

    return new_pts1, new_pts2, tps
def transform_single_point(tps, point):
    point = np.array([[point]], dtype=np.float32)
    _, transformed_point = tps.applyTransformation(point)
    return transformed_point[0, 0]

global tps

Zp = np.array([
    [68, 423], [177, 406], [219, 397], [363, 370], [507, 344], [539, 342], [662, 324], [761,324], [799, 324], [850, 317],[884, 291],
    [1109, 691], [1131, 679], [1160, 467], [1261, 368],[1252, 313], [1186, 472], [1263, 532], [1254, 517], [1205, 274], [1005, 277], [1190, 257]
])  # (x, y) of source in each row
Zs = np.array([
    [315,445], [354, 427], [370, 418], [425, 394], [504, 361], [519, 357], [606, 319], [657, 323],[667, 339], [718, 316],[865,192],  
    [485,674], [499, 674], [566, 634], [755, 574], [1124, 415], [567, 642], [539, 678], [550, 672], [1504, 143], [1074, 130],[1786, 72]
])
new_pts1, new_pts2, tps = WarpImage_TPS(Zp, Zs)
speed_line_queue = {}
def estimatespeed(Location1, Location2):


    Location1 = list(Location1)
    Location2 = list(Location2)

    Location1 = transform_single_point(tps, Location1)
    Location2 = transform_single_point(tps, Location2)

    #Euclidean Distance Formula
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    # defining thr pixels per meter
    ppm = 4.787 # WILL VARY WITH EVERY VIDEO
    d_meters = d_pixel/ppm

    time_constant = (30/300) *3.6 #30 FPS  and 300 is after how much frame the location and time is taken , 3.6 is used for converting it to km/h
    #distance = speed/time

    speed = d_meters * time_constant

    return int(speed)
def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
def display_counts(img, object_counter, object_counter1, width):
    for idx, (key, value) in enumerate(object_counter1.items()):
        cnt_str = str(key) + ":" + str(value)
        cv2.line(img, (width - 500, 25), (width, 25), [85, 45, 255], 40)
        cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (width - 150, 65 + (idx * 40)), (width, 65 + (idx * 40)), [85, 45, 255], 30)
        cv2.putText(img, cnt_str, (width - 150, 75 + (idx * 40)), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    for idx, (key, value) in enumerate(object_counter.items()):
        cnt_str1 = str(key) + ":" + str(value)
        cv2.line(img, (20, 25), (500, 25), [85, 45, 255], 40)
        cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (20, 65 + (idx * 40)), (127, 65 + (idx * 40)), [85, 45, 255], 30)
        cv2.putText(img, cnt_str1, (11, 75 + (idx * 40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

def handle_boat_crossing(direction, id, object_id, obj_name, object_speed, speed_line_queue, current_frame, cap):
    if direction == 'north':
        if obj_name not in object_counter:
            object_counter[obj_name] = 1
        else:
            object_counter[obj_name] += 1

        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamp = current_frame / fps
        new_data = [str(timestamp), str(int(sum(speed_line_queue[id]) / len(speed_line_queue[id]))), "", object_counter[obj_name]]
        append_data_iteration(filename, new_data)

    elif direction == 'south':
        if obj_name not in object_counter1:
            object_counter1[obj_name] = 1
        else:
            object_counter1[obj_name] += 1

        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamp = current_frame / fps
        new_data = [str(timestamp), str(int(sum(speed_line_queue[id]) / len(speed_line_queue[id]))), "", "", object_counter1[obj_name]]
        append_data_iteration(filename, new_data)



def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str

boats_crossed_ids = []
def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    global current_frame, interval_data, density_calculated, cap
    current_frame += 1

    cv2.line(img, line1[0], line1[1], (46, 162, 112), 3)
    cv2.line(img, line2[0], line2[1], (46, 162, 112), 3)

    height, width, _ = img.shape

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
        id = int(identities[i]) if identities is not None else 0

        if id not in data_deque:
            data_deque[id] = deque(maxlen=500)
            speed_line_queue[id] = []

        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)
        data_deque[id].appendleft(center)

        if len(data_deque[id]) >= 150:
            direction = get_direction(data_deque[id][0], data_deque[id][149])
            object_speed = estimatespeed(data_deque[id][149], data_deque[id][0])
            speed_line_queue[id].append(object_speed)

            if id not in boats_crossed_ids_line2 and id not in boats_crossed_ids_line1 and intersect(data_deque[id][0], data_deque[id][29], line1[0], line1[1]):
                cv2.line(img, line1[0], line1[1], (255, 255, 255), 3)
                boats_crossed_ids_line1.add(id)
                # Your logic for handling boats crossing line1
                if "South" in direction:
                    handle_boat_crossing('south', id, object_id, obj_name, object_speed, speed_line_queue, current_frame, cap)
                elif "North" in direction:
                    handle_boat_crossing('north', id, object_id, obj_name, object_speed, speed_line_queue, current_frame, cap)
            elif id not in boats_crossed_ids_line2 and id not in boats_crossed_ids_line1 and intersect(data_deque[id][0], data_deque[id][29], line2[0], line2[1]):
                cv2.line(img, line2[0], line2[1], (255, 255, 255), 3)
                boats_crossed_ids_line2.add(id)
                # Your logic for handling boats crossing line2
                if "South" in direction:
                    handle_boat_crossing('south', id, object_id, obj_name, object_speed, speed_line_queue, current_frame, cap)
                elif "North" in direction:
                    handle_boat_crossing('north', id, object_id, obj_name, object_speed, speed_line_queue, current_frame, cap)

        try:
            avg_speed = int(sum(speed_line_queue[id]) / len(speed_line_queue[id]))
            label = f"{id}:{obj_name} {avg_speed} km/h"
        except:
            pass

        UI_box(box, img, label=label, color=color, line_thickness=2)
        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            thickness = 3
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], (255, 255, 255), thickness)

    # Display counts
    display_counts(img, object_counter, object_counter1, width)

    return img


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            if cls.item() not in [8]:  # Skip if not a boat
                continue
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)


        if xywhs.numel() > 0:  # Check if xywhs is not empty
            outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]

            # Density Calculation
            num_vehicles = len(bbox_xyxy)
            length = 0.310  # 310m
            k = int(num_vehicles / length)

            # Append data to CSV
            fps = cap.get(cv2.CAP_PROP_FPS)
            #print(fps)
            timestamp = current_frame / fps
            new_data = [str(round(timestamp)), "", str(k), ""]  # Update with speed and flow if available

            # Write to Csv after every 'frame_interval' frames (using pandas)
            frame_interval = int(fps)

            if current_frame % frame_interval == 0:
                with open(filename, 'a', newline= "") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(new_data)

            draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    global cap
    cap = cv2.VideoCapture(str(cfg.source))  # Convert Path object to string


    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":

    predict()
