#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import cv2
import os
import numpy as np
import time
import pickle

test_path = "../coco/val2014"
save_path = "../coco/val2014/predictions"

test_classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

#topk
topk = 40


images_folder = os.path.join(test_path, "images")
annotations_folder = os.path.join(test_path, "annotations")

images = os.listdir(images_folder)

def detect(image_name):
	prediction_path = os.path.join(save_path, image_name)
	with open(prediction_path, "rb") as f:
		predictions = pickle.load(f)
	return predictions[1]

def sign(x):
    if x>=0: return 1
    else: return -1

def get_IoU(boxA, boxB):
    classA, confidenceA, positionA = boxA
    classB, confidenceB, positionB = boxB
    if classA != classB: return 0

    xA, yA, wA, hA = positionA
    xB, yB, wB, hB = positionB
    
    tx = sign(xB - xA)
    ty = sign(yB - yA)

    Ix = (wA/2 + wB/2 + tx*(xA - xB))
    Iy = (hA/2 + hB/2 + ty*(yA - yB))

    if Ix < 0 or Iy < 0:    I = 0
    else:                   I = Ix * Iy

    U = wA*hA + wB*hB - I
    return I/U

TPs = np.zeros(topk)
ABs = np.zeros(topk)
PBs = np.zeros(topk)

n = 0
for image_name in images:
    n += 1
    print("{} Image {}".format(n, image_name))
    image_path = os.path.join(images_folder, image_name)
    annotation_path = os.path.join(annotations_folder, image_name[:-4] + ".txt")
    if not os.path.exists(annotation_path): continue

    img = cv2.imread(image_path)

    with open(annotation_path, "r") as f:
        annotation_raw = f.read()
    annotation = []
    for box_raw in annotation_raw.split("\n"):
        pack = box_raw.split(" ")
        try: 
            class_name, x, y, w, h = pack[0], pack[1], pack[2], pack[3], pack[4]
        except IndexError:
            continue
        class_name = test_classes[int(class_name)]
        x = img.shape[1]*float(x)
        y = img.shape[0]*float(y)
        w = img.shape[1]*float(w)
        h = img.shape[0]*float(h)
        box = (class_name, 1, (x, y, w, h))
        annotation.append(box)

    prediction = detect(image_name)
    if len(prediction) == 0: continue


    #print("predicted {} over {} images".format(len(prediction), len(annotation)))


    TP = np.zeros(topk)


    for k in range(topk):
      	if k < len(prediction):
            IoU_list = []
            for box in annotation:
                pred_box = prediction[k]
                IoU = get_IoU(box, pred_box)
                IoU_list.append(IoU)
            if max(IoU_list) > 0.5:
                TP[k] = 1 + TP[k-1]
            else:
                TP[k] = 0 + TP[k-1]
        else:
            TP[k] = TP[len(prediction)-1]

    PB = np.zeros(topk)
    if topk <= len(prediction):
        for k in range(topk):
            PB[k] = k+1
    else:
        for k in range(len(prediction)):
            PB[k] = k+1
        for k in range(len(prediction), topk):
            PB[k] = len(prediction)

    AB = np.zeros(topk)
    AB += len(annotation)

                #k top k predictions
    TPs += TP   #true positive
    PBs += PB   #predicted boxes
    ABs += AB   #annotated boxes
    #print("TP {}\nPB {}\nAB {}".format(TP, PB, AB))
    #import pdb
    #pdb.set_trace()
    #print("TPs {}\nPBs {}\n ABs{}".format(TPs, PBs, ABs))

    precision   = TPs/PBs
    recall      = TPs/ABs

    #print("precision    {}".format(precision))
    #print( "recall      {}".format(recall))

import matplotlib.pyplot as plt

recall 		= np.hstack((0,recall,1))
precision 	= np.hstack((1, precision, 0))

mAP = 0
for i in range(len(recall) -1):
    dw = recall[i+1] - recall[i]
    h = (precision[i+1] + precision[i])/2
    mAP += h*dw

print(mAP)

plt.plot(recall, precision, "ro")
plt.show()
