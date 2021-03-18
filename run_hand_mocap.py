
import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import json
import torch
from torchvision.transforms import Normalize

from demo.demo_options import DemoOptions
import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time

def run_hand_mocap(args, bbox_detector, hand_mocap, visualizer, frame):
    #Set up input data (images or webcam)
    img_original_bgr = frame 
    detect_output = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
    body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output
    

    if len(hand_bbox_list) < 1:
        print(f"No hand deteced")
        print(f"revised")
        return "None" 
    
    # Hand Pose Regression
    pred_output_list = hand_mocap.regress(img_original_bgr, hand_bbox_list, add_margin=True)
    
    hand = pred_output_list[0]

    if hand['right_hand'] == None and hand['left_hand'] == None:
        return "None"
    elif hand['right_hand'] == None and hand['left_hand'] != None:
        Lefthand = hand['left_hand']
        print('Lefthand:',Lefthand)
        Lhandjoint = Lefthand['pred_joints_img']
        return [0,Lhandjoint]
    elif hand['right_hand'] != None and hand['left_hand'] == None:
        Righthand = hand['right_hand']
        print(Righthand)
        Rhandjoint = Righthand['pred_joints_img']
        return [Rhandjoint,0]
    else:
        Righthand = hand['right_hand']
        Rhandjoint = Righthand['pred_joints_img']
        Lefthand = hand['left_hand']
        Lhandjoint = Lefthand['pred_joints_img']
        return [Rhandjoint,Lhandjoint]
    
    # extract mesh for rendering (vertices in image space and faces) from pred_output_list
    pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualize
    res_img = visualizer.visualize(img_original_bgr, pred_mesh_list = pred_mesh_list,hand_bbox_list = hand_bbox_list)

        # show result in the screen
    
    if not args.no_display:
        res_img = res_img.astype(np.uint8)
        ImShow(res_img)

        # save the image (we can make an option here)
    
    