
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
import time

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time
from renderer.screen_free_visualizer import Visualizer
from run_hand_mocap import *

from easy_tcp_python2_3 import socket_utils as su

class HandClient():
    def __init__(self):
        self.args = DemoOptions().parse()
        self.args.use_smplx = True
        self.args.save_pred_pkl = True
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.bbox_detector = HandBboxDetector(self.args.view_type,self.device)
        self.hand_mocap = HandMocap(self.args.checkpoint_hand,self.args.smpl_dir,device=self.device)
        self.visualizer = Visualizer(self.args.renderer_type)

    def get_pose(self,frame):
        result = run_hand_mocap(self.args,self.bbox_detector,self.hand_mocap,self.visualizer,frame)
        print(result)
        return result
        #result =['Right_hand,Left_hand].[Right_hand,0],[0,Left_hand],"None"
        #'Right_hand': (21,3) np.arrayuz

def main():
    sock = su.initialize_client('localhost',7777)
    
    client = HandClient()
    
    while True:

        frame = su.recvall_pickle(sock)
        pose = client.get_pose(frame)    
        su.sendall_pickle(sock,pose)
        

if __name__ == '__main__':
    main() 