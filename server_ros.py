from easy_tcp_python2_3 import socket_utils as su
import cv2, cv_bridge
from sensor_msgs.msg import Image
import rospy
import numpy as np 


class PoseServer:
    def __init__(self):
        rospy.init_node("pose_estimator")
        rospy.loginfo("Starting Pose Estimator node")
        self.cv_bridge = cv_bridge.CvBridge()

        self.sock,_ = su.initialize_server('localhost',7777)
        rgb_sub = rospy.Subscriber('azure1/rgb/image_raw',Image,self.callback)
        self.result_pub = rospy.Publisher('/estimation_result',Image,queue_size=1)

    def callback(self,rgb):
        rgb = self.cv_bridge.imgmsg_to_cv2(rgb,desired_encoding='bgr8')
        rospy.loginfo_once('send image to client')
        su.sendall_pickle(self.sock,rgb)
        rospy.loginfo_once('receive inference results from client')
        result = su.recvall_pickle(self.sock)
        # result : BODY25 keypoints 
        print(result)
        img_msg = self.cv_bridge.cv2_to_imgmsg(rgb,encoding='bgr8')
        self.result_pub.publish(img_msg)
        rospy.loginfo_once('Published the result as topic. Topic name : /estimation_result')
        
if __name__ == '__main__':

    server = PoseServer()
    rospy.spin()