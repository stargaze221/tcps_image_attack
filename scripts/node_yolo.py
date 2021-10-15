#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import PIL

from sensor_msgs.msg import Image

from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from cv_bridge import CvBridge


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True, channels=3, classes=80, autoshape=True).to(DEVICE)

FREQ_NODE = 20

### ROS Subscriber Callback ###
IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg

if __name__=='__main__':

    rospy.init_node('perception_node')   # rosnode node initialization
    sub_image = rospy.Subscriber('/airsim_node/camera_frame', Image, fnc_img_callback)   # subscriber init.
    pub_yolo_prediction = rospy.Publisher('/yolo_node/yolo_predictions', Float32MultiArray, queue_size=1)   # publisher1 initialization.
    pub_yolo_boundingbox_video = rospy.Publisher('/yolo_node/yolo_pred_frame', Image, queue_size=1)   # publisher2 initialization.
    rate=rospy.Rate(FREQ_NODE)   # Running rate at 20 Hz

    # a bridge from cv2 image to ROS image
    mybridge = CvBridge()

    # msg init. the msg is to send out numpy array.
    msg_mat = Float32MultiArray()
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim[0].label = "height"
    msg_mat.layout.dim[1].label = "width"

    ##############################
    ### Instructions in a loop ###
    ##############################
    while not rospy.is_shutdown():

        if IMAGE_RECEIVED is not None:
            ### Use Yolo Model with Recived Image 
            np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
            results = YOLO_MODEL([np_im])

            ### Publish the prediction results in results.xyxy[0]) ###
            #                   x1           y1           x2           y2   confidence        class
            # tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
            #         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
            #         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])
            prediction_np = results.xyxy[0].cpu().numpy()
            msg_mat.layout.dim[0].size = prediction_np.shape[0]
            msg_mat.layout.dim[1].size = prediction_np.shape[1]
            msg_mat.layout.dim[0].stride = prediction_np.shape[0]*prediction_np.shape[1]
            msg_mat.layout.dim[1].stride = prediction_np.shape[1]
            msg_mat.layout.data_offset = 0
            msg_mat.data = prediction_np.flatten().tolist()
            pub_yolo_prediction.publish(msg_mat)

            ### Publish the bounding box image ###
            np_im = results.render()
            image_message = mybridge.cv2_to_imgmsg(np_im[0], encoding="passthrough")
            pub_yolo_boundingbox_video.publish(image_message)

        rate.sleep()