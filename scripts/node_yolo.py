#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import PIL
import os

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from cv_bridge import CvBridge

from yolo_wrapper import YoloWrapper
from setting_params import SETTING, FREQ_MID_LEVEL, DEVICE


DEVICE = torch.device("cpu")

YOLO_MODEL = YoloWrapper(SETTING['yolov5_param_path'], DEVICE)
YOLO_MODEL.model.eval()
YOLO_MODEL.model.to(DEVICE)
FREQ_NODE = FREQ_MID_LEVEL

### ROS Subscriber Callback ###
IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg

IMAGE_ATTACK_ON_CMD_RECEIVED = Bool()
IMAGE_ATTACK_ON_CMD_RECEIVED.data=True
def fnc_callback5(msg):
    global IMAGE_ATTACK_ON_CMD_RECEIVED
    IMAGE_ATTACK_ON_CMD_RECEIVED = msg

ATTACKED_IMAGE = None
def fnc_callback6(msg):
    global ATTACKED_IMAGE
    ATTACKED_IMAGE = msg

TARGET_RECEIVED = None
def fnc_target_callback(msg):
    global TARGET_RECEIVED
    TARGET_RECEIVED = msg

KF_BOX_RECEIVED = None
def fnc_target_callback7(msg):
    global KF_BOX_RECEIVED
    KF_BOX_RECEIVED = msg



if __name__=='__main__':

    # rosnode node initialization
    rospy.init_node('perception_node')   # rosnode node initialization
    print("Perception_node is initialized at", os.getcwd())

    # subscriber init.
    sub_image = rospy.Subscriber('/airsim_node/camera_frame', Image, fnc_img_callback)   # subscriber init.
    sub_bool_image_attack = rospy.Subscriber('/key_teleop/image_attack_bool', Bool, fnc_callback5)
    sub_attacked_image = rospy.Subscriber('/attack_generator_node/attacked_image', Image, fnc_callback6)   # subscriber init.
    sub_target = rospy.Subscriber('/decision_maker_node/target', Twist, fnc_target_callback)
    sub_kf_box = rospy.Subscriber('/controller_node/kf_box', Twist, fnc_target_callback7)

    # publishers init.
    pub_yolo_prediction = rospy.Publisher('/yolo_node/yolo_predictions', Float32MultiArray, queue_size=10)   # publisher1 initialization.
    pub_yolo_boundingbox_video = rospy.Publisher('/yolo_node/yolo_pred_frame', Image, queue_size=10)   # publisher2 initialization.
    rate=rospy.Rate(FREQ_NODE)   # Running rate at 20 Hz

    # a bridge from cv2 image to ROS image
    mybridge = CvBridge()

    # msg init. the msg is to send out numpy array.
    msg_mat = Float32MultiArray()
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim[0].label = "height"
    msg_mat.layout.dim[1].label = "width"

    t_step = 0

    ##############################
    ### Instructions in a loop ###
    ##############################
    while not rospy.is_shutdown():

        t_step += 1

        if IMAGE_RECEIVED is not None:
            if ATTACKED_IMAGE is not None and IMAGE_ATTACK_ON_CMD_RECEIVED is not None and IMAGE_ATTACK_ON_CMD_RECEIVED.data:
                np_im = np.frombuffer(ATTACKED_IMAGE.data, dtype=np.uint8).reshape(ATTACKED_IMAGE.height, ATTACKED_IMAGE.width, -1)
            else:
                np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)

            np_im = np.array(np_im)

            #print('KF_BOX_RECEIVED', KF_BOX_RECEIVED)
            with torch.no_grad():
                x_image = torch.FloatTensor(np_im).to(DEVICE).permute(2, 0, 1).unsqueeze(0)/255
                if TARGET_RECEIVED is not None and KF_BOX_RECEIVED is None:
                    action = (TARGET_RECEIVED.linear.x, TARGET_RECEIVED.linear.y, TARGET_RECEIVED.linear.z, TARGET_RECEIVED.angular.x)
                    cv2_images_uint8, prediction_np = YOLO_MODEL.draw_image_w_prediction_and_target(x_image.detach(), action)
                elif TARGET_RECEIVED is not None and KF_BOX_RECEIVED is not None:
                    action = (TARGET_RECEIVED.linear.x, TARGET_RECEIVED.linear.y, TARGET_RECEIVED.linear.z, TARGET_RECEIVED.angular.x)
                    kf_box = (KF_BOX_RECEIVED.linear.x, KF_BOX_RECEIVED.linear.y, KF_BOX_RECEIVED.linear.z, KF_BOX_RECEIVED.angular.x)
                    cv2_images_uint8, prediction_np = YOLO_MODEL.draw_image_w_prediction_and_target_and_kf_box(x_image.detach(), action, kf_box)
                elif TARGET_RECEIVED is None and KF_BOX_RECEIVED is not None:
                    kf_box = (KF_BOX_RECEIVED.linear.x, KF_BOX_RECEIVED.linear.y, KF_BOX_RECEIVED.linear.z, KF_BOX_RECEIVED.angular.x)
                    cv2_images_uint8, prediction_np = YOLO_MODEL.draw_image_w_prediction_and_kf_box(x_image.detach(), kf_box)
                else:
                    cv2_images_uint8, prediction_np = YOLO_MODEL.draw_image_w_predictions(x_image.detach())
            
            ### Publish the prediction results in results.xyxy[0]) ###
            #                   x1           y1           x2           y2   confidence        class
            # tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
            #         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
            #         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])
            if len(prediction_np)>0:    
                msg_mat.layout.dim[0].size = prediction_np.shape[0]
                msg_mat.layout.dim[1].size = prediction_np.shape[1]
                msg_mat.layout.dim[0].stride = prediction_np.shape[0]*prediction_np.shape[1]
                msg_mat.layout.dim[1].stride = prediction_np.shape[1]
                msg_mat.layout.data_offset = 0
                msg_mat.data = prediction_np.flatten().tolist()
                pub_yolo_prediction.publish(msg_mat)

            ### Publish the bounding box image ###
            image_message = mybridge.cv2_to_imgmsg(cv2_images_uint8, encoding="passthrough")
            pub_yolo_boundingbox_video.publish(image_message)
        
        try:
            experiment_done_done = rospy.get_param('experiment_done')
        except:
            experiment_done_done = False
        if experiment_done_done  and t_step > FREQ_MID_LEVEL*3:
            rospy.signal_shutdown('Finished 100 Episodes!')

        rate.sleep()