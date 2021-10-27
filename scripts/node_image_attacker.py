#!/usr/bin/env python3
import numpy as np
import rospy, cv2
import torch
import os

from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

from agents.image_attack_agent import ImageAttacker
from setting_params import N_ACT_DIM, DEVICE, SETTING, FREQ_MID_LEVEL

IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg

TARGET_RECEIVED = None
def fnc_target_callback(msg):
    global TARGET_RECEIVED
    TARGET_RECEIVED = msg


if __name__ == '__main__':
    rospy.init_node('image_attack_node')
    print('Image_attack_node is initialized at', os.getcwd())
    
    sub_image = rospy.Subscriber('/airsim_node/camera_frame', Image, fnc_img_callback)
    sub_target = rospy.Subscriber('/decision_maker_node/target', Twist, fnc_target_callback)
    pub_attacked_image = rospy.Publisher('/attack_generator_node/attacked_image', Image, queue_size=1)    
    
    agent = ImageAttacker()
    mybridge = CvBridge()
    rate=rospy.Rate(FREQ_MID_LEVEL)
    count = 0


    error_count = 0

    while not rospy.is_shutdown():
        count += 1
        # Load the saved Model every 10 iteration
        if count%FREQ_MID_LEVEL == 0:
            try:
                #print(os.getcwd())
                agent.load_the_model()
                error_count = 0
            except:
                error_count +=1
                if error_count > 10:
                    print('In image_attack_node, model loading failed more than 10 times!')

        # Image generation
        if IMAGE_RECEIVED is not None and TARGET_RECEIVED is not None:
            with torch.no_grad():
                # Get camera image
                np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
                np_im = np.array(np_im)
                # Get action
                act = np.array([TARGET_RECEIVED.linear.x, TARGET_RECEIVED.linear.y, TARGET_RECEIVED.linear.z, TARGET_RECEIVED.angular.x])
                # Get attacked image
                attacked_obs = agent.generate_attack(np_im, act)
            attacked_obs = (attacked_obs*255).astype('uint8')
            attacked_frame = mybridge.cv2_to_imgmsg(attacked_obs)

            # Publish messages
            pub_attacked_image.publish(attacked_frame)
        
        rate.sleep()