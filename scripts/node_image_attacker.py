#!/usr/bin/env python3
import numpy as np
import rospy, cv2
import torch
import os

from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from agents.image_attack_agent import ImageAttacker
from setting_params import N_ACT_DIM, DEVICE, SETTING, FREQ_MID_LEVEL

IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg

ACTION_REWARD_RECEIVED = None
def fnc_action_reward_callback(msg):
    global ACTION_REWARD_RECEIVED
    ACTION_REWARD_RECEIVED = msg


if __name__ == '__main__':
    rospy.init_node('image_attack_node')
    print('Image_attack_node is initialized at', os.getcwd())
    
    sub_image = rospy.Subscriber('/airsim_node/camera_frame', Image, fnc_img_callback)
    sub_action_reward = rospy.Subscriber('/decision_maker_node/action_reward', Float32MultiArray, fnc_action_reward_callback)
    pub_attacked_image = rospy.Publisher('/attack_generator_node/attacked_image', Image, queue_size=1)    
    
    agent = ImageAttacker()
    mybridge = CvBridge()
    rate=rospy.Rate(FREQ_MID_LEVEL)
    count = 0

    while not rospy.is_shutdown():
        count += 1
        # Load the saved Model every 10 iteration
        if count%10 == 0:
            try:
                print(os.getcwd())
                agent.load_the_model(777)
            except:
                print('An error in loading the saved model. Two possible reasons: 1. no saved model, 2. both of nodes simultaneously try to access the file together')

        # Image generation
        if IMAGE_RECEIVED is not None: #(IMAGE_RECEIVED is not None) and (ACTION_REWARD_RECEIVED is not None):
            with torch.no_grad():
                # Get camera image
                np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
                np_im = np.array(np_im)
                # Get action
                # np_action_reward = np.array(ACTION_REWARD_RECEIVED.data)
                # act = np_action_reward[:N_ACT_DIM]
                act = np.array([-0.5, -0.5, -0.5, -0.5])
                # Get attacked image
                attacked_obs = agent.generate_attack(np_im, act)
            attacked_obs = (attacked_obs*255).astype('uint8')
            attacked_frame = mybridge.cv2_to_imgmsg(attacked_obs)

            # Publish messages
            pub_attacked_image.publish(attacked_frame)
        
        rate.sleep()