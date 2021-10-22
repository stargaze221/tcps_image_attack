#!/usr/bin/env python3
# generates the attack to send to the tello drone

import numpy as np
import rospy, cv2
import torch

#from Tello_Image.Agent import ImageAttacker
from agent import ImageAttacker
import os


from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import Float32

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from setting_params import N_ACT_DIM, DEVICE, SETTING


IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg

ACTION_REWARD_RECEIVED = None
def fnc_action_reward_callback(msg):
    global ACTION_REWARD_RECEIVED
    ACTION_REWARD_RECEIVED = msg


from yolo_wrapper import YoloWrapper

YOLO_MODEL = YoloWrapper('yolov5m.pt')

if __name__ == '__main__':

    rospy.init_node('image_attack_node')
    
    sub_image = rospy.Subscriber('/airsim_node/camera_frame', Image, fnc_img_callback)
    sub_action_reward = rospy.Subscriber('/decision_maker_node/action_reward', Float32MultiArray, fnc_action_reward_callback)
    pub_attacked_image = rospy.Publisher('/attack_generator_node/attacked_image', Image, queue_size=1)    
    
    agent = ImageAttacker()
    mybridge = CvBridge()

    rate=rospy.Rate(20)
    loss_val = Float32()

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


            # Get camera image
            np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
            np_im = np.array(np_im)

            

            # Get action
            # np_action_reward = np.array(ACTION_REWARD_RECEIVED.data)
            # act = np_action_reward[:N_ACT_DIM]

            act = np.array([-0.5, -0.5, -0.5, -0.5])
            #act = np.random.rand(4)

            # Get attacked image
            attacked_obs = agent.generate_attack(np_im, act)



            attacked_obs = (attacked_obs*255).astype('uint8')
            attacked_frame = mybridge.cv2_to_imgmsg(attacked_obs)


            # # Calcualte the online attack loss
            # if ATTACK_LEVER.data > 0.5:
            #     X = torch.FloatTensor(attacked_obs).to(DEVICE).permute(2,0,1).unsqueeze(0).detach()
            # else:
            #     X = torch.FloatTensor(frame).to(DEVICE).permute(2,0,1).unsqueeze(0).detach()
            # Y = torch.FloatTensor(act).to(DEVICE).unsqueeze(0).detach()
            # """
            # X: minibatch image    [(1 x 3 x 448 x 448), ...]
            # Y: target coordinates [(x, y, w, h), ...]
            # """

            # total_loss, _, dict_loss_values = agent.calculate_loss(X,Y)
            # total_loss = total_loss.item()
            
            # error_xy = dict_loss_values['error_xy']
            # error_wh = dict_loss_values['error_wh']
            # error_obj_confidence = dict_loss_values['error_obj_confidence']
            # error_no_obj_confidence = dict_loss_values['error_no_obj_confidence']
            # error_class = dict_loss_values['error_class']
            # loss_reconst = dict_loss_values['loss_reconst']
            # loss_l2 = dict_loss_values['loss_l2']

            # attack_loss = error_xy + error_no_obj_confidence
            # loss_val.data = total_loss

            # Publish messages
            pub_attacked_image.publish(attacked_frame)
            # pub_attack_loss.publish(loss_val)
        
        rate.sleep()