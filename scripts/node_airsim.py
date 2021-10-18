#!/usr/bin/env python3

import rospy

from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from std_msgs.msg import Bool, Int32

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R

import airsim
import numpy as np
import cv2
import pprint

### Global Objects and Variables ###

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.takeoffAsync().join()

FREQ_NODE = 20
filter_coeff = 0.1



### ROS Subscriber Callback ###
KEY_CMD_RECEIVED = None
def fnc_callback(msg):
    global KEY_CMD_RECEIVED
    KEY_CMD_RECEIVED = msg

CTL_CMD_RECEIVED = None
def fnc_callback1(msg):
    global CTL_CMD_RECEIVED
    CTL_CMD_RECEIVED = msg

TAKING_OFF_CMD_RECEIVED = None
def fnc_callback2(msg):
    global TAKING_OFF_CMD_RECEIVED
    TAKING_OFF_CMD_RECEIVED = msg

LANDING_OFF_CMD_RECEIVED = None
def fnc_callback3(msg):
    global LANDING_OFF_CMD_RECEIVED
    LANDING_OFF_CMD_RECEIVED = msg

TRACKING_ON_CMD_RECEIVED = None
def fnc_callback4(msg):
    global TRACKING_ON_CMD_RECEIVED
    TRACKING_ON_CMD_RECEIVED = msg

ENVIRONMENT_CMD_RECEIVED = None
def fnc_callback5(msg):
    global ENVIRONMENT_CMD_RECEIVED
    ENVIRONMENT_CMD_RECEIVED = msg



def run_airsim_node():

    ### Finite State ###
    ON_FLIGHT = False
    ON_TRACKING = False

    COUNT_ERROR = 0

    # rosnode node initialization
    rospy.init_node('airsim_node')

    # subscriber init.
    sub_key_cmd         = rospy.Subscriber('/key_teleop/vel_cmd_body_frame', Twist, fnc_callback)
    sub_vel_cmd_control = rospy.Subscriber('/controller_node/vel_cmd', Vector3, fnc_callback1)
    sub_bool_cmd_taking_off = rospy.Subscriber('/key_teleop/taking_off_bool', Bool, fnc_callback2)
    sub_bool_cmd_landing = rospy.Subscriber('/key_teleop/landing_bool', Bool, fnc_callback3)
    sub_bool_cmd_tracking_on = rospy.Subscriber('/key_teleop/tracking_control_bool', Bool, fnc_callback4)
    sub_int_cmd_highlvl_environment = rospy.Subscriber('/key_teleop/highlvl_environment_command', Int32, fnc_callback5)

    

    # publishers init.
    pub_camera_frame = rospy.Publisher('/airsim_node/camera_frame', Image, queue_size=1)
    pub_state_values = rospy.Publisher('/airsim_node/state_values', Float32MultiArray, queue_size=1)

    # msg init. the msg is to send out state value array.
    msg_mat = Float32MultiArray()
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim[0].label = "height"
    msg_mat.layout.dim[1].label = "width"

    # a bridge from cv2 image to ROS image
    mybridge = CvBridge()

    # Running rate
    rate=rospy.Rate(FREQ_NODE)

    vx = 0
    vy = 0
    vz = 0
    yaw_rate = 0

    ##############################
    ### Instructions in a loop ###
    ##############################
    while not rospy.is_shutdown():
        # Get state value
        state = client.getMultirotorState()
        # s = pprint.pformat(state)
        # print("state: %s" % s)

        state_orientation = state.kinematics_estimated.orientation
        body_angle = R.from_quat(state_orientation.to_numpy_array()).as_euler('zxy', degrees=False)
        body_yaw_angle = body_angle[0] - np.pi

        linear_velocity = state.kinematics_estimated.linear_velocity.to_numpy_array()
        position = state.kinematics_estimated.position.to_numpy_array()
        np_state = np.stack([body_angle, linear_velocity, position])

        msg_mat.layout.dim[0].size = np_state.shape[0]
        msg_mat.layout.dim[1].size = np_state.shape[1]
        msg_mat.layout.dim[0].stride = np_state.shape[0]*np_state.shape[1]
        msg_mat.layout.dim[1].stride = np_state.shape[1]
        msg_mat.layout.data_offset = 0
        msg_mat.data = np_state.flatten().tolist()
        pub_state_values.publish(msg_mat)

        # Get Camera Images
        try:                
            responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
            img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image array H X W X 3
            img_ros = mybridge.cv2_to_imgmsg(img_rgb)  
            # Publish the topics
            pub_camera_frame.publish(img_ros)
        except:
            COUNT_ERROR += 1
            print('The image was not retrieved!', COUNT_ERROR)

        # Send key command velocity to Airsim
        if KEY_CMD_RECEIVED is not None:
            cmd_vx = -KEY_CMD_RECEIVED.linear.x
            cmd_vy = -KEY_CMD_RECEIVED.linear.y
            cmd_vz = -KEY_CMD_RECEIVED.linear.z
            cmd_yaw = KEY_CMD_RECEIVED.angular.z*5
        else:
            cmd_vx = 0
            cmd_vy = 0
            cmd_vz = 0
            cmd_yaw = 0

        # Send tracking control command velocity to Airsim
        if CTL_CMD_RECEIVED is not None and TRACKING_ON_CMD_RECEIVED is not None and TRACKING_ON_CMD_RECEIVED.data:
            cmd_vx += -CTL_CMD_RECEIVED.z
            cmd_vy += -CTL_CMD_RECEIVED.x
            cmd_vz += -CTL_CMD_RECEIVED.y
            cmd_yaw += 0

        # First order filter for smoothing
        vx = (1-filter_coeff)*cmd_vx + filter_coeff*vx
        vy = (1-filter_coeff)*cmd_vy + filter_coeff*vy
        vz = (1-filter_coeff)*cmd_vz + filter_coeff*vz
        yaw_rate = (1-filter_coeff)*cmd_yaw + filter_coeff*yaw_rate
        vx_body = float(np.cos(body_yaw_angle)*vx - np.sin(body_yaw_angle)*vy)
        vy_body = float(np.sin(body_yaw_angle)*vx + np.cos(body_yaw_angle)*vy)
        client.moveByVelocityAsync(vx_body, vy_body, vz, 1/FREQ_NODE, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, yaw_rate))

        # Send enviornment command to Airsim
        #if (KEY_CMD_RECEIVED is not None) and (KEY_CMD_RECEIVED.angular.x>0):
        if TAKING_OFF_CMD_RECEIVED is not None and TAKING_OFF_CMD_RECEIVED.data and not ON_FLIGHT:
            print('Force Taking Off Command!')
            client.confirmConnection()
            client.enableApiControl(True)
            client.takeoffAsync().join()
            ON_FLIGHT = True
           
        #elif (KEY_CMD_RECEIVED is not None) and (KEY_CMD_RECEIVED.angular.x<0):
        elif TAKING_OFF_CMD_RECEIVED is not None and LANDING_OFF_CMD_RECEIVED.data and ON_FLIGHT:
            print('Emergency Landing!')
            client.landAsync().join()
            ON_FLIGHT = False

        elif ENVIRONMENT_CMD_RECEIVED is not None and ENVIRONMENT_CMD_RECEIVED.data == 1:
            print('Reset!')
            client.reset()
            pose = client.simGetVehiclePose("")
            pose.position.z_val += np.random.uniform(-5, -2)    #random [-5, 2]#
            pose.position.y_val += np.random.uniform(-5.0, 5.0) #random [-2.5, 2.5]# 

            client.simSetVehiclePose(pose, False)
            
            # tmp = client.simGetObjectPose("BP_Hatchback_2")
            # print('tmp', tmp)
            
            client.confirmConnection()
            client.enableApiControl(True)
            #client.takeoffAsync().join()

        # Sleep for Set Rate
        rate.sleep()
            
        
        

if __name__ == '__main__':
    try:
        run_airsim_node()
    except rospy.ROSInterruptException:
        pass
