#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from std_msgs.msg import Bool, Int32
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import airsim
import numpy as np
from scipy.spatial.transform import Rotation as R # Quaternion to Euler
import pprint
from setting_params import FREQ_LOW_LEVEL, filter_coeff
import pickle
from trajectory.cubic_spline_planner import generate_path


# Way point path generation
waypoints_2d = [[-0.799999952316284,  0.699999988079071,  0.799999952316284, -53.79999923706055, -102.89999389648438, -108.29999542236328, -101.79999542236328,  -53.69999694824219,   0.799999952316284,  1.699999988079071],
                [-10.199999809265137, 24.19999885559082, 62.099998474121094,  66.19999694824219,  59.0,                 24.19999885559082,   -9.399999618530273, -16.5,              -11.199999809265137, 24.19999885559082]]
RXY = generate_path(waypoints_2d)
TARGET_SPEED = 5
P_speed = 1

### ROS Subscriber Callback ###
KEY_CMD_RECEIVED = None
def fnc_callback(msg):
    global KEY_CMD_RECEIVED
    KEY_CMD_RECEIVED = msg

CTL_CMD_RECEIVED = None
def fnc_callback1(msg):
    global CTL_CMD_RECEIVED
    CTL_CMD_RECEIVED = msg

TRACKING_ON_CMD_RECEIVED = Bool()
TRACKING_ON_CMD_RECEIVED.data = True
def fnc_callback4(msg):
    global TRACKING_ON_CMD_RECEIVED
    TRACKING_ON_CMD_RECEIVED = msg

ENVIRONMENT_CMD_RECEIVED = None
def fnc_callback7(msg):
    global ENVIRONMENT_CMD_RECEIVED
    ENVIRONMENT_CMD_RECEIVED = msg

def reset(client):
    print("=======================================")
    client.reset()
    client.confirmConnection()
    client.enableApiControl(True, "Car1")


def run_car_airsim_node():
    # connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True, "Car1") # This car is the follower.

    ### State Values ###
    COUNT_ERROR = 0
    car_controls1 = airsim.CarControls() # initilize control vaule object
    t_step = 0
    rospy.set_param('done_ack', False)
    rospy.set_param('episode_done', False)

    # rosnode node initialization
    rospy.init_node('car_airsim_node')
    reset(client)

    # subscriber init.
    sub_key_cmd         = rospy.Subscriber('/key_teleop/vel_cmd_body_frame', Twist, fnc_callback)
    sub_vel_cmd_control = rospy.Subscriber('/controller_node/vel_cmd', Vector3, fnc_callback1)
    sub_bool_cmd_tracking_on = rospy.Subscriber('/key_teleop/tracking_control_bool', Bool, fnc_callback4)
    sub_highlvl_environment_command = rospy.Subscriber('/key_teleop/highlvl_environment_command', Int32, fnc_callback7)   # subscriber init.

    # publishers init.
    pub_camera_frame = rospy.Publisher('/airsim_node/camera_frame', Image, queue_size=1)
    pub_state_obs_values = rospy.Publisher('/airsim_node/state_obs_values', Float32MultiArray, queue_size=1)
    
    # a bridge from cv2 (np.uint8 image) image to ROS image
    mybridge = CvBridge()

    # msg init. the msg is to send out state value array.
    msg_mat = Float32MultiArray()
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim[0].label = "height"
    msg_mat.layout.dim[1].label = "width"

    # Running rate
    rate=rospy.Rate(FREQ_LOW_LEVEL)

    ### The Car Charater ###
    # trajectory
    rx = np.array(RXY[0])
    ry = np.array(RXY[1])
    r_xy = np.array([rx,ry])

    car_controls2 = airsim.CarControls() # initilize control vaule object
    client.enableApiControl(True, "Car2") # This car is the leader

    ##############################
    ### Instructions in a loop ###
    ##############################
    while not rospy.is_shutdown():
        t_step+=1
        # Get state value
        state = client.getCarState("Car1")
        #s = pprint.pformat(state)
        #print("state: %s" % s)
        collision_state = client.simGetCollisionInfo()
        #print('collision_state', collision_state.has_collided)
        state_orientation = state.kinematics_estimated.orientation
        body_angle = R.from_quat(state_orientation.to_numpy_array()).as_euler('zxy', degrees=False)
        body_yaw_angle = body_angle[0] - np.pi
        linear_velocity = state.kinematics_estimated.linear_velocity.to_numpy_array()
        position = state.kinematics_estimated.position.to_numpy_array()
        target = client.simGetObjectPose("TargetCube")
        target_position = target.position.to_numpy_array()
        #print(target_position)        
        distance_to_target = np.linalg.norm(target_position-position)
        speed = np.linalg.norm(linear_velocity)
        acceleration  = np.linalg.norm(state.kinematics_estimated.linear_acceleration.to_numpy_array())
        null_speed_collision = np.array([distance_to_target, speed, collision_state.has_collided])
        np_state = np.stack([body_angle, linear_velocity, position, null_speed_collision])

        msg_mat.layout.dim[0].size = np_state.shape[0]
        msg_mat.layout.dim[1].size = np_state.shape[1]
        msg_mat.layout.dim[0].stride = np_state.shape[0]*np_state.shape[1]
        msg_mat.layout.dim[1].stride = np_state.shape[1]
        msg_mat.layout.data_offset = 0
        msg_mat.data = np_state.flatten().tolist()

        pub_state_obs_values.publish(msg_mat)

        # Get Camera Images
        try:                
            responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)], "Car1")
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
            car_controls1.throttle = max(KEY_CMD_RECEIVED.linear.x, 0)
            car_controls1.brake = min(KEY_CMD_RECEIVED.linear.x, 0)
            car_controls1.steering = KEY_CMD_RECEIVED.linear.y
        else:
            car_controls1.throttle = 0
            car_controls1.brake = 0
            car_controls1.steering = 0

        # Send tracking control command velocity to Airsim
        if CTL_CMD_RECEIVED is not None and TRACKING_ON_CMD_RECEIVED is not None and TRACKING_ON_CMD_RECEIVED.data:
            car_controls1.throttle += max(CTL_CMD_RECEIVED.z, 0)*0.5
            car_controls1.brake += min(CTL_CMD_RECEIVED.z, 0)
            car_controls1.steering += np.clip(CTL_CMD_RECEIVED.x, -0.3, 0.3)

        client.setCarControls(car_controls1, "Car1")


        ### Control the leading car ###

        # get state of the leading car
        state = client.getCarState("Car2")

        x_val = state.kinematics_estimated.position.x_val
        y_val = state.kinematics_estimated.position.y_val
        vx_val = state.kinematics_estimated.linear_velocity.x_val
        vy_val = state.kinematics_estimated.linear_velocity.y_val


        n_way_points = r_xy.shape[-1]
        car_xy = np.array([x_val, y_val]).reshape(2,-1).repeat(n_way_points, 1)
        distance = (np.sum((r_xy - car_xy)**2, axis=0))**0.5
        index = min(np.argmin(distance)+1, n_way_points-1)
        tgt_xy = r_xy.T[index]
        pos_error_vector = tgt_xy - np.array([x_val, y_val])
        v_xy = np.array([vx_val, vy_val])
    
        steer_angle = np.cross(v_xy, pos_error_vector)
        steering = np.clip(steer_angle, -0.5, 0.5)

        throttle = P_speed * (max(TARGET_SPEED - state.speed, 0))
        brake = P_speed * abs(min(TARGET_SPEED - state.speed, 0))

        car_controls2.steering = steering
        car_controls2.throttle = throttle
        car_controls2.brake = brake

        client.setCarControls(car_controls2, "Car2")

        # Send enviornment command to Airsim
        
        if ENVIRONMENT_CMD_RECEIVED is not None and ENVIRONMENT_CMD_RECEIVED.data == 1:
            print('Reset!')
            reset(client)


        

        # Condition 1: if !done_ack & episode_done then done_ack=True

        # Condition 2: if done_ack & !episode_done then done_ack=False

        # Condition 3: if done_ack & episode_done then done_ack=True

        # Condition 4: if !done_ack & !episode_done then done_ack=False

        if not(rospy.get_param('done_ack')) and rospy.get_param('episode_done'):
            reset(client)
            t_step = 0
            rospy.set_param('done_ack', True)
        elif rospy.get_param('done_ack') and not(rospy.get_param('episode_done')):
            rospy.set_param('done_ack', False)
        # elif rospy.get_param('done_ack') and rospy.get_param('episode_done'):
        #     print('keep ack true')
        # elif not(rospy.get_param('done_ack')) and not(rospy.get_param('episode_done')):
        #     print('keep ack false')
                 
            
        

        try:
            experiment_done = rospy.get_param('experiment_done')
        except:
            experiment_done = False
        if experiment_done and t_step>FREQ_LOW_LEVEL*3:
            reset(client)
            rospy.set_param('done_ack', True)
            rospy.signal_shutdown('Finished 100 Episodes!')



        
    
        # Sleep for Set Rate
        rate.sleep()
        

if __name__ == '__main__':
    try:
        run_car_airsim_node()
    except rospy.ROSInterruptException:
        pass
