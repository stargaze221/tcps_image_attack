#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 PAL Robotics SL.
# Released under the BSD License.
#
# Authors:
#   * Siegfried-A. Gevatter

import curses
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool, Int32


class TextWindow():

    _screen = None
    _window = None
    _num_lines = None

    def __init__(self, stdscr, lines=10):
        self._screen = stdscr
        self._screen.nodelay(True)
        curses.curs_set(0)

        self._num_lines = lines

    def read_key(self):
        keycode = self._screen.getch()
        return keycode if keycode != -1 else None

    def clear(self):
        self._screen.clear()

    def write_line(self, lineno, message):
        if lineno < 0 or lineno >= self._num_lines:
            raise ValueError('lineno out of bounds')
        height, width = self._screen.getmaxyx()
        y = int((height / self._num_lines) * lineno)
        x = 10
        for text in message.split('\n'):
            text = text.ljust(width)
            self._screen.addstr(y, x, text)
            y += 1

    def refresh(self):
        self._screen.refresh()

    def beep(self):
        curses.flash()



class SimpleKeyTeleop():

    movement_bindings = {
        curses.KEY_UP:    (+1,  0,  0,  0,  0,  0,  0),  # +x forward
        curses.KEY_DOWN:  (-1,  0,  0,  0,  0,  0,  0),  # -x backward
        curses.KEY_LEFT:  ( 0, -1,  0,  0,  0,  0,  0),  # +y 
        curses.KEY_RIGHT: ( 0, +1,  0,  0,  0,  0,  0),  # -y
        ord('w'):         ( 0,  0, +1,  0,  0,  0,  0),  # +z
        ord('s'):         ( 0,  0, -1,  0,  0,  0,  0),  # -z
        ord('a'):         ( 0,  0,  0,  0,  0, -1,  0),  # +yaw
        ord('d'):         ( 0,  0,  0,  0,  0, +1,  0),  # -yaw
        ord('e'):         ( 0,  0,  0,  1,  0,  0,  0),  # toggle take-off / landing
        ord('c'):         ( 0,  0,  0,  2,  0,  0,  0),  # toggle tracking control on / off
        ord('k'):         ( 0,  0,  0,  3,  0,  0,  0),  # toggle image attack on / off
        ord('r'):         ( 0,  0,  0,  0,  1,  0,  0),  # reset environment
        ord('t'):         ( 0,  0,  0,  4,  0,  0,  0)   # training mode on/off
    }

    def __init__(self, interface):
        self._interface = interface
        self._pub_vel_cmd = rospy.Publisher('/key_teleop/vel_cmd_body_frame', Twist)
        self._hz = rospy.get_param('~hz', 10)

        self._forward_rate = rospy.get_param('~forward_rate', 1.0)
        self._backward_rate = rospy.get_param('~backward_rate', 1.0)
        self._side_rate = rospy.get_param('~side_move_rate', 1.0)
        self._elevation_rate = rospy.get_param('~elevation_rate', 1.0)

        self._rotation_rate = rospy.get_param('~rotation_rate', 1.0)
        self._last_pressed = {}
        self._angular = (0, 0, 0)
        self._linear = (0, 0, 0)
        self.m = -1

        ### Other commands ###
        self._tracking_control_bool = Bool()
        self._tracking_control_bool.data=True
        self._taking_off_bool = Bool()
        self._taking_off_bool.data = False
        self._landing_bool = Bool()
        self._landing_bool.data = False
        self._image_attack_bool = Bool()
        self._image_attack_bool.data=True
        self._training_mode_bool = Bool()
        self._training_mode_bool.data = True

        ### High level command ###
        self._environment_command = Int32()
        self._environment_command.data = int(0)

        self._pub_tracking_control_bool = rospy.Publisher('/key_teleop/tracking_control_bool', Bool)
        self._pub_taking_off_bool = rospy.Publisher('/key_teleop/taking_off_bool', Bool)
        self._pub_landing_bool = rospy.Publisher('/key_teleop/landing_bool', Bool)
        self._pub_image_attack_bool = rospy.Publisher('/key_teleop/image_attack_bool', Bool)
        self._pub_training_mode_bool = rospy.Publisher('/key_teleop/training_mode_bool', Bool)
        self._pub_highlvl_environment_command_int = rospy.Publisher('/key_teleop/highlvl_environment_command', Int32)
    

    def run(self):
        rate = rospy.Rate(self._hz)
        self._running = True
        while self._running:
            while True:
                keycode = self._interface.read_key()
                if keycode is None:
                    break
                self._key_pressed(keycode)
            self._set_velocity()
            self._publish()
            rate.sleep()

    def _get_velcmd(self, linear, angular):
        velcmd = Twist()
        velcmd.linear.x = linear[0]
        velcmd.linear.y = linear[1]
        velcmd.linear.z = linear[2]
        velcmd.angular.x = angular[0]
        velcmd.angular.y = angular[1]
        velcmd.angular.z = angular[2]
        return velcmd

    def _set_velocity(self):
        now = rospy.get_time()
        keys = []
        for a in self._last_pressed:
            if now - self._last_pressed[a] < 0.2:
                keys.append(a)
        
        vx, vy, vz, wx, wy, wz = (0, 0, 0, 0, 0, 0)
        attack_on_off = 0

        for k in keys:
            vx, vy, vz, wx, wy, wz, attack_on_off = self.movement_bindings[k]

        print(vx, vy, vz, wx, wy, wz)

        if wx == 1:   # toggle take-off / landing
            self._taking_off_bool.data = not(self._taking_off_bool.data)
            self._landing_bool.data = not(self._taking_off_bool.data)
        elif wx == 2:   # toggle tracking control on / off
            self._tracking_control_bool.data = not(self._tracking_control_bool.data)
        elif wx == 3:   # toggle image attack on / off
            self._image_attack_bool.data = not(self._image_attack_bool.data)
        elif wx == 4:   # training mode on / off
            self._training_mode_bool.data = not(self._training_mode_bool.data)

        ### High-level environment command ###
        self._environment_command.data = int(wy)

        if vx > 0:
            vx = vx * self._forward_rate
        else:
            vx = vx * self._backward_rate

        vy = vy * self._side_rate
        vz = vz * self._elevation_rate
        wz = wz * self._rotation_rate * 5

        self._linear = (vx, vy, vz)
        self._angular = (wx, wy, wz)
        self._attack_on_off = attack_on_off
        

    def _key_pressed(self, keycode):
        print('preseed key:', keycode)
        if keycode == ord('o'):
            self._running = False
            rospy.signal_shutdown('Bye')
        elif keycode in self.movement_bindings:
            self._last_pressed[keycode] = rospy.get_time()

    def _publish(self):
        self._interface.clear()
        self._interface.write_line(1, 'Foward/Backward: %f' % (self._linear[0]))
        self._interface.write_line(2, 'Left/Right     : %f' % (self._linear[1]))
        self._interface.write_line(3, 'Up/Down        : %f' % (self._linear[2]))
        self._interface.write_line(4, 'Angular        : %f' % (self._angular[2]))
        self._interface.write_line(5, 'Use c to toggle tracking controller on/off: %s' %(self._tracking_control_bool.data))
        self._interface.write_line(6, 'Use e to toggle takeoff: %s and landing: %s' %(self._taking_off_bool, self._landing_bool.data))
        self._interface.write_line(7, 'Use k to toggle image attack: %s' %(self._image_attack_bool.data))
        self._interface.write_line(8, 'Use t to toggle training mode: %s' %(self._training_mode_bool.data))
        self._interface.write_line(9, 'Use o to close the window.')
        self._interface.refresh()

        velcmd = self._get_velcmd(self._linear, self._angular)
        self._pub_vel_cmd.publish(velcmd)

        self._pub_tracking_control_bool.publish(self._tracking_control_bool)
        self._pub_taking_off_bool.publish(self._taking_off_bool)
        self._pub_landing_bool.publish(self._landing_bool)
        self._pub_image_attack_bool.publish(self._image_attack_bool)
        self._pub_training_mode_bool.publish(self._training_mode_bool)

        self._pub_highlvl_environment_command_int.publish(self._environment_command)


def main(stdscr):
    rospy.init_node('key_teleop')
    app = SimpleKeyTeleop(TextWindow(stdscr))
    app.run()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException:
        pass