#!/usr/bin/env python3
import roslaunch
import rospy


def run(name):
    t = 0
    rospy.init_node('experiment', anonymous=True)
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, ["./launch/train.launch"])
    launch.start()
    rospy.loginfo("started")
    rospy.set_param('name', name)
    rospy.set_param('experiment_done', False)

    while True:
        t += 1
        rospy.sleep(1)
        if rospy.get_param('experiment_done') and t > 10:
            print('Experiment is done')
            break
    # 3 seconds later
    launch.shutdown()

if __name__ == "__main__":

    import names
    from datetime import datetime
    import tqdm

    for i in tqdm.tqdm(range(5)):
        now = datetime.now() # current date and time
        last_name = names.get_last_name()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        name = last_name + '_' + date_time

        run(name)