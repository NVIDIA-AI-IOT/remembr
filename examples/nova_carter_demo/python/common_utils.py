def format_pose_msg(msg: PoseWithCovarianceStamped):

    position = np.array([
        msg.pose.pose.position.x, 
        msg.pose.pose.position.y, 
        msg.pose.pose.position.z
    ])

    quat = np.array([
        msg.pose.pose.orientation.x, 
        msg.pose.pose.orientation.y, 
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    ])

    euler_rot_z = R.from_quat(quat).as_euler('xyz')[-1] # take z rotation

    stamp = odom_msg.header.stamp
    converted_time = float(str(stamp.sec) + '.' + str(stamp.nanosec))
    

    return position, angle, converted_time
