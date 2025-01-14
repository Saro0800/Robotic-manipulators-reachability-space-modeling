import rospy
import struct
import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from ikpy.chain import Chain


def gen_point_cloud(num_samples):
    # Load the robot chain from a URDF file
    robot_chain = Chain.from_urdf_file("/home/rosario/Desktop/Base-Optimization-for-Mobile-Robots/reach_space_estimation/generate_pointcloud/model/ur5e.urdf",
                                    active_links_mask=[False, False, True, True, True, False, False, False, False, False])
    
    # Example joint values
    joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    joint_upper_limits = [0, 0, 2*np.pi, 2*np.pi, np.pi, 0, 0, 0, 0, 0]
    joint_lower_limits = [0, 0, -2*np.pi, -2*np.pi, -np.pi, 0, 0, 0, 0, 0]

    point_cloud = []

    start = time.time()

    # Compute FK
    for joint1 in np.linspace(joint_lower_limits[2], joint_upper_limits[2], num_samples):
        for joint2 in np.linspace(joint_lower_limits[3], joint_upper_limits[3], num_samples):
            for joint3 in np.linspace(joint_lower_limits[4], joint_upper_limits[4], num_samples):
                joint_values = joint_angles
                joint_values[2] = joint1
                joint_values[3] = joint2
                joint_values[4] = joint3

                fk = robot_chain.forward_kinematics(joint_angles, full_kinematics=True)
                rpp_frame_matr = fk[7]
                rpp_pos = rpp_frame_matr[:3,3]
                point_cloud.append(rpp_pos)

    tot_time = time.time() - start

    point_cloud = np.array(point_cloud)

    return point_cloud, tot_time

def create_pointcloud_msg(points, link):
    print(link)
    points_list = points.tolist()
    points_list.append([np.mean([np.min(points[:,0]), np.max(points[:,0])]),
                        np.mean([np.min(points[:,1]), np.max(points[:,1])]),
                        np.mean([np.min(points[:,2]), np.max(points[:,2])])])
    points = np.array(points_list)
    print([np.mean([np.min(points[:,0]), np.max(points[:,0])]),
                        np.mean([np.min(points[:,1]), np.max(points[:,1])]),
                        np.mean([np.min(points[:,2]), np.max(points[:,2])])])

    # Create PointCloud2 message
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = link

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]

    pointcloud_msg = PointCloud2()
    pointcloud_msg.header = header
    pointcloud_msg.height = 1
    pointcloud_msg.width = points.shape[0]
    pointcloud_msg.fields = fields
    pointcloud_msg.is_bigendian = False
    pointcloud_msg.point_step = 12  # 3 * 4 bytes (float32)
    pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width
    pointcloud_msg.is_dense = True

    # Pack the point data
    data = []
    for point in points:
        data.append(struct.pack('fff', *point))
    pointcloud_msg.data = b''.join(data)

    return pointcloud_msg

def publish_msg(pub_points, rate, pointcloud_msg):
    print("Start publishing message")
    while not rospy.is_shutdown():
        pub_points.publish(pointcloud_msg)
        rate.sleep()

def create_ros_node():
    rospy.init_node('reachability_pointcloud_publisher', anonymous=True)
    pub_points = rospy.Publisher('/reachability_pointcloud', PointCloud2, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    return pub_points, rate

def perform_analysis(samples, num_runs):
    results = np.zeros((samples.shape[0], num_runs))
    
    for i in range(0,samples.shape[0]):
        print("\n"+str(samples[i]), end=": ", flush=True)

        for j in range(0, num_runs):
            print(j, end=" ", flush=True)
            points, tot_time = gen_point_cloud(samples[i])
            results[i][j] = tot_time
    
    return results

def draw_plot(samples, avg_results):
    step = samples[1]-samples[0]
    # create the plot
    plt.scatter(x=samples, y=avg_results, marker="o")
    plt.plot(samples, avg_results)
    plt.grid(visible=True, alpha=0.5)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(r"$Number\; of\; samples$")
    plt.xlim(samples[0]-step, samples[-1]+step)
    plt.xticks(samples)
    plt.ylabel(r"$Avg\; generation\; time\; [s]$")
    plt.show()

def main():
    # get the pointcloud points
    start = 10
    end = 100
    step = 10
    num_runs = 10

    samples = np.arange(start, end+step, step)

    results = perform_analysis(samples, num_runs)
    avg_results = np.average(results, axis=1)
    var_results = np.var(results, axis=1)

    scipy.io.savemat("results/ur5e_gen_times_FK.mat", {"samples": samples, 
                                               "avg_times": avg_results,
                                               "var_times": var_results})   

    draw_plot(samples, avg_results)

if __name__== "__main__":
    main()

    