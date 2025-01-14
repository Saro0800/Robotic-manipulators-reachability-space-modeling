import ikpy.chain
import ikpy.inverse_kinematics
import rospy
import struct
import time
import ikpy
import matplotlib.pyplot as plt
import scipy
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import math


def gen_point_cloud(num_samples):
    # Load the robot chain from a URDF file
    robot_chain = ikpy.chain.Chain.from_urdf_file("/home/rosario/Desktop/Base-Optimization-for-Mobile-Robots/reach_space_estimation/generate_pointcloud/model/ur5e.urdf",
                                    active_links_mask=[False, False, True, True, True, False, False, False, False, False])
    
    # set the robot in an extended state
    fk = robot_chain.forward_kinematics(np.array([0.0, 0.0, 0.0, -np.pi/2, 0.0, 0.0, np.pi, 0.0, 0.0, 0.0]), full_kinematics=True)

    rpp_frame_matr = fk[7]
    rpp_pos = rpp_frame_matr[:3,3]
    robot_len = np.linalg.norm(rpp_pos, ord=2)

    # create a cube
    x_span = np.linspace(-robot_len, robot_len, num_samples)
    y_span = np.linspace(-robot_len, robot_len, num_samples)
    z_span = np.linspace(-robot_len, robot_len, num_samples)

    # create an empty list
    point_cloud = []
    all_points = []

    # start the timer
    start = time.time()

    # iterate on all samples
    for x in x_span:
        for y in y_span:
            for z in z_span:
                joint_angs = robot_chain.inverse_kinematics(target_position=[x,y,z])
                fk = robot_chain.forward_kinematics(joints=joint_angs)
                
                if math.isclose(fk[0,3],x) and math.isclose(fk[1,3], y) and math.isclose(fk[2,3], z):
                    point_cloud.append(fk[:3,3])
                all_points.append([x,y,z])
          
    tot_time = time.time() - start

    point_cloud = np.array(point_cloud)

    return point_cloud, tot_time, all_points

def perform_analysis(samples, num_runs):
    results = np.zeros((samples.shape[0], num_runs))
    
    for i in range(0,samples.shape[0]):
        print("\n"+str(samples[i]), end=": ", flush=True)

        for j in range(0, num_runs):
            print(j, end=" ", flush=True)
            point_cloud, tot_time, all_points = gen_point_cloud(samples[i])
            results[i][j] = tot_time
        print("- time required: {:.6f}".format(results[i][num_runs-1]), end="")
    
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

def pack_rgb_into_float(r, g, b):
    # Packs the RGB values into a single float value.
    rgb = (r << 16) | (g << 8) | b
    return struct.unpack('f', struct.pack('I', rgb))[0]

def create_pointcloud_msg(point_cloud, link):
    
    points = []

    r, g, b = 0, 255, 0  # green color
    color = pack_rgb_into_float(r,g,b)
    for point in point_cloud:
        elem = [point[0], point[1], point[2], color]
        points.append(elem)
    
    points = np.array(points)

    # Create PointCloud2 message
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = link

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.FLOAT32, 1)
    ]

    return pcl2.create_cloud(header=header, fields=fields, points=points)

def create_all_points_msg(all_points, link):
    
    points = []

    r, g, b = 255, 0, 0  # green color
    color = pack_rgb_into_float(r,g,b)
    for point in all_points:
        elem = [point[0], point[1], point[2], color]
        points.append(elem)
    
    points = np.array(points)

    # Create PointCloud2 message
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = link

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.FLOAT32, 1)
    ]

    return pcl2.create_cloud(header=header, fields=fields, points=points)

def publish_msg(pub_point_cloud, pub_all_points, rate, pointcloud_msg, all_points_msg):
    print("Start publishing message")
    while not rospy.is_shutdown():
        pub_point_cloud.publish(pointcloud_msg)
        pub_all_points.publish(all_points_msg)
        rate.sleep()

def create_ros_node():
    rospy.init_node('reachability_pointcloud_publisher', anonymous=True)
    pub_point_cloud = rospy.Publisher('/reachability_pointcloud', PointCloud2, queue_size=10)
    pub_all_points = rospy.Publisher('/reachability_all_points', PointCloud2, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    return pub_point_cloud, pub_all_points, rate
    
def main():
    # pub_point_cloud, pub_all_points, rate = create_ros_node()

    link = "base_link_inertia"
    # get the pointcloud points
    start = 10
    end = 100
    step = 10
    num_runs = 1

    samples = np.arange(start, end+step, step)

    # point_cloud, tot_time, all_points = gen_point_cloud(20)

    results = perform_analysis(samples, num_runs)
    avg_results = np.average(results, axis=1)
    var_results = np.var(results, axis=1)
    scipy.io.savemat("results/ur5e_gen_times_IK.mat", {"samples": samples, 
                                               "avg_times": avg_results,
                                               "var_times": var_results})     
    

    draw_plot(samples, avg_results)

    # pointcloud_msg = create_pointcloud_msg(point_cloud, link)
    # all_points_msg = create_all_points_msg(all_points, link)

    # publish_msg(pub_point_cloud, pub_all_points, rate, pointcloud_msg, all_points_msg)


if __name__=="__main__":
    main()
