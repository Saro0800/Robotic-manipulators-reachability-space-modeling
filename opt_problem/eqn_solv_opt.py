import numpy as np
import rospy
import math
import time

from opt_problem.problem_formulation import EllipsoidEquationOptProblem
from generate_pointcloud.gen_cloud_GUI import GenereatePointCloud

from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination

from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs.point_cloud2 import create_cloud


def solve_eqn_prob(points, alg_name, link=None, center=None, viz_res=False):
    # define the problem
    problem = EllipsoidEquationOptProblem(center, points, viz_res, None, None)
        
    # define the weights of the points and of the volume, and the optimization algorithm
    if alg_name=="PatternSearch":
        problem.num_points_wt = 1
        problem.volume_wt = pow(10, int(math.log10(points.shape[0])-1))

        x0 = np.array([0.1, 0.1, 0.1, problem.center[0], problem.center[1], problem.center[2]])
        algorithm = PatternSearch(
                                delta=0.1,
                                rho=0.1,
                                step_size=0.1,
                                x0 = x0)
    
    elif alg_name=="GA":
        problem.num_points_wt = 1
        problem.volume_wt = pow(10, int(math.log10(points.shape[0])-1))

        x0 = np.array([[1, 1, 1, problem.center[0], problem.center[1], problem.center[2]]])
        pop = Population.new(X=x0)
        algorithm = GA(sampling=pop)
    
    
    elif alg_name=="PSO":
        problem.num_points_wt = 1
        problem.volume_wt = 1

        algorithm = PSO()


    termination = RobustTermination(
                        SingleObjectiveSpaceTermination(tol=pow(10,-6))     
                    )

    # solve the optimization problem
    res = minimize(problem=problem,
                   algorithm=algorithm,
                   termination=termination,
                   verbose = False,
                   seed = 1)
    
    if viz_res==True:
        print("Best solution found: \n\ta={:.4f}, b={:.4f}, c={:.4f}\n\txC={:.4f}, yC={:.4f}, zC={:.4f}".format(res.X[0], res.X[1], res.X[2], res.X[3], res.X[4], res.X[5]))
    return res

def create_marker_msg(points, link, res, center = None):
    # retrieve the solution of the opt problem
    a = res.X[0]
    b = res.X[1]
    c = res.X[2]

    # compute the center
    if center is None:
        center = np.array([np.mean([np.min(points[:,0]), np.max(points[:,0])]),
                            np.mean([np.min(points[:,1]), np.max(points[:,1])]),
                            np.mean([np.min(points[:,2]), np.max(points[:,2])])])
    
    marker_msg = Marker()
    marker_msg.header.frame_id = link
    marker_msg.header.stamp = rospy.Time.now()

    # set the shape
    marker_msg.type = 2
    marker_msg.id = 0

    # set the scale of the marker
    marker_msg.scale.x = 2*a
    marker_msg.scale.y = 2*b
    marker_msg.scale.z = 2*c

    # set the color
    marker_msg.color.r = 0.0
    marker_msg.color.g = 1.0
    marker_msg.color.b = 0.0
    marker_msg.color.a = 0.5

    # set the pose of the marker
    marker_msg.pose.position.x = center[0]
    marker_msg.pose.position.y = center[1]
    marker_msg.pose.position.z = center[2]
    marker_msg.pose.orientation.x = 0.0
    marker_msg.pose.orientation.y = 0.0
    marker_msg.pose.orientation.z = 0.0
    marker_msg.pose.orientation.w = 1.0

    return marker_msg

def create_center_msg(center, link):

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = link

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        pointcloud_msg = create_cloud(header, fields, (center,))

        return pointcloud_msg


if __name__=="__main__":
    # get the pointcloud points
    gen_cloud = GenereatePointCloud()
    gen_cloud.create_ros_node()
    gen_cloud.create_GUI()
    gen_cloud.from_extern = True

    # rospy.init_node('reachability_ellipsoid_publisher', anonymous=True)
    pub_ellipsoids = rospy.Publisher('/vis_reachability_ellipsoid', Marker, queue_size=10)
    pub_center = rospy.Publisher('/viz_ellipsoid_center', PointCloud2, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    # set the number of sample
    gen_cloud.num_samples = 20

    # create the point cloud
    gen_cloud.generate_point_cloud()

    # solve the optimization problem
    points = gen_cloud.points
    link = gen_cloud.point_cloud_orig_frame
    center = np.array([np.mean([np.min(points[:,0]), np.max(points[:,0])]),
                       np.mean([np.min(points[:,1]), np.max(points[:,1])]),
                       np.mean([np.min(points[:,2]), np.max(points[:,2])])])
    
    start = time.time()

    # alg_name = "PatternSearch"
    # alg_name = "GA"
    alg_name = "PSO"
    res = solve_eqn_prob(points, alg_name, link, center, viz_res=True)
    
    end = time.time() - start
    print("Solution found in {:.4f}s".format(end))
    
    center = np.array([res.X[3], res.X[4], res.X[5]])

    # create the marker msg
    marker_msg = create_marker_msg(points, link, res, center)

    # create the center message
    center_msg = create_center_msg(center, link)

    pub_cloud = gen_cloud.pub_points
    cloud_msg = gen_cloud.pointcloud_msg


    print("Start publishing message")
    while not rospy.is_shutdown():
        pub_ellipsoids.publish(marker_msg)
        rate.sleep()
        pub_center.publish(center_msg)
        rate.sleep()
        pub_cloud.publish(cloud_msg)
        rate.sleep()

