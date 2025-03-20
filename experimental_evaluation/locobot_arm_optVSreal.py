import numpy as np
import rospy
import time
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from generate_pointcloud.gen_cloud_GUI import GenereatePointCloud
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2

from opt_problem.problem_formulation import EllipsoidEquationOptProblem
from generate_pointcloud.gen_cloud_GUI import GenereatePointCloud

from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.default import DefaultMultiObjectiveTermination


def solve_eqn_prob(points, init_pop, center=None, n_gen=100, viz_res=False):
    # define the problem
    problem = EllipsoidEquationOptProblem(center, points, viz_res, None, None)

    problem.num_points_wt = 1
    problem.volume_wt = 1

    algorithm = PSO(pop_size=25,
                    sampling=init_pop)

    termination = RobustTermination(
        SingleObjectiveSpaceTermination(tol=pow(10, -6))
    )

    # solve the optimization problem
    res = minimize(problem=problem,
                   algorithm=algorithm,
                   termination=termination,
                   verbose=False,
                   seed=1,
                   save_history=True)

    if viz_res == True:
        print("Best solution found: \n\ta={:.4f}, b={:.4f}, c={:.4f}\n\txC={:.4f}, yC={:.4f}, zC={:.4f}".format(
            res.X[0], res.X[1], res.X[2], res.X[3], res.X[4], res.X[5]))
    return res


def compute_3DIoU(gt, pred):
    # ground truth params
    a_gt = gt[0]
    b_gt = gt[1]
    c_gt = gt[2]
    xc_gt = gt[3]
    yc_gt = gt[4]
    zc_gt = gt[5]

    # predicted params
    a = pred[0]
    b = pred[1]
    c = pred[2]
    xc = pred[3]
    yc = pred[4]
    zc = pred[5]

    # compute the volume of the intersection and of the union
    # using the Monte Carlo numeric integration method
    xmin = min([xc_gt-a_gt, xc-a])
    ymin = min([yc_gt-b_gt, yc-b])
    zmin = min([zc_gt-c_gt, zc-c])
    xmax = max([xc_gt+a_gt, xc+a])
    ymax = max([yc_gt+b_gt, yc+b])
    zmax = max([zc_gt+c_gt, zc+c])

    #  generate points
    num_points = int(1e7)
    x = np.random.uniform(xmin, xmax, size=(num_points, 1))
    y = np.random.uniform(ymin, ymax, size=(num_points, 1))
    z = np.random.uniform(zmin, zmax, size=(num_points, 1))

    #  compute the volume of the generation space
    V = abs(xmin-xmax)*abs(ymin-ymax)*abs(zmin-zmax)

    gt_mask = ((x-xc_gt)/a_gt)**2 + ((y-yc_gt)/b_gt)**2 + \
        ((z-zc_gt)/c_gt)**2 <= 1
    mask = ((x-xc)/a)**2 + ((y-yc)/b)**2 + ((z-zc)/c)**2 <= 1

    #  retrieve the points belonging to the union and count them
    union_mask = gt_mask | mask
    Nd_unions = np.sum(union_mask)

    # retrieve the points belonging to the intersection and count them
    inters_mask = gt_mask & mask
    Nd_inters = np.sum(inters_mask)

    vol_union = V*Nd_unions/num_points
    vol_inters = V*Nd_inters/num_points

    iou_3d = vol_inters/vol_union

    return iou_3d


# the real robot's RS is approximated as a sphere
# with a radius equal to the reachability declared
# for the last joint of the wrist
R = 0.570

# generate the point cloud
gen_cloud = GenereatePointCloud()
# gen_cloud.create_ros_node()
gen_cloud.from_extern = True
gen_cloud.urdf_file_path = "/home/rosario/Desktop/base_pose_opt_ws/src/reach_space_modeling/src/reach_space_modeling/generate_pointcloud/model/mobile_wx250s.urdf"
gen_cloud.parse_urdf()
gen_cloud.wrist_lst_j_name = "wrist_rotate"
gen_cloud.arm_lst_j_name = "elbow"
gen_cloud.arm_frt_j_name = "waist"
# gen_cloud.num_samples = 10
# gen_cloud.generate_point_cloud()
# print("Reachability point cloud created...")


# # TEST 1:   accuracy and IoU for different number of samples w.r.t to
# #           the approximation of the real shape
# # num_samples = np.arange(10, 101, 10)
# # num_samples = np.arange(5, 51, 5)
# num_samples = np.arange(5, 21, 5)
# opt_prob_res = np.zeros((num_samples.shape[0], 4))
# opt_prob_time = np.zeros(num_samples.shape)
# gen_cloud_time = np.zeros(num_samples.shape)

# for i, num in enumerate(num_samples):
#     # generate the point cloud
#     gen_cloud.num_samples = num
#     start = time.time()
#     gen_cloud.generate_point_cloud()
#     gen_cloud_time[i] = time.time() - start
#     print("Reachability point cloud with {:d} samples per joint created...".format(num))

#     # solve the optimization problem
#     points = gen_cloud.points
#     link = gen_cloud.point_cloud_orig_frame
#     center = np.array([np.mean([np.min(points[:, 0]), np.max(points[:, 0])]),
#                     np.mean([np.min(points[:, 1]), np.max(points[:, 1])]),
#                     np.mean([np.min(points[:, 2]), np.max(points[:, 2])])])

#     alg_name = "PSO"
#     start = time.time()
#     res = solve_eqn_prob(points, alg_name, link, center, viz_res=True)
#     opt_prob_time[i] = time.time() - start

#     # compute the relative percentage error for each parameter
#     opt_prob_res[i,0] = np.abs(res.X[0] - R)/R
#     opt_prob_res[i,1] = np.abs(res.X[1] - R)/R
#     opt_prob_res[i,2] = np.abs(res.X[2] - R)/R

#     # TODO: compute the IoU metric
#     gt = np.array([R, R, R, res.X[3], res.X[4], res.X[5]])
#     opt_prob_res[i,3] = compute_3DIoU(gt, res.X)


# # print a table with all the statistics
# print()
# for i, num in enumerate(num_samples):
#     print(num, end="\t")
#     for err in opt_prob_res[i,:3]:
#         print("{:.3f}".format(err*100) , end="\t")
#     print("{:.2f}".format(opt_prob_res[i,3]*100) , end="\t")
#     print("{:.2f}".format(gen_cloud_time[i]*1000), end="\t")
#     print("{:.2f}".format(opt_prob_time[i]), end="\t")
#     print("{:.2f}".format((gen_cloud_time[i]+opt_prob_time[i])), end="\t")
#     print(flush=True)

# # figure 1: relation between the number of sample per joints and the relative percentage errors
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.grid(linewidth=0.2)

# ax.plot(num_samples, opt_prob_res[:,0]*100, label="$A$")
# ax.scatter(num_samples, opt_prob_res[:,0]*100)

# ax.plot(num_samples, opt_prob_res[:,1]*100, label="$B$")
# ax.scatter(num_samples, opt_prob_res[:,1]*100)

# ax.plot(num_samples, opt_prob_res[:,2]*100, label="$C$")
# ax.scatter(num_samples, opt_prob_res[:,2]*100)

# plt.xlabel("$Number\ of\ samples\ per\ joint$", fontsize=12)
# plt.xticks(num_samples, [rf'${tick}$' for tick in num_samples])

# yticks = np.linspace(np.min(opt_prob_res[:,:3]*100, axis=(0,1)),
#                      np.max(opt_prob_res[:,:3]*100, axis=(0,1)),
#                      10)
# plt.yticks(yticks, ["{:.2f}".format(tick) for tick in yticks])

# plt.ylabel("$Relative\ percentage\ error\ [\%]$", fontsize=12)

# plt.title("$Relative\ percentage\ error\ vs.\ number\ of\ samples\ per\ joint$", fontsize=14)
# plt.legend()
# plt.show()

# # figure 2: relation between the number of sample per joints and 3DIoU
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.grid(linewidth=0.2)

# ax.plot(num_samples, opt_prob_res[:,3]*100, label="$3DIoU$")
# ax.scatter(num_samples, opt_prob_res[:,3]*100)

# plt.xlabel("$Number\ of\ samples\ per\ joint$", fontsize=12)
# plt.xticks(num_samples, [rf'${tick}$' for tick in num_samples])

# yticks = np.linspace(np.min(opt_prob_res[:,3]*100),
#                      np.max(opt_prob_res[:,3]*100),
#                      10)
# plt.yticks(yticks, ["{:.2f}".format(tick) for tick in yticks])

# plt.ylabel("$3DIoU\ [\%]$", fontsize=12)

# plt.title("$3DIoU\ metric\ vs.\ number\ of\ samples\ per\ joint$", fontsize=14)
# plt.legend()
# plt.show()

# # figure 3: relation between the number of sample per joints and different times
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.grid(linewidth=0.2)

# ax.plot(num_samples, gen_cloud_time*1000, label="$Point\ cloud\ generation$")
# ax.scatter(num_samples, gen_cloud_time*1000)

# ax.plot(num_samples, opt_prob_time*1000, label="$Optimization\ problem\ [PSO]$")
# ax.scatter(num_samples, opt_prob_time*1000)

# ax.plot(num_samples, (gen_cloud_time+opt_prob_time)*1000, label="$Total$")
# ax.scatter(num_samples, (gen_cloud_time+opt_prob_time)*1000)

# plt.xlabel("$Number\ of\ samples\ per\ joint$", fontsize=12)
# plt.xticks(num_samples, [rf'${tick}$' for tick in num_samples])

# yticks = np.linspace(np.min(gen_cloud_time*1000),
#                      np.max((gen_cloud_time+opt_prob_time)*1000),
#                      10)
# plt.yticks(yticks, ["{:.2f}".format(tick) for tick in yticks])

# plt.ylabel("$Point\ cloud\ generation\ time[ms]$", fontsize=12)

# plt.title("$Point\ cloud\ generation\ time\ vs.\ number\ of\ samples\ per\ joint$", fontsize=14)
# plt.legend()
# plt.show()


# TEST 2:   solve the optimization problem with random intialization
#           but fixed number of evaluations

num_samples = [5, 10, 20]
colors_soft = ["#a5c5fc", "#fca5a5", "#90fc9d"]
colors = ["blue", "red", "green"]
num_iter = 1000

fig = plt.figure(figsize=(800/100, 500/100))
plt.rcParams['text.usetex'] = True
ax = fig.add_subplot()

for i, num in enumerate(num_samples):
    gen_cloud.num_samples = num
    gen_cloud.generate_point_cloud()  # solve the optimization problem
    points = gen_cloud.points
    link = gen_cloud.point_cloud_orig_frame
    center = np.array([np.mean([np.min(points[:, 0]), np.max(points[:, 0])]),
                       np.mean([np.min(points[:, 1]), np.max(points[:, 1])]),
                       np.mean([np.min(points[:, 2]), np.max(points[:, 2])])])

    max_len = 0
    all_f_values = []
    # ax = fig.add_subplot(3,1,i+1)
    # solve the problem 'num_iter' times with random initial conditions
    for j in tqdm(range(num_iter),
                  total=num_iter,
                  desc="Optimization problem solved",
                  ncols=100):

        init_pop = np.zeros((25, 6))
        init_pop[:, :3] = np.random.uniform(0, 1, (25, 3))
        init_pop[:, 3:] = np.random.uniform(-1, 1, (25, 3))

        res = solve_eqn_prob(points, init_pop, center, n_gen=21, viz_res=j==(num_iter-1))
        # res = solve_eqn_prob(points=points, alg_name="PSO", link=None, center=center, viz_res=j==(num_iter-1))
        f_values = [abs(entry.opt.get("F")) for entry in res.history]
        f_values = np.reshape(f_values, (len(f_values,)))
        
        if max_len < len(f_values):
            max_len = len(f_values)
        
        all_f_values.append(f_values)
    
    for j, f_values in enumerate(all_f_values):
        f_values = np.pad(f_values, (0, max(0, max_len - len(f_values))), mode='edge')
        all_f_values[j] = f_values
        ax.plot(np.arange(1, len(f_values)+1), f_values, color=colors_soft[i])

    mean_f_val = np.mean(all_f_values, axis=0)
    min_f_val = np.min(all_f_values, axis=0)
    max_f_val = np.max(all_f_values, axis=0)
    ax.plot(np.arange(1, len(mean_f_val)+1), mean_f_val, color=colors[i],
            label="{:d} Samples per joint".format(num))
            # label="Mean val.")
    ax.fill_between(np.arange(1, len(min_f_val)+1),
                    min_f_val, max_f_val, color=colors_soft[i])

ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.tick_params('both', labelsize=14)

# plt.xticks(np.arange(1, len(f_values)+1))
plt.xlabel("Number of iterations", fontsize=16)
plt.ylabel("Value of the objective function", fontsize=16)
plt.yscale("log")
plt.xscale("log")
plt.grid(linewidth=0.2)
plt.legend(fontsize=14, loc="lower right")
plt.savefig("plot_1e-6.svg", format="svg")
plt.show()
