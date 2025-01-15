## Optimization problem

### Table of contents
+ [Introduction](#introduction)
    + [Problem definition](#problem-definition)
    + [Usage](#usage)
    + [Example](#example-1)

### Introduction
The first way to compute the parameters of the ellipsoid is by solving a proper optimization problem. For more details about the design of the optimization problem please refere to the paper.

All the files needed to obtain the equation of the ellipsoid enclosing the reachability space are contained in the ***opt_problem*** folder.

### Problem definition
The ***problem_formulation.py*** file contains the definition of a class where the optimization problem is defined, named **EllipsoidEquationOptProblem**, by extending the *ElementwiseProblem* class of the *pymoo* library.

When an object of this class is instantiated, a set of few operations are completed, as specified in the *\__init__* method. In order:
1. the points building the point cloud of the desired manipulator are retrieved;
2. if not provided, an estimation of the center of the point cloud as the mean point is computed;
3. the parameters of the optimization problem are set. More in details, the number of optimization variables, the number of objective functions, the inequality contraints, the lower and upper bounds of the optimization variables.

A second method, namely *_evaluate*, is defined. It is the function that is called at every iteration of the optimization problem.  It retrieves the current solution and computes the values of the objective function until now.

### Usage
It is possible to obtain the equation of the ellipsoid by running the script *eqn_solv_opt.py*:
```
python3 opt_problem/eqn_solv_opt.py
```
This script is based on the GUI described above to select the URDF of the robot, to select the important joints, the number of samples per joint and, finally, to genereate the point cloud exploited by the optimization problem.

It is important to notice that the parameters characterizing the equation of the ellipsoid are computed as soon as the point cloud is obtained and after the GUI has been closed.

It is possible to select the desired optimization algoritm by changing the value of the variable *alg_name*. The possible strings are:
* ***"PatternSearch"*** to run the optimization problem using the Pattern Search algorithm;
* ***"GA"*** to run the optimization problem exploiting the Genetic Algorithm;
* ***"PSO"*** to run the optimization problem using the Particle Swarm Optimization Algorithm.

Once the optimization problem is solved, the parameters of the best solution found are printed on the screen. In addition to that, 2 ROS messages are created and published:
1. A ***Marker*** message is published by the *reachability_pointcloud_publisher* node (initialized when the GUI is created) on a topic named ***/vis_reachability_ellipsoid***. The marker's position and the lenght of its axes are equal to the parameters characterizing the ellipsoid equation. In this way, a visualization of the computed equation can be obtained and displayed in Rviz.

2. A ***PointCloud2*** message is published by the *reachability_pointcloud_publisher* node on a topic named ***/vis_ellipsoid_center***. It is used to show the center of the ellipsoid.

### Example
As an example, here it is visualized the ellipsoid (in green) enclosing the reachability space of the LoCoBot WX250s from Trossen Robotics (whose URDF can be found in the *generate_pointcloud/models* folder), along with the point cloud (in red).

<div align="center">
    <img src="../images/locobot_ellipsoid.png" width="800">
</div>