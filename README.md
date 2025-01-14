# Robotic manipulators reachability space modeling

This repository provides tools and resources for modeling and analyzing the reachability space of robotic manipulators using ellipsoid equations. A first tool allows to obtain the **point cloud representing a set of the reachable points** using only the kinematic model of the desired robot. This method is applicable to **different kind of manipulators**, with fixed or mobile bases. For the latters, the additional Degree of Freedoms (DOFs) introduced by the mobile base are not considered. Hence an estimation, and a subsequent model, of the reachability space of the manipulator depending on the current pose of the base is obtained.

Starting from the point cloud obtained before, **an ellipsoid equation is obtained**. The parameters of the ellipsoid, namely the coordinates of the center and the lenghts of axes, are obtained as a result of an** optimization problem**. Two different tools to solve the optimization problem can be used:
* a proper minimization problem
* different variants of the PointNet models.

The code contained in this repository has been used for the experimental evaluation in paper "Modeling the Reachability Space of Robotic Manipulators through Ellipsoid Equations".

**Keywords**: Robotic Manipulators, Reachability Space, Ellipsoid Modeling, Optimization.

## Point cloud generation
This tool allows to generate a point cloud representing the reachability space starting from the URDF of a robot.

To obtain a point cloud representing the points that can be reached by the desired manipulator, it is possible to run the Python code *gen_cloud_GUI.py* in the *generate_pointcloud* folder:

```python
python3 generate_pointcloud/gen_cloud_GUI.py
```

### Notice
The use of this script alone is intended for visualization purposes only. Before creating and visualizing the GUI (Graphical User Interface) to generate the desired point cloud, **a ROS node named *reachability_pointcloud_publisher* is created**. For this reason, **the ROS master node must be running** before the desired point cloud can be generated using the proposed GUI.

### Explanation
The center of the code is represented by the GUI showed below.

<div align="center">
    <img src="images/GUI_img.jpg" width="600">
</div>

It is possible to identify a total of 6 sections, as highlighted in the following picture.

<div align="center">
    <img src="images/GUI_img_commented.jpg" width="600">
</div>

In more details:

1. The first component constitues of a search bar and a *Browse* button that allow to select the desired URDF file. Some URDF of well known manipualtors, both with fixed and mobile bases, are provided in the *generate_pointcloud/models* folder.

2. The second section is composed by 3 drop-down menus. Each of them contains a list of the names of the actuated joints of the robot. By clicking on one item of the list, it is possible to select the *last joint of the wrist*, the *last joint of the arm* and the *first joint of the arm*. It is important to notice that **all the points constituting the point cloud are computed considering as origin the one of the reference frame of the joint selected as first joint of the arm**.

3. The third component allows to select the number of samples that will compose the span of possible values for each joint. As a consequence, if the manipulator is composed by a total of *j* joints, and a total of *N* values for each joint are considered, the total number of points constituting the point cloud will be equal to *N^j*.

4. The *Generate* button allows the computataion of the point cloud.

5. The *Publish* button allows the ROS node created at the beginning (*reachability_pointcloud_publisher*) to publish a **PointCloud2 ROS message**, containing the newly generated point cloud, on a topic named **/reachability_pointcloud**.

6. A read-only text box is used to show important information during the generation of the point cloud, as well as error and/or debugging messages.

### Example
As an example, the point cloud obtained for the ur5e manipulator from the Universal Robotics (whose URDF is available in the *generate_pointcloud/models* folder) is showed. The visualization of the robot model along with the newly computed point cloud is obtained exploiting **Rviz for ROS Noetic**.

Please note that, as cited above, using the *gen_cloud_GUI.py* script will result only in the computation of the point cloud and eventually its visualization. As a result, the ellipsoid is not visible, since its parameters are not computed.

<div align="center">
    <img src="images/ur5e_pointcloud.png" width="1000">
</div>







## Optimization problem
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
    <img src="images/locobot_ellipsoid.png" width="800">
</div>








## Variants of the PointNet
This tool allow to obtain the parameters of the ellipsoid equation using different variants of the PointNet model.








## Experimental evaluation





