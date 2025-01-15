## Point cloud generation

### Table of contents
+ [Introduction](#introduction)
    + [Notice](#notice)
    + [Usage](#usage)
    + [Example](#example)

### Introduction
This tool allows to generate a point cloud representing the reachability space starting from the URDF of a robot.

To obtain a point cloud representing the points that can be reached by the desired manipulator, it is possible to run the Python code *gen_cloud_GUI.py* in the *generate_pointcloud* folder:

```python
python3 generate_pointcloud/gen_cloud_GUI.py
```

### Notice
The use of this script alone is intended for visualization purposes only. Before creating and visualizing the GUI (Graphical User Interface) to generate the desired point cloud, **a ROS node named *reachability_pointcloud_publisher* is created**. For this reason, **the ROS master node must be running** before the desired point cloud can be generated using the proposed GUI.

### Usage
The center of the code is represented by the GUI showed below.

<div align="center">
    <img src="../images/GUI_img.jpg" width="600">
</div>

It is possible to identify a total of 6 sections, as highlighted in the following picture.

<div align="center">
    <img src="../images/GUI_img_commented.jpg" width="600">
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
    <img src="../images/ur5e_pointcloud.png" width="1000">
</div>