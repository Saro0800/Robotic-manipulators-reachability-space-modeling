# Robotic manipulators reachability space modeling

This repository provides tools and resources for modeling and analyzing the reachability space of robotic manipulators using ellipsoid equations. A first tool allows to obtain the **point cloud representing a set of the reachable points** using only the kinematic model of the desired robot. This method is applicable to **different kind of manipulators**, with fixed or mobile bases. For the latters, the additional Degree of Freedoms (DOFs) introduced by the mobile base are not considered. Hence an estimation, and a subsequent model, of the reachability space of the manipulator depending on the current pose of the base is obtained.

Starting from the point cloud obtained before, **an ellipsoid equation is obtained**. The parameters of the ellipsoid, namely the coordinates of the center and the lenghts of axes, are obtained as a result of an** optimization problem**. Two different tools to solve the optimization problem can be used:
* a proper minimization problem
* different variants of the PointNet models.

The code contained in this repository has been used for the experimental evaluation in paper "Modeling the Reachability Space of Robotic Manipulators through Ellipsoid Equations".

**Keywords**: Robotic Manipulators, Reachability Space, Ellipsoid Modeling, Optimization.

## Point cloud generation
This tool allows to generate a point cloud representing the reachability space starting from the URDF of a robot.

To obtain a point cloud representing the points that can be reached by the desired manipulator, it is possible to run the Python code *gen_cloud_GUI.py* in the *generate_pointcloud* folder.

```python
python3 generate_pointcloud/gen_cloud_GUI.py
```









## Optimization problem
This tool allow to obtain the parameters of the ellipsoid equation fallowing a minimization fashion.








## Variants of the PointNet
This tool allow to obtain the parameters of the ellipsoid equation using different variants of the PointNet model.








## Experimental evaluation





