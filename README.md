# Robotic manipulators reachability space modeling

## Table of contents
+ [Introduction](#introduction)
+ [Prerequisites](#prerequisites)
+ [Installation](#installation)
+ [Cite us](#cite-us)


## Introduction
This repository provides tools and resources for modeling and analyzing the reachability space of robotic manipulators using ellipsoid equations. A first tool allows to obtain the **point cloud representing a set of the reachable points** using only the kinematic model of the desired robot. This method is applicable to **different kind of manipulators**, with fixed or mobile bases. For the latters, the additional Degree of Freedoms (DOFs) introduced by the mobile base are not considered. Hence an estimation, and a subsequent model, of the reachability space of the manipulator depending on the current pose of the base is obtained.

Starting from the point cloud obtained before, **an ellipsoid equation is obtained**. The parameters of the ellipsoid, namely the coordinates of the center and the lenghts of axes, are obtained as a result of an** optimization problem**. Two different tools to solve the optimization problem can be used:
* a proper minimization problem
* different variants of the PointNet models.

**Documentation of each module can be found inside each specific folder.**

The code contained in this repository has been used for the experimental evaluation in paper "Modeling the Reachability Space of Robotic Manipulators through Ellipsoid Equations".

**Keywords**: Robotic Manipulators, Reachability Space, Ellipsoid Modeling, Optimization.

## Prerequisites
The code available in this repository has been tested and developed on **Ubuntu 20.04 LTS**.

Before installing and using this repo, please be sure to meet the following prerequisites:
1. **Python version**: The code proposed here has been developed and tested using Python 3.8.10. You can download it from [here](https://www.python.org/downloads/). For higher version of Python, the libraries used may have received major updates and some errors may arise.
2. **ROS**: You need to have **ROS Noetic** installed. For the installation, please check the [official website](https://wiki.ros.org/noetic/Installation/Ubuntu).
3. **Rviz**: although not strictly necessar, it is suggested to have the Rviz ROS package installed, in order to try all the features of the proposed code. You can install by executing the following command:
    ```
    sudo apt-get install ros-noetic-rviz
    ```
4. **Needed libraries**: please make sure to install all the libraries needed to run the code by executing the command:
    ```
    pip install -r requirements.txt
    ```

## Installation

## Cite us









