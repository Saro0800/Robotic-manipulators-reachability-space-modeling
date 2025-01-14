import os
import shutil
import argparse
import tensorflow as tf
import numpy as np
import sage
from tensorflow import keras
from dataset.load_dataset import get_dataset
import models
from matplotlib import pyplot as plt
from keras.losses import MeanSquaredError, Huber, LogCosh, MeanAbsoluteError
from mpl_toolkits.mplot3d import axes3d

import models.pointNet_cls


parser = argparse.ArgumentParser(
                    prog='train_process',
                    description='Script to launch the training of the esired PointNet')

# parser.add_argument("--model_type", 
#                     default="pointnet_eqn_full",
#                     choices=["pointnet_eqn_full", "pointNet_cls", "pointnet_eqn_tiny", "pointnet_eqn_micro"],
#                     help="Choose the model to use.")
parser.add_argument("--restore_from_last", type=bool,  default=False,
                    choices=[True, False],
                    help="Set to True if you want to restore the last training from the latest epoch.")
parser.add_argument("--restore_from_best", type=bool,  default=False,
                    choices=[True, False],
                    help="Set to True if you want to restore the last training from the best results.")
parser.add_argument("--num_points", type=int, default=2048, help="Number of points for each point cloud")
parser.add_argument("--gen_rad", type=int, default=3, help="Radius of the area where to generate points")

args = parser.parse_args()

if args.restore_from_last==True and args.restore_from_best==True:
    print("Only one between --restore_from_last and --restore_from_best can be True")
    assert(False)
elif args.restore_from_last==False and args.restore_from_best==False:
    print("One between --restore_from_last and --restore_from_best must be True")
    assert(False)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPU available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# generate the equation's parameter
global A_gt, B_gt, C_gt, R_gt, xc_gt, yc_gt, zc_gt

def eqn(x,y,z):
    res = ((x-xc_gt)*A_gt)**2 + ((y-yc_gt)*B_gt)**2 + ((z-zc_gt)*C_gt)**2 - R_gt**2
    return  res

def eqn_pred(x,y,z, param):
    A = param[1]
    B = param[2]
    C = param[3]
    R = param[4]
    xc = param[5]
    yc = param[6]
    zc = param[7]

    return ((x-xc)*A)**2 + ((y-yc)*B)**2 + ((z-zc)*C)**2 - R**2

def plot_implicit(fn_gt, fn_pred, param, fig, index, title, bbox=(-2.5,2.5)):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    ax = fig.add_subplot(1,1,index, projection='3d')
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 50) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn_gt(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z', colors=['g'])
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn_gt(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y', colors=['g'])

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn_gt(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x', colors=['g'])

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn_pred(X,Y,z, param)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z', colors=['b'])
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn_pred(X,y,Z, param)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y', colors=['b'])

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn_pred(x,Y,Z, param)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x', colors=['b'])

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title(title)

# get a sample from the dataset
train_ds, train_labels, test_ds, test_labels, class_map = get_dataset("new_train_dataset_5000.h5", shuffle=True, vis_sample=False)
label = test_labels[0,:]
label[0] -= 1

# A_gt = np.random.uniform(0,1,1)[0]
# B_gt = np.random.uniform(0,1,1)[0]
# C_gt = np.random.uniform(0,1,1)[0]
# R_gt = np.random.uniform(0,1,1)[0]
# xc_gt = np.random.uniform(-1.5, 1.5, 1)[0]
# yc_gt = np.random.uniform(-1.5, 1.5, 1)[0]
# zc_gt = np.random.uniform(-1.5, 1.5, 1)[0]

# points = []
# while len(points) < args.num_points:
#     gen_points = np.random.uniform(-args.gen_rad, args.gen_rad, (args.num_points,3))
#     x = gen_points[:,0]
#     y = gen_points[:,1]
#     z = gen_points[:,2]
#     sel_points = gen_points[(eqn(x,y,z)<=0)]
#     points.extend(sel_points)
#     print(len(points))
# points = np.array(points)
# points = np.expand_dims(points[:args.num_points, :], 0)

# create the label
# label = np.array([0.0, A_gt, B_gt, C_gt, R_gt, xc_gt, yc_gt, zc_gt])

# load the models' architecture
pointNet_cls = models.pointNet_cls.get_model()

saves = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")

# load their weights
if args.restore_from_last==True:
    print("Loading last sets of weights obtained")
    save_dir = os.path.join(saves, pointNet_cls.name, "last/")
    pointNet_cls.load_weights(save_dir)
    print("\tpointNet_cls: last weights loaded")

elif args.restore_from_best==True:
    print("Loading best sets of weights obtained")
    save_dir = os.path.join(saves, pointNet_cls.name, "best/")
    pointNet_cls.load_weights(save_dir)
    print("\tpointNet_cls: best weights loaded")

for i in range(4):
    index = np.random.random_integers(low=0, high=test_ds.shape[0], size=(1))[0]
    points = test_ds[index, :, :]
    points = np.expand_dims(points, 0)
    label = test_labels[index, :]

    A_gt = label[1]
    B_gt = label[2]
    C_gt = label[3]
    R_gt = label[4]
    xc_gt = label[5]
    yc_gt = label[6]
    zc_gt = label[7]

    preds = pointNet_cls.predict(points)

    label = np.expand_dims(label, 1)
    real_preds = np.expand_dims(preds[0], 1)
    preds = np.zeros(label.shape)
    preds[[1,2,3,5,6,7]] = real_preds
    preds[0] = label[0]
    preds[4] = label[4]

    error = np.expand_dims(keras.losses.mean_squared_error(label, preds), 1)

    small_error_perc = 100 * np.abs((error)/label)

    # pront the results and the errors
    print()
    print("   label:\t   small_pred:\t   small_error:\t   small_error_perc:")
    stacked = np.hstack((label, preds, error, small_error_perc))
    for i in range(stacked.shape[0]):
        print("   {:.4f}\t   {:.4f}\t   {:.4f}\t   {:.4f}".format(stacked[i,0],stacked[i,1],stacked[i,2],stacked[i,3]))

    # plot the shapes
    fig = plt.figure()
    plot_implicit(eqn, eqn_pred, preds, fig, 1, title="Small model")
    plt.show()
    plt.pause(0)
    plt.close(fig)

