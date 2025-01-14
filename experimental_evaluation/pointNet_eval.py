import sys
sys.path.append("..")

import os
import shutil
import h5py
import argparse
import tensorflow as tf
import numpy as np
import h5py
import time
from tensorflow import keras
from pointNet.dataset.load_dataset import get_dataset
from pointNet.models import pointNet_cls, pointNet_cls_small, pointNet_cls_tiny
from matplotlib import pyplot as plt
from keras.losses import MeanSquaredError, Huber, LogCosh, MeanAbsoluteError

parser = argparse.ArgumentParser(
                    prog='train_process',
                    description='Script to launch the training of the esired PointNet')

parser.add_argument("--model_type", 
                    default="pointnet_cls",
                    choices=["pointnet_cls", "pointnet_cls_small", "pointnet_cls_tiny"],
                    help="Choose the model to use.")
                    
parser.add_argument("--restore_from_last", type=bool,  default=False,
                    choices=[True, False],
                    help="Set to True if you want to restore the last training from the latest epoch.")

parser.add_argument("--restore_from_best", type=bool,  default=True,
                    choices=[True, False],
                    help="Set to True if you want to restore the last training from the best results.")

parser.add_argument("--num_points", type=int, default=2048, help="Number of points for each point cloud")
parser.add_argument("--gen_rad", type=int, default=3, help="Radius of the area where to generate points")

def data_normalization(array):
    x_min = np.min(array[:,:,0])
    y_min = np.min(array[:,:,1])
    z_min = np.min(array[:,:,2])

    x_max = np.max(array[:,:,0])
    y_max = np.max(array[:,:,1])
    z_max = np.max(array[:,:,2])

    norm_array = np.zeros(array.shape)

    norm_array[:,:,0] = (array[:,:,0] - x_min)/(x_max - x_min)
    norm_array[:,:,1] = (array[:,:,1] - y_min)/(y_max - y_min)
    norm_array[:,:,2] = (array[:,:,2] - z_min)/(z_max - z_min)

    return norm_array

if __name__=="__main__":
    args = parser.parse_args()

    if args.restore_from_last==True and args.restore_from_best==True:
        print("Only one between --restore_from_last and --restore_from_best can be True")
        assert(False)
    elif args.restore_from_last==False and args.restore_from_best==False:
        print("One between --restore_from_last and --restore_from_best must be True")
        assert(False)

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # print("GPU available: ", len(physical_devices))
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # load the models' architecture
    if args.model_type == "pointnet_cls":
        model = pointNet_cls.get_model()
    elif args.model_type == "pointnet_cls_small":
        model = pointNet_cls_small.get_model()
    elif args.model_type == "pointnet_cls_tiny":
        model = pointNet_cls_tiny.get_model()

    saves = "../pointNet/saves"

    # load the weights
    if args.restore_from_last==True:
        print("Loading last sets of weights obtained")
        save_dir = os.path.join(saves, model.name, "last/")
        model.load_weights(save_dir)
        print("\tpointNet_cls: last weights loaded")

    elif args.restore_from_best==True:
        print("Loading best sets of weights obtained")
        save_dir = os.path.join(saves, model.name, "best/")
        model.load_weights(save_dir)
        print("\tpointNet_cls: best weights loaded")

    # load the .h5 file
    h5file_path = "./data/evaluation_Dataset_1000.h5"
    h5_file = h5py.File(h5file_path, "r")

    # extract the equation parameters and the corresponding point clouds
    equation_params = h5_file.get("labels")[:,:].transpose((1,0))
    pointclouds = h5_file.get("train_dataset")[:,:,:].transpose((2,1,0))

    # init the result data structure
    results = np.zeros(equation_params.shape)
    results[:,(0,4)] = equation_params[:,(0,4)]

    equation_params = equation_params[:,(1,2,3,5,6,7)]
    num_elements = equation_params.shape[0]

    # data normalization
    pointclouds_norm = data_normalization(pointclouds)

    # init the error data structure:
    errors = np.zeros(equation_params.shape)

    # init the time data structure
    times = np.zeros(equation_params.shape[0])

    for i in range(num_elements):
        print(i, flush=True)
        points_norm = pointclouds_norm[i,:,:]
        points_norm = np.expand_dims(points_norm, 0)

        start = time.time()
        preds = model.predict(points_norm)
        times[i] = time.time() - start
        print(times[i])

        # compute a prediction of the center coordinates
        points = pointclouds[i,:,:]
        xc = np.mean([np.min(points[:,0]), np.max(points[:,0])])
        yc = np.mean([np.min(points[:,1]), np.max(points[:,1])])
        zc = np.mean([np.min(points[:,2]), np.max(points[:,2])])

        # store the result
        results[i,1] = preds[0,0]
        results[i,2] = preds[0,1]
        results[i,3] = preds[0,2]
        results[i,5] = xc
        results[i,6] = yc
        results[i,7] = zc

        # compute the relative error
        errors[i,0] = np.abs(preds[0,0]-equation_params[i,0])/equation_params[i,0]
        errors[i,1] = np.abs(preds[0,1]-equation_params[i,1])/equation_params[i,1]
        errors[i,2] = np.abs(preds[0,2]-equation_params[i,2])/equation_params[i,2]
        errors[i,3] = np.abs(xc-equation_params[i,3])/np.abs(equation_params[i,3])
        errors[i,4] = np.abs(yc-equation_params[i,4])/np.abs(equation_params[i,4])
        errors[i,5] = np.abs(zc-equation_params[i,5])/np.abs(equation_params[i,5])
    
    # compute different errors metrics for each predicted parameter
    mean_err = np.mean(errors, 0)*100
    median_err = np.median(errors,0)*100
    min_err = np.min(errors,0)*100
    max_err = np.max(errors,0)*100
    var_err = np.var(errors,0)

    # compute the formatted string
    formatted_mean_err = "\t".join(f"{err:08.4f}%" for err in mean_err)
    formatted_median_err = "\t".join(f"{err:08.4f}%" for err in median_err)
    formatted_min_err = "\t".join(f"{err:08.4f}%" for err in min_err)
    formatted_max_err = "\t".join(f"{err:08.4f}%" for err in max_err)

    print("\t\tA\t\tB\t\tC\t\tXc\t\tYc\t\tZc")
    print(f"Mean Error:\t{formatted_mean_err}")
    print(f"Median Error:\t{formatted_median_err}")
    print(f"Min Error:\t{formatted_min_err}")
    print(f"Max Error:\t{formatted_max_err}")

    # store all the results in a .h5 file
    res_h5_path = './results/'+model.name+'_result_'+h5file_path.split('/')[-1]
    with h5py.File(res_h5_path, 'w') as h5file:
        # save the times
        h5file.create_dataset('times', data=times)

        # save the results to the file
        h5file.create_dataset('result', data=results)

        # save the errors to the file
        h5file.create_dataset('errors', data=errors)
        
        # save the error metrics
        h5file.create_dataset('mean_error', data=mean_err)
        h5file.create_dataset('median_error', data=median_err)
        h5file.create_dataset('min_error', data=min_err)
        h5file.create_dataset('max_error', data=max_err)
