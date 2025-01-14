import os
import numpy as np
import h5py
import random
from matplotlib import pyplot as plt

# load the h5 file in h5file_path
def load_h5_file(h5file_path="./train_dataset.h5", see_keys=False):
    # check h5file_path is only a file name, a relative path or a full path
    if h5file_path.split('/')[0]!='.' and h5file_path.split('/')[0]!='..':  # just a file name
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        h5file_path = os.path.join(curr_dir, h5file_path)
    elif h5file_path.split('/')[0]=='.' or h5file_path.split('/')[0]=='..': # relative path
        dir = os.path.dirname(os.path.abspath(__file__))
        dir_list = h5file_path.split('/')
        h5file_name = dir_list[-1]
        i=0
        while dir_list[i]!=h5file_name:
            if dir_list[i]=='..':
                dir = os.path.dirname(dir)
            elif dir_list[i]!='..' and dir_list[i]!='.':
                dir = os.path.join(dir, dir_list[i])
            i+=1
        h5file_path = os.path.join(dir, h5file_name)
            
    # load the h5 file to read it
    h5_file = h5py.File(h5file_path, "r")

    if see_keys:
        # print all keys
        for key in h5_file.keys():
            print(key)
    
    return h5_file

# load the dataset
def load_dataset(h5_file):
    # load the dataset
    dataset = h5_file.get("train_dataset")
    dataset = dataset[:,:,:]
    dataset = np.transpose(dataset, (2,1,0))

    # load the labels
    labels = h5_file.get("labels")
    labels = labels[:,:]
    labels = np.transpose(labels, (1,0))

    # create a CLASS_MAP to convert from integer to string
    class_map = {0: "Ellipsoid", 1: "Sphere"}

    return dataset, labels, class_map

# extract a random point cloud an
def show_sample(dataset, labels, class_map):
    # define a random valid index
    num_elem = dataset.shape[0]
    index = random.randint(0, num_elem-1)

    # extract the point cloud
    points = dataset[1,:,:]

    # visualize the point cloud and its label
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_title("label: {:}".format(class_map[labels[1,1]]))
    # ax.set_axis_off()
    plt.show()

# function to reduce the number of points for each point cloud, and 
# to split the entire dataset in train and test sets
def parse_dataset(dataset, labels, shuffle=True, des_num_points=2048, perc_train=0.8, perc_val=0.2):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []

    # retrieve the number of point clouds in the dataset
    num_point_clouds = dataset.shape[0]

    # retrieve the number of points
    curr_num_points = dataset.shape[1]

    last_ind_train = int(num_point_clouds * perc_train)

    # if the number of desired points is less then the actual num_points
    # sample each point cloud in the datsaset
    if des_num_points < curr_num_points:
        # create a random selection of the indeces
        random_indices = np.arange(curr_num_points)
        random.shuffle(random_indices)
        random_indices = random_indices[:des_num_points]
        dataset = dataset[:, random_indices, :]

    # split the dataset
    train_points = dataset[:last_ind_train, :, :]
    train_labels = labels[:last_ind_train, :]
    test_points = dataset[last_ind_train:, :, :]
    test_labels = labels[last_ind_train:, :]

    # if required, shuffle the dataset and the labels
    if shuffle==True:
        # shuffle the train dataset
        shuffled_index = np.arange(train_points.shape[0])
        random.shuffle(shuffled_index)
        train_points = train_points[shuffled_index, :, :]
        train_labels = train_labels[shuffled_index, :]

        # shuffle the test dataset
        shuffled_index = np.arange(test_points.shape[0])
        random.shuffle(shuffled_index)
        test_points = test_points[shuffled_index, :, :]
        test_labels = test_labels[shuffled_index, :]

    return(
        train_points,
        train_labels,
        test_points,
        test_labels
    )

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


def get_dataset(h5file_path="train_dataset.h5", shuffle=True, des_num_points=2048, perc_train=0.8, perc_val=0.2, vis_sample=False):
    # load the h5file
    h5_file = load_h5_file(h5file_path)

    # load the dataset
    dataset, labels, class_map = load_dataset(h5_file)

    # show a sample (only if desired)
    if vis_sample==True:
        show_sample(dataset, labels, class_map)

    # parse the dataset and split it
    train_points, train_labels, test_points, test_labels = parse_dataset(dataset, labels, shuffle=shuffle,
                                                                         des_num_points=des_num_points,
                                                                         perc_train=perc_train,
                                                                         perc_val=perc_val)
    
    train_points = data_normalization(train_points)
    test_points = data_normalization(test_points)

    return(
        train_points,
        train_labels,
        test_points,
        test_labels,
        class_map
    ) 

if __name__=='__main__':
    NUM_POINTS = 2048
    shape = (100, NUM_POINTS, 3)
    dataset = np.random.random(shape)
    labels = np.zeros((100, 1))
    
    train_points, test_points, train_labels, test_labels = parse_dataset(dataset, labels, des_num_points=1024)




