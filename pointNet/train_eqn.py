import os
import shutil
import argparse
import tensorflow as tf
import numpy as np
import pickle5 as pickle
from tensorflow import keras
from dataset.load_dataset import get_dataset
import models
from matplotlib import pyplot as plt
from keras.losses import MeanSquaredError, Huber, LogCosh, MeanAbsoluteError

import models.pointNet_cls
import models.pointNet_cls_small
import models.pointNet_cls_tiny

keras.utils.set_random_seed(8421)

parser = argparse.ArgumentParser(
                    prog='train_process',
                    description='Script to launch the training of the esired PointNet')

parser.add_argument("--h5file",
                    default="dataset_10000.h5",
                    help="Path to the dataset to load.")
parser.add_argument("--model_type", 
                    default="pointnet_cls",
                    choices=["pointnet_cls", "pointnet_cls_small", "pointnet_cls_tiny"],
                    help="Choose the model to use.")
parser.add_argument("--restore_training", type=bool, default=False,
                    choices=[True, False],
                    help="Set to True if you want to restore the latest training.")
parser.add_argument("--restore_from_last", type=bool,  default=False,
                    choices=[True, False],
                    help="Set to True if you want to restore the last training from the latest epoch.")
parser.add_argument("--restore_from_best", type=bool,  default=False,
                    choices=[True, False],
                    help="Set to True if you want to restore the last training from the best results.")
parser.add_argument("--save_last", type=bool, default=True,
                    choices=[True, False],
                    help="Set to True if you want to sasve the last epoch.")
parser.add_argument("--save_best", type=bool, default=True,
                    choices=[True, False],
                    help="Set to True if you want to save the best set of weights.")
parser.add_argument("--epochs", "-e", default=10, type=int,
                    help="Integer number defining the number of training epochs")

args = parser.parse_args()
print(args)

# Use hardware accelerator for training
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPU available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# get the train and test datasets
print("Loading the dataset... ", end="", flush=True)
train_ds, train_labels, test_ds, test_labels, class_map = get_dataset(h5file_path = args.h5file, shuffle=True, vis_sample=False)
print("done")

# train_labels[:, 0] = train_labels[:, 0] - 1
# test_labels[:, 0] = test_labels[:, 0] - 1

train_labels = train_labels[:, [1,2,3]]
test_labels = test_labels[:, [1,2,3]]
print(train_labels.shape)

# get the model
BATCH_SIZE = train_ds.shape[0]
NUM_POINTS = train_ds.shape[1]
print(train_labels.shape)

if args.model_type == "pointnet_cls":
    model = models.pointNet_cls.get_model()
elif args.model_type == "pointnet_cls_small":
    model = models.pointNet_cls_small.get_model()
elif args.model_type == "pointnet_cls_tiny":
    model = models.pointNet_cls_tiny.get_model()


print(model.name)

# create the path to the folder where to save the checkpoints
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")
model_save_dir = os.path.join(save_dir, model.name)

# if it doesn't exist, create it
if os.path.exists(model_save_dir) == False:
    os.mkdir(model_save_dir)

model.summary()

# set the callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                 patience=25, 
                                                 min_lr=1e-6)
save_checkpoints = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_save_dir, "best/"),                                                
                                                      save_best_only=args.save_best,
                                                      save_weights_only=True,
                                                      verbose=1)
callbacks = [reduce_lr, save_checkpoints]
# callbacks = [save_checkpoints]

BATCH_SIZE = 16

repeat_number = int((args.epochs * BATCH_SIZE * BATCH_SIZE) / train_ds.shape[0])
if repeat_number == 0:
    repeat_number = 1
train_dataset = tf.data.Dataset.from_tensor_slices((train_ds, train_labels)).repeat(repeat_number).batch(BATCH_SIZE).shuffle(buffer_size=BATCH_SIZE, reshuffle_each_iteration=False)
test_dataset = tf.data.Dataset.from_tensor_slices((test_ds, test_labels)).repeat(repeat_number).batch(BATCH_SIZE).shuffle(buffer_size=BATCH_SIZE, reshuffle_each_iteration=False)

# if desired, continute the previous training
if args.restore_training == True:
    saved_weights_dir = model_save_dir
    if args.restore_from_last==True and args.restore_from_best==True:
        print("\n\tOnly one between --restore_from_last and --restore_from_training can be true")
        assert(False)
    elif args.restore_from_last==True:
        model.load_weights(os.path.join(saved_weights_dir, "last/"))
    elif args.restore_from_best==True:
        model.load_weights(os.path.join(saved_weights_dir, "best/"))
    else:
        print("\n\tOne between --restore_from_last and --restore_from_training must be true")
        assert(False)

# compile the model
model.compile(
    loss = MeanSquaredError(),
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    metrics = [keras.losses.mean_squared_error]
)


if args.restore_training==True:
    batch = test_dataset.take(1)
    loss, accuracy = model.evaluate(batch, verbose=0)
    print("\n\tRestoring training from the ", "last" if args.restore_from_last==True else "best"," set of weights -> Accuracy: ", accuracy, end="\n\n")

# start training
history = model.fit(train_dataset,
          epochs=args.epochs,
          steps_per_epoch=BATCH_SIZE,
          validation_data=test_dataset,
          callbacks=callbacks)

if args.save_last==True:
    model.save_weights(os.path.join(model_save_dir, "last/"))

# save the training history
history_file = "history/training_history"
with open(os.path.join(model_save_dir, history_file), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# to load
# with open(file_path, "rb") as file_pi:
#     history = pickle.load(file_pi)

points = train_ds[:8, ...]
labels = train_labels[:8, ...]

# run test data through model
preds = model.predict(points)

label = np.expand_dims(labels[0], 1)
pred = np.expand_dims(preds[0], 1)
error = np.abs(label - pred)

print(error.shape)

print("\nResults right after training.")
to_print = np.hstack((label, pred, error))

for i in range(to_print.shape[0]):
    print("{:.4f}, {:.4f}, {:.4f}".format(to_print[i,0], to_print[i,1], to_print[i,2]))
print("\n")

points = test_ds[:8, ...]
labels = test_labels[:8, ...]

# run test data through model
preds = model.predict(points)

label = np.expand_dims(labels[0], 1)
pred = np.expand_dims(preds[0], 1)
error = np.abs(label -pred)

print("\nResults right after training.")
to_print = np.hstack((label, pred, error))

for i in range(to_print.shape[0]):
    print("{:.4f}, {:.4f}, {:.4f}".format(to_print[i,0], to_print[i,1], to_print[i,2]))
