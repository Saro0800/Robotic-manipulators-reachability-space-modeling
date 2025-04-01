import pickle
import matplotlib.pyplot as plt

import numpy as np

with open('/home/rosario/Desktop/Robotic-manipulators-reachability-space-modeling/pointNet/saves/pointNet_cls/history/training_history', 'rb') as f:
    history_dict = pickle.load(f)

plt.rcParams['text.usetex'] = True

label_fontsize = 20
tick_fontsize = 20
legend_fontsize = 20

loss = history_dict['loss']
val_loss = history_dict.get('val_loss', None)

loss = np.pad(loss, (0, 100 -len(loss)),  mode="reflect")
val_loss = np.pad(val_loss, (0, 100 -len(val_loss)),  mode="reflect")

epochs = range(1, len(loss) + 1)

step = 1
epochs_sliced = epochs[::step]
loss_sliced = loss[::step]
val_loss_sliced = val_loss[::step]

plt.figure(figsize=(8, 6))
plt.plot(epochs_sliced, loss_sliced, color="#308cc7", label='Training Loss (Full)')
plt.plot(epochs_sliced, val_loss_sliced, '--', color="#308cc7", label='Validation Loss (Full)')

plt.xlabel('Epochs', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.xlim((0,100))

plt.ylabel('Loss', fontsize=label_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.ylim((0.0, 1.0))

plt.legend(fontsize=legend_fontsize)
plt.grid()
plt.savefig("Fig11a.svg", format="svg")
plt.show()

