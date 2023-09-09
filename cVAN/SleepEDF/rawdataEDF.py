import random
from os import path

# Set to your dataset path
path_Extracted = './data/ExtractedChannels/'
path_RawData = './data/TrainRawData/'
path_output = './data/'
import matplotlib.pyplot as plt
import numpy as np

ReadList = np.load('ST_all.npz', allow_pickle=True)
Fold_Data = ReadList['X'] # data of each fold
Fold_Label = ReadList['sleep_stages']  # Labels of each fold


fold_label = []
fold_psg = []
fold_len = []
print(np.array(Fold_Data).shape)

for i in range(150):
    print("sub:"+str(i))
    data =  np.transpose(Fold_Data[i], (0, 2, 1))
    data[:,:,-3000:] *= 1e5
    label =Fold_Label[i]
    index = [i for i in range(data.shape[0])]
    random.shuffle(index)
    data = data[index]
    label = label[index]

    plt.plot(data[1][1])
    plt.show()
    plt.plot(data[2][1])
    plt.show()
    plt.plot(data[3][1])
    plt.show()
    plt.plot(data[4][1])
    plt.show()

    fold_label.append(np.eye(5)[label])
    fold_psg.append(np.float32(data))
    fold_len.append(len(label))

print(np.array(Fold_Data).shape)
print(np.array(fold_label).shape)
print(fold_len)

np.random.seed(42)
data_list = fold_psg
label_list = fold_label
data_array = np.vstack(data_list)
label_array = np.vstack(label_list)
idx = np.random.permutation(len(data_array))
shuffled_data = data_array[idx]
shuffled_labels = label_array[idx]
print(len(shuffled_data))
num_splits = 10
data_splits = np.array_split(shuffled_data, num_splits)
labels_splits = np.array_split(shuffled_labels,num_splits)
np.savez(path.join('Edf.npz'),
         Fold_data=data_splits,
         Fold_label=labels_splits,
         Fold_len=fold_len,
         )
print('Saved to', path.join('Edf.npz'))

