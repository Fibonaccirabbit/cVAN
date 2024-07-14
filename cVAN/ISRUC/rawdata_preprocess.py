import random
from collections import Counter
import numpy as np
from scipy import signal
import scipy.io as scio
from os import path

# Set to your dataset path
path_Extracted = './data/ISRUC_S3/ExtractedChannels/'
path_RawData = './data/ISRUC_S3/RawData/'
path_output = './data/ISRUC_S3/'

channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
            'LOC_A2', 'ROC_A1', 'X1', 'X2']


# read data
def read_psg(path_Extracted, sub_id, channels, resample=3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))
    print("--------------------------------")
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use


# read label
def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    return np.array(label[:-ignore])


fold_label = []
fold_psg = []
fold_len = []
fold_label_count = [0, 0, 0, 0, 0]
for sub in range(1, 11):
    print('Read subject', sub)
    label = read_label(path_RawData, sub)
    # print(label)
    psg = read_psg(path_Extracted, sub, channels)
    # print(psg)
    print('Subject', sub, ':', label.shape, psg.shape)
    # 确保数据量一致 对应
    assert len(label) == len(psg)
    # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM

    np.random.seed(42)
    index = [i for i in range(psg.shape[0])]
    random.shuffle(index)
    psg = psg[index]
    label = label[index]

    label[label == 5] = 4  # make 4 correspond to REM

    fold_label_count[0] += Counter(label)[0]
    fold_label_count[1] += Counter(label)[1]
    fold_label_count[2] += Counter(label)[2]
    fold_label_count[3] += Counter(label)[3]
    fold_label_count[4] += Counter(label)[4]
    # 将 0-4类转换为one-hot
    fold_label.append(np.eye(5)[label])
    # fold_label.append(label)

    fold_psg.append(np.float32(psg))
    fold_len.append(len(label))
print('Preprocess over.')
print(len(fold_psg[0]))
print(len(fold_label[0]))


np.savez(path.join(path_output, 'Train.npz'),
         Fold_data=fold_psg,
         Fold_label=fold_label,
         Fold_len=fold_len,
         Fold_label_conunt=fold_label_count
         )
print('Saved to', path.join(path_output, 'Train.npz'))
