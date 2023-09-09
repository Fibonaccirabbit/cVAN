

import numpy as np

from mne.io import read_raw_edf, concatenate_raws

import glob
import dhedfreader
import os



# In[2]:


ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30

picks = [i for i in range(6)]

num_sample = 25
num_sub = 10
srate = 100


# In[3]:


def get_files_name(file_type):
    psg_fnames = glob.glob(os.path.join("./", "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join("../sleep-cassette", "*Hypnogram.edf"))
    print( psg_fnames)
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames).reshape(-1 ,1)
    ann_fnames = np.asarray(ann_fnames).reshape(-1, 1)
    assert len(psg_fnames) == len(ann_fnames)
    
    return np.concatenate((psg_fnames, ann_fnames), axis=1)


# In[4]:


def read_edf(fnames):
    raw = read_raw_edf(fnames[0], preload=False, stim_channel=None)
    srate = int(raw.info['sfreq'])
    step = EPOCH_SEC_SIZE*srate
    
    data = raw.get_data().T[:,:2]
    
    data = data[:data.shape[0]//step*step].reshape(-1, step, 2)
    anno = get_annotations(fnames[1])
    last_event = int((anno[-1][0] + anno[-1][1])/ 30)
    data = data[:last_event]
    labels = get_labels(anno, srate, data.shape[0])
    data, labels = remove_bad_labels(data, labels)
    start_idx, end_idx = pick_idx(labels)
    
    if start_idx < 60:
        return (data[:end_idx + 60], labels[:end_idx + 60])      
    return (data[start_idx - 60:end_idx + 60], labels[start_idx - 60:end_idx + 60])


# In[5]:


def get_annotations(fnames):
    with open(fnames, 'r', encoding='utf8') as f:
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, anno = zip(*reader_ann.records())
    return anno[0]


# In[6]:


def get_labels(anno, srate, size):
    step = srate*EPOCH_SEC_SIZE
    labels = np.full((int(size)), -1)
    ann = np.array([[a[0], a[1], ann2label[a[2]]] for a in anno], dtype=int)
    for a in ann:
        labels[int(a[0]/30):int((a[0]+a[1])/30)] = a[2]
    return labels


# In[7]:


def remove_bad_labels(data, labels):
    bad_idx = [i for i, x in enumerate(labels) if x == 5 or x == -1]
    x = np.delete(data, bad_idx, axis=0)
    y = np.delete(labels, bad_idx, axis=0)
    return x, y


# In[8]:


def pick_idx(labels):
    start_idx = next(i for i, x in enumerate(labels) if x == 1)
    end_idx = next(i for i, x in reversed(list(enumerate(labels))) if x != 0) + 1

    return start_idx, end_idx


# In[9]:


def pick_sub(data, labels, num_sub):
    size = data.shape[0]//num_sub*num_sub
    data = data[:size].reshape(num_sub, -1, EPOCH_SEC_SIZE*srate, data.shape[2])
    labels = labels[:size].reshape(num_sub, -1)
    
    return data, labels


# In[10]:


def pick_sample(data, labels, num_sample):
    x = [d[:num_sample] for d in data]
    y = [l[:num_sample] for l in labels]
    
    return np.array(x).reshape(-1, EPOCH_SEC_SIZE*srate, data.shape[3]), np.asarray(y, dtype=int).reshape(-1)


# In[11]:


def concate_np(rawx, rawy):
    x = rawx[0]
    for r in rawx[1:]:
        x = np.concatenate((x, r), axis=0)
    
    y = rawy[0]
    for r in rawy[1:]:
        y = np.concatenate((y, r), axis=0)
        
    return np.float32(x), y


# In[12]:


# files_name_sc = get_files_name("SC")
files_name_st = get_files_name("ST")


# In[13]:


# data_sc, labels_sc = zip(*[read_edf(f) for f in files_name_sc])
data_st, labels_st = zip(*[read_edf(f) for f in files_name_st])


# In[14]:


# x_sc, y_sc = concate_np(data_sc, labels_sc)
x_st, y_st = concate_np(data_st, labels_st)

# print(x_sc.shape, y_sc.shape)
print(x_st.shape, y_st.shape)


# In[ ]:


# pid_st = np.hstack([np.full(len(data_st[i]), f) for i, f in enumerate(files_name_st[:,0])])
# pid_st = pid_st.reshape(-1)

# pid_sc = np.hstack([np.full(len(data_sc[i]), f) for i, f in enumerate(files_name_sc[:,0])])
# pid_sc = pid_sc.reshape(-1)


# In[ ]:


#data_sub_sc, labels_sub_sc = zip(*[pick_sub(x, labels_sc[i], num_sub) for i, x in enumerate(data_sc)])
#data_sub_st, labels_sub_st = zip(*[pick_sub(x, labels_st[i], num_sub) for i, x in enumerate(data_st)])


# In[ ]:


#x_sample_sc, y_sample_sc = zip(*[pick_sample(x, labels_sub_sc[i], num_sample) for i, x in enumerate(data_sub_sc)])
#x_sample_st, y_sample_st = zip(*[pick_sample(x, labels_sub_st[i], num_sample) for i, x in enumerate(data_sub_st)])


# In[ ]:


"""x_st = np.asarray(x_sample_st)
y_st = np.asarray(y_sample_st, dtype=int).reshape(-1)
x_st = x_st.reshape(-1, x_st.shape[2], x_st.shape[3])

pid_st = np.array([np.full((num_sample*num_sub), f) for f in files_name_st[:,0]])
pid_st = pid_st.reshape(-1)

print(x_st.shape, y_st.shape, pid_st.shape)"""


# In[ ]:


# np.savez("ST_all.npz", X=x_st, sleep_stages=y_st, pid=pid_st)
np.savez("SC_all.npz", X=x_st, sleep_stages=y_st)


# In[ ]:




