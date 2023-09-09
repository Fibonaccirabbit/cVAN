import configparser
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.sparse.linalg import eigs

from scipy import signal

import tensorflow as tf

##########################################################################################
# Read configuration file ################################################################
def ReadConfig(configfile):
    config = configparser.ConfigParser()
    print('Config: ', configfile)
    config.read(configfile)
    cfgPath = config['path']
    cfgFeat = config['feature']
    cfgTrain = config['train']
    cfgModel = config['model']
    return cfgPath, cfgFeat, cfgTrain, cfgModel

#########################################################
# TF Image Construction
def get_spectrogram(waveform, n_length):
    _, _, spectrogram = signal.stft(waveform, fs=1.0, nperseg=n_length,
                                    window='hann', nfft=None, noverlap=None, return_onesided=False)
    spectrogram = np.abs(spectrogram)
    # Obtain the magnitude of the STFT.
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).

    spectrogram = spectrogram[..., np.newaxis]
    # print(np.array(spectrogram).shape)
    return spectrogram


def process_batch_signal(signals, n_length=100):
    signal_spectrogram_list = []
    for record in signals:
        for i in range(len(record)):
            if (i == 0):
                spectrogram = get_spectrogram(record[i], n_length)
            else:
                spectrogram = np.concatenate((spectrogram, get_spectrogram(record[i], n_length)), axis=-1)

        img = np.abs(spectrogram)

        img = np.pad(img, ((0, 0), (19, 20), (0, 0)), 'constant', constant_values=(0))
        # print(np.array(img).shape)
        signal_spectrogram_list.append(img)

    return signal_spectrogram_list

#############################################################
# GRU Att



##########################################################################################
# Print score between Ytrue and Ypred ####################################################

def PrintScore(true, pred, savePath=None, average='macro'):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    print('Acc\tF1S\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2], F1[3], F1[4]),
          file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred,
                                        target_names=['Wake', 'N1', 'N2', 'N3', 'REM'],
                                        digits=4), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true, pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t', metrics.accuracy_score(true, pred), file=saveFile)
    print(' Cohen Kappa\t', metrics.cohen_kappa_score(true, pred), file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    if savePath != None:
        saveFile.close()
    return


##########################################################################################
# Print confusion matrix and save ########################################################

def ConfusionMatrix(y_true, y_pred, classes, savePath, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Confusion matrix'
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n = cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           # title=title,
           # ylabel='True Sleep Stage',
           # xlabel='Predicted Sleep Stage',
           )
    ax.set_xlabel(xlabel='Predicted Sleep Stage', fontsize=12)
    ax.set_ylabel(ylabel='True Sleep Stage', fontsize=12)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    # plt.rc('font', family='Times New Roman')
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j] * 100, '.2f') + '%\n' + format(cm_n[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(savePath + title + ".png", dpi=600)
    # plt.show()
    return ax


##########################################################################################
# Draw ACC / loss curve and save #########################################################

def VariationCurve(fit, val, yLabel, savePath, figsize=(9, 6)):
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(fit) + 1), fit, label='Train')
    plt.plot(range(1, len(val) + 1), val, label='Val')
    plt.title('Model ' + yLabel)
    plt.xlabel('Epochs')
    plt.ylabel(yLabel)
    plt.legend()
    plt.savefig(savePath + 'Model_' + yLabel + '.png')
    # plt.show()
    return


