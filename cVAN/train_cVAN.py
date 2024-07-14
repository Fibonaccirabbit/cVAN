import argparse
import gc
import os
import shutil
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras

from CvACN_NET import build_cVAN
from utils.Utils import PrintScore, VariationCurve, process_batch_signal, ReadConfig
from utils.DataGenerator import kFoldGenerator
from utils.Utils import ConfusionMatrix

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def cVAN_Loss(y_true, y_pred):
    print(y_pred)
    output1 = tf.keras.layers.Lambda(lambda x: x[:, :5])(y_pred)
    # output2 =  tf.keras.layers.Lambda(lambda x:x[:,5:133])(y_pred)
    # output3 =  tf.keras.layers.Lambda(lambda x:x[:,133:])(y_pred)
    print(output1)
    cross_loss = tf.keras.losses.categorical_crossentropy(y_true, output1)
    # cos_loss =  tf.keras.losses.cosine_similarity(output2,output3)
    return cross_loss


def cos_Loss(y_true, y_pred):
    output1 = tf.keras.layers.Lambda(lambda x: x[:, :128])(y_pred)
    output2 = tf.keras.layers.Lambda(lambda x: x[:, 128:])(y_pred)
    # cos_losss =  tf.keras.losses.cosine_similarity(output1,output2)
    output1 = tf.nn.l2_normalize(output1, axis=1)
    output2 = tf.nn.l2_normalize(output2, axis=1)
    margin = 1
    square_pred = tf.square(output1)
    margin_square = tf.square(tf.maximum(margin - output1, 0))
    loss = tf.reduce_mean(output2 * square_pred + (1 - output2) * margin_square)
    return loss


sys.path.append(r"D:\SleepNet\Transformer\model")
# command line parameters -c -g
parser = argparse.ArgumentParser()
parser.add_argument("-c", type=str, help="configuration file", required=True)
parser.add_argument("-g", type=str, help="GPU number to use, set '-1' to use CPU", required=True)
args = parser.parse_args()
Path, cfgFeature, _, _ = ReadConfig(args.c)

# set GPU number or use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = args.g

channels = int(cfgFeature["channels"])
fold = int(cfgFeature["fold"])
num_epochs_f = int(cfgFeature["epoch_f"])
batch_size_f = int(cfgFeature["batch_size_f"])
optimizer_f = cfgFeature["optimizer_f"]
learn_rate_f = float(cfgFeature["learn_rate_f"])

# ## 1.3. Parameter check and enable

# Create save pathand copy .config to it
if not os.path.exists(Path['Save']):
    os.makedirs(Path['Save'])
shutil.copyfile(args.c, Path['Save'] + "last.config")

# # 2. Read data and process data

# ## 2.1. Read data

# #  Test_data
ReadList = np.load(Path['Edf_young_data'], allow_pickle=True)
Test_Fold_Num = ReadList['Fold_len']  # Num of samples of each fold
Test_Fold_Data = ReadList['Fold_data']  # data of each fold
Test_Fold_Label = ReadList['Fold_label']  # Labels of each fold

print("Read data successfully")
print('Number of test samples: ', np.sum(Test_Fold_Num))

print("Read data successfully")
print('Number of train samples: ', np.sum(Test_Fold_Num))


def stander(data):
    standard_scaler = MinMaxScaler()
    data_2d = data.reshape(-1, data.shape[-1])
    normalized_data_standard = standard_scaler.fit_transform(data_2d)
    normalized_data_standard = normalized_data_standard.reshape(data.shape)
    return normalized_data_standard


DataGenerator = kFoldGenerator(Test_Fold_Data, Test_Fold_Label)
all_scores = []
AllPred = []
AllTrue = []
for i in range(1):
    print(128 * '_')
    print('Fold #', i)
    checkpoint_filepath = "model_cVAN_Edf_young_" + str(i) + ".hdf5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_logit_output_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    x_train, y_train, x_test, y_test = DataGenerator.getFold(2)
    x_train_1 = np.array(process_batch_signal(x_train))
    x_test_1 = np.array(process_batch_signal(x_test))
    x_train = np.apply_along_axis(lambda x: np.interp(np.linspace(0, len(x), 500), np.arange(len(x)), x), axis=2,
                                  arr=x_train)
    x_train = np.abs(np.array(x_train))
    x_test = np.apply_along_axis(lambda x: np.interp(np.linspace(0, len(x), 500), np.arange(len(x)), x), axis=2,
                                 arr=x_test)
    x_test = np.abs(np.array(x_test))

    vocab_size = 1000
    maxlen = 500
    num_class = 5
    d_model = 64
    num_heads = 2
    num_layers = 3
    ff_dim = 64
    model = build_cVAN(
        vocab_size=vocab_size,
        maxlen=maxlen,
        num_class=num_class,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
    )

    # Compile and train the model

    optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.001)
    model.compile(loss={"logit_output": "categorical_crossentropy", "sim_output": cos_Loss},
                  loss_weights={"logit_output": 1, "sim_output": 0.5},
                  metrics={'logit_output': "accuracy"}, optimizer=optimizer)

    history_fea = model.fit([x_train, x_train_1], [y_train, y_train], batch_size=20, epochs=30,
                            validation_data=([x_test, x_test_1], [y_test, y_test]),
                            callbacks=[checkpoint_callback], )  # assuming training data is already defined
    model.load_weights(checkpoint_filepath)
    accuracy = model.evaluate([x_test, x_test_1], [y_test, y_test])
    print(accuracy)
    all_scores.append(accuracy[3])
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # Save training information
    if i == 0:
        fit_loss = np.array(history_fea.history['logit_output_loss']) * Test_Fold_Num[i]
        fit_acc = np.array(history_fea.history['logit_output_accuracy']) * Test_Fold_Num[i]
        fit_val_loss = np.array(history_fea.history['val_logit_output_loss']) * Test_Fold_Num[i]
        fit_val_acc = np.array(history_fea.history['val_logit_output_accuracy']) * Test_Fold_Num[i]
    else:
        fit_loss = fit_loss + np.array(history_fea.history['logit_output_loss']) * Test_Fold_Num[i]
        fit_acc = fit_acc + np.array(history_fea.history['logit_output_accuracy']) * Test_Fold_Num[i]
        fit_val_loss = fit_val_loss + np.array(history_fea.history['val_logit_output_loss']) * Test_Fold_Num[i]
        fit_val_acc = fit_val_acc + np.array(history_fea.history['val_logit_output_accuracy']) * Test_Fold_Num[i]

    saveFile = open(Path['save'] + "Result_cVAN.txt", 'a+')
    print('Fold #' + str(i), file=saveFile)
    print(history_fea.history, file=saveFile)
    saveFile.close()
    predicts = model.predict([x_test, x_test_1])[0]
    print(predicts.shape)
    AllPred_temp = np.argmax(predicts, axis=1)
    AllTrue_temp = np.argmax(y_test, axis=1)
    if i == 0:
        AllPred = AllPred_temp
        AllTrue = AllTrue_temp
    else:
        AllPred = np.concatenate((AllPred, AllPred_temp))
        AllTrue = np.concatenate((AllTrue, AllTrue_temp))

    # Fold finish
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    del model, x_train, y_train, x_test, y_test
    gc.collect()

print(128 * '=')
print("All folds' acc: ", all_scores)
print("Average acc of each fold: ", np.mean(all_scores))
print(AllTrue.shape)
# Print score to console
print(128 * '=')
PrintScore(AllTrue, AllPred)
# Print score to Result.txt file
PrintScore(AllTrue, AllPred, savePath=Path['save'])
# Print confusion matrix and save
ConfusionMatrix(AllTrue, AllPred, classes=['W', 'N1', 'N2', 'N3', 'REM'], savePath=Path['save'])

# Draw ACC / loss curve and save
VariationCurve(fit_acc, fit_val_acc, 'Acc', Path['Save'], figsize=(9, 6))
VariationCurve(fit_loss, fit_val_loss, 'Loss', Path['Save'], figsize=(9, 6))

saveFile = open(Path['Save'] + "Result_RT.txt", 'a+')
print(history_fea.history, file=saveFile)
saveFile.close()
print('End of evaluating cVAN.')
print(128 * '#')
