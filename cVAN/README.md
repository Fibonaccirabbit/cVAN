## cVAN：Cross-View Alignment Network via Scale-Aware Attention for Sleep Stage Classification





### Environment：

+ python 3.9
+ tensorflow 2.11
+ cuda 11.1



### DataSets

+ ISRUC-S1-S3，You can download this data from this website：https://sleeptight.isr.uc.pt/
+ SleepEDF-153，You can download this data from this website: https://www.physionet.org/content/sleep-edfx/1.0.0/



### Start

+ Run the rawdata_preprocess.py to pre-process the data.

+ Run the following command to start training:

  ```
  python train_cVAN.py -c config/config -g -1 // -1 means use cpu only,0 means use gpu.
  ```

+ You can change the training parameters by modifying the config.config file

+ You can run the following commands for evaluation:

  ```
  python evaluate_cVAN.py -c config/config -g -1 // -1 means use cpu only,0 means use gpu.
  ```

  