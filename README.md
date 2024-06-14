## cVAN: A Novel Sleep Staging Method Via Cross-View Alignment Network





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


### Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```bibtex
@ARTICLE{10555125,
  author={Yang, Zhanjiang and Qiu, Meiyu and Fan, Xiaomao and Dai, Genan and Ma, Wenjun and Peng, Xiaojiang and Fu, Xianghua and Li, Ye},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={cVAN: A Novel Sleep Staging Method Via Cross-View Alignment Network}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  keywords={Sleep;Physiology;Feature extraction;Transformers;Convolutional neural networks;Brain modeling;Biomedical monitoring;Sleep stages classification;Scale-aware attention;View alignment;Residual- like network;Transformer- like network},
  doi={10.1109/JBHI.2024.3413081}}
```


  
