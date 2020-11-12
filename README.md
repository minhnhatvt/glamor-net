# Global-Local Attention for Emotion Recognition



# Getting Started
If you want to validate our result or training by yourself. We've already setup all the process in the google colab link below.\
Some of the examples in PASCAL SBD dataset. 

![image](dataset_examples/ex1.png)

## Requirements
- Python 3
- Install tensorflow (or tensorflow-gpu) >= 2.0.0 [tensorflow](https://www.tensorflow.org/install).
- Install some other packages
       ```Shell
       pip install cython
       pip install opencv-python==4.3.0.36 matplotlib numpy==1.18.5 Keras-Preprocessing==1.1.2 dlib
       ```




4. Download our prepared dataset from google drive [PASCAL_SBD.zip](https://drive.google.com/file/d/1uyZtl6LDxbgHC7ctDl0rbGlxOOrvCssG/view?usp=sharing).
Extract and put it into data/sbd folder. (the folder should have sbd/imgs and <anotation_files>.json)
5. Run the desired command above for training or evaluating.
Notice that the orignal [PASCAL SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) using the voc's annotation format.\
To run this code, we've converted it into coco's format. we put the prepared dataset download link in the tutorial below

# Dataset statistics
We also provide the code for examine the dataset (how many classess? or how many object and annotation in each class?).\
You can see the statistics by running
```
python dataset_stats.py --dataset="path to dataset" --subset="subset name (train or val)"
```
# Training on PASCAL SBD

Firstly, please download Resnet50 pretrained imagenet [here](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view)
and put it into `./weights` folder
```

# Train a new model starting from pretrained ImageNet with specified parameters
python train.py --config=yolact_resnet50_pascal_config --batch_size=8 --validation_epoch=5 

# Continue training a model that you had trained earlier
python train.py --config=yolact_resnet50_pascal_config --batch_size=8 --validation_epoch=5 --resume="path to weights.pth"

# Use the help option to see a description of all available command line arguments
python train.py --help

```

The training schedule, learning rate, and other hyperparameters should be set in `data/config.py`

## Evaluating
You can also run the validation on the sbd_val subset by command:
```
python eval.py --trained_model="weights/yolact_resnet50_pascal_112_120000.pth" --config=yolact_resnet50_pascal_config
```

For inference on images
```
python eval.py --trained_model="weights/yolact_resnet50_pascal_112_120000.pth" --score_threshold=0.6 --top_k=15 --images=path/to/input/folder:path/to/output/folder
```

## Pretrained Weights
Our model achieved the result show in the table after 120k iterations on `sbd val set`
type |  all  |  .50  |  .55  |  .60  |  .65  |  .70  |  .75  |  .80  |  .85  |  .90  |  .95  
--- | --- | --- | --- |--- | --- | --- | --- | --- | --- | --- | ---  
box | 46.77 | 75.00 | 72.46 | 68.85 | 63.77 | 57.47 | 50.02 | 39.87 | 26.89 | 11.84 |  1.55 
mask| 45.32 | 71.73 | 68.04 | 64.16 | 60.08 | 54.37 | 47.35 | 39.19 | 28.78 | 16.18 |  3.27 


Download our trained weights here: https://drive.google.com/file/d/1GvYj7T5rv4grRHeOwgm30cPLaorwFAGB/view













