# Global-Local Attention for Emotion Recognition

<p float="left">
  <img src="/out/1.jpg" width="200" />
  <img src="/out/2.jpg" width="200" /> 
  <img src="/out/3.jpg" width="200" />
  <img src="/out/4.jpg" width="200" />
</p>


## Requirements
- Python 3
- Install [tensorflow](https://www.tensorflow.org/install) (or tensorflow-gpu) >= 2.0.0 
- Install some other packages
```Shell
pip install cython
pip install opencv-python==4.3.0.36 matplotlib numpy==1.18.5 dlib
```



# Dataset
We provide the NCAER-S dataset with original images and extracted faces (a .txt file with 4 bounding box coordinate) in the NCAERS dataset.

***The dataset can be downloaded at [Google Drive](https://bit.ly/NCAERS_dataset)***

Note that the dataset and label should have structure like the followings:
```
NCAER-S 
│
└───images
│   │
│   └───class_1
│   │   │   img1.jpg
│   │   │   img2.jpg
│   │   │   ...
│   └───class_2
│       │   img1.jpg
│       │   img2.jpg
│       │   ...
│   
└───crop
│   │
│   └───class_1
│   │   │   img1.txt
│   │   │   img2.txt
│   │   │   ...
│   └───class_2
│       │   img1.txt
│       │   img2.txt
│       │   ...
```


# Running
Our code supports these types of execution with argument -m or --mode:
```Shell
#extract faces from <train, val or test> dataset (specified in config.py)
python run.py -m extract dataset_type=train

#train the model with config specified in the config.py
python run.py -m train 

#evaluate the trained model on the dataset <dataset_type>
python run.py -m eval --dataset_type=test --trained_weights=path/to/weights
```

# Evaluation
Our trained model is available at ```weights/glamor-net/Model```.
- Firstly, please download the dataset and extract it into "data/" directory.
- Then specified the path to the test data (images and crop):
```Python
config = config.copy({
    'test_images': 'path_to_test_images',
    'test_crop':   'path_to_test_cropped_faces' #(.txt files),
})
```
- Run this command to evaluate the model. We are using the classification accuracy as our evaluation metric.
```Shell
# Evaluate our model in the test set
python run.py -m eval --dataset_type=test --trained_weights=weights/glamor-net/Model
```


# Training 
Firstly please extract the faces from train set (val set is optional)
- Specify the path to the dataset in config.py (train_images, val_images, test_images)
- Specify the desired face-extracted output path in config.py (train_crop, val_crop, test_crop)
```Python
config = config.copy({

    'train_images': 'path_to_training_images',
    'train_crop':   'path_to_training_cropped_faces' #(.txt files),

    'val_images': 'path_to_validation_images',
    'val_crop':   'path_to_validation_cropped_faces' #(.txt files)

})
```
- Perform face extraction on both dataset_type by running the commands:
```Shell
python run.py -m extract --dataset_type=<train, val or test>
```
Start training:
```Shell
# Train a new model from sratch
python run.py -m train 

# Continue training a model that you had trained earlier
python run.py -m train --resume=path/to/trained_weights

# Resume the last checkpoint model
python run.py -m train --resume=last
```

# Prediction
We support prediction on single image or on images in a directory by running this command:

```Shell
# Predict on single image
python predict.py --trained_weights=weights/glamor-net/Model --input=test_images/1.jpg --output=path/to/out/directory

# Predict on images in directory
python predict.py --trained_weights=weights/glamor-net/Model --input=test_images/ --output=out/

```

# Use the help option to see a description of all available command line arguments











