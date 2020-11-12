import tensorflow as tf
import numpy as np
from config import config
import os
import pathlib
import dlib
import cv2
from tensorflow.keras.utils import Progbar


def extract_faces(kind):
    """
    :param kind: "train", "val" or "test
    Extract cropped faces using dlib face detector
    """
    if kind == 'train':
        image_path = config.train_images
        crop_path = config.train_crop
    elif kind == 'val':
        image_path = config.val_images
        crop_path = config.val_crop
    elif kind == 'test':
        image_path = config.test_images
        crop_path = config.test_crop
    else:
        raise ValueError('Wrong type of dataset ("train", "val" or "test" is acceptable)')
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("detector/mmod_human_face_detector.dat")
    imgs = []

    valid_images = [".jpg", ".gif", ".png", ".tga"]

    print(f"Extracting faces in {kind} set...")

    imgs_cnt = len([1 for category in os.listdir(image_path) for f in os.listdir(os.path.join(image_path, category))])
    pb = Progbar(target=imgs_cnt, width=30)
    for category in os.listdir(image_path):
        if not os.path.exists(os.path.join(crop_path,category)):
            os.makedirs(os.path.join(crop_path,category))
        for f in os.listdir(os.path.join(image_path, category)):
            fname, ext = os.path.splitext(f)
            # print(os.path.join(path,category,f))
            if ext.lower() not in valid_images:
                continue
            img = cv2.imread(os.path.join(image_path, category, f))
            h, w, _ = img.shape
            # if (h>400):
            #     print(img.shape)
            #     print(os.path.join(path,category,f))
            # continue
            result = dnnFaceDetector(img, 1)
            max_confidence = -100
            final_rect = None
            for rect in result:
                if rect.confidence > max_confidence:
                    max_confidence = rect.confidence
                    final_rect = rect.rect

            # print(max_confidence)
            if (max_confidence == -100):
                fo = open(os.path.join(crop_path, category, fname + '.txt'), "w")
                fo.write(",".join([str(0), str(0), str(0), str(0)]))
                fo.close()
                continue
            x1 = final_rect.left()
            y1 = final_rect.top()
            x2 = final_rect.right()
            y2 = final_rect.bottom()
            fo = open(os.path.join(crop_path, category, fname + '.txt'), "w")
            fo.write(",".join([str(x1), str(y1), str(x2), str(y2)]))
            fo.close()
            pb.add(1)
            imgs_cnt += 1
    print(f"Sucessfully cropped {imgs_cnt} images!")

def mask_face(img, bbox):  # create a mask with 0 in region of the face bbox in the whole image and the other pixels are 1
    h, w, _ = img.shape
    x1, y1, x2, y2 = bbox
    x1 = np.clip(x1, 0, w)
    y1 = np.clip(y1, 0, h)
    x2 = np.clip(x2, 0, w)
    y2 = np.clip(y2, 0, h)
    mask = np.ones((h, w, 3))
    mask[y1:y2, x1:x2] = 0
    return mask.astype(np.uint8)


def parse_image(crop_path):  # read image from filename and load it to a tensor

    def proc(filename):
        parts = tf.strings.split(filename, os.sep)
        label_text = parts[-2]
        name = tf.strings.split(parts[-1], '.')[0]

        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image, channels=3)
        # image = tf.image.convert_image_dtype(image, tf.float32) #convert to range [0,1]

        # read the pre-detected bounding box (4 number x1,y1,x2,y2) of the corresponding image
        bbox = tf.io.read_file(crop_path + "/" + parts[-2] + "/" + name + ".txt")  # read a tf strings with 1 line
        bbox = tf.strings.to_number(tf.strings.split(bbox, ","), out_type=tf.dtypes.int32)  # shape (4,)

        return image, bbox, label_text

    return proc


def image_augment(img, flip=True, rotation=True, shift=True, zoom=True, shear=True):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    if flip:
        img = tf.image.random_flip_left_right(img)
    if rotation:
        img = tf.keras.preprocessing.image.random_rotation(img.numpy(), 20, row_axis=0, col_axis=1, channel_axis=2)
    # img = tf.image.random_brightness(img, 0.05)
    #     if shift:
    #         img = tf.keras.preprocessing.image.random_shift(img,0.2,0.2, 0, 1, 2, )
    # #     if zoom:
    # #         img = tf.keras.preprocessing.image.random_zoom(img, (0.8,0.8), 0,1,2)
    #     if shear:
    #         img = tf.keras.preprocessing.image.random_shear(img,0.2, 0,1,2)

    return img


def training_preprocess(table_label):
    def proc(image, bbox, label):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        x1 = tf.clip_by_value(x1, 0, w)
        y1 = tf.clip_by_value(y1, 0, h)
        x2 = tf.clip_by_value(x2, 0, w)
        y2 = tf.clip_by_value(y2, 0, h)

        face = tf.zeros([96, 96, 3])
        if (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):  # picture contain no face
            face = tf.zeros([96, 96, 3], dtype=tf.uint8)
        else:
            face = image[y1:y2, x1:x2]  # crop the region of the face
        face = tf.image.resize(face, config.face_input_size)

        mask = tf.numpy_function(mask_face, [image, bbox],
                                 tf.uint8)  # execute a numpy function with the inputs are tensors, because we cannot use eager tensor.numpy() in a dataset mapping
        # maybe it converts input from tensor -> numpy and converts back the output from numpy to tensor
        image = tf.multiply(image, mask)
        image = tf.image.encode_jpeg(
            image)  # convert the tensor to image to apply tf.image.resize, i dont know why must do this thing, otherwise it keep getting error
        image = tf.image.decode_jpeg(image)

        image = tf.image.convert_image_dtype(image, tf.float32)  # normalize to range [0,1]
        face = tf.image.convert_image_dtype(face, tf.float32) / 255  # normalize to range [0,1]

        image = tf.image.resize(image, [128, 171])
        image = tf.image.random_crop(image, size=list(config.context_input_size) + [3])
        label = table_label.lookup(label)

        #     image = tf.numpy_function(image_augment, inp=[image], Tout = [tf.float32])
        #     image = tf.squeeze(image)
        face = tf.numpy_function(image_augment, inp=[face], Tout=[tf.float32])
        face = tf.squeeze(face)
        return image, face, label

    return proc


def get_train_dataset():
    table_label = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(config.class_names),
            values=tf.constant([0, 1, 2, 3, 4, 5, 6])
        ),
        default_value=tf.constant(-1),
    )
    ds_root = pathlib.Path(config.train_images)
    list_ds = tf.data.Dataset.list_files(str(ds_root / '*/*'))
    labeled_ds = list_ds.map(
        parse_image(config.train_crop))  # map each image  filenames to 3 tensors image, bbox and label_text
    dataset = labeled_ds.map(training_preprocess(table_label),
                             num_parallel_calls=config.num_parallel_calls)  # map image,bbox, label to image, face, label
    dataset = dataset.batch(config.batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)  # prefetch a batch to reduced the hardware memory bottleneck
    return dataset


def parse_image_test(crop_path):  # read image from filename and load it to a tensor

    def proc(filename):
        parts = tf.strings.split(filename, os.sep)
        label_text = parts[-2]
        name = tf.strings.split(parts[-1], '.')[0]

        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image, channels=3)

        bbox = tf.io.read_file(crop_path + "/" + parts[-2] + "/" + name + ".txt")  # read a tf strings with 1 line
        bbox = tf.strings.to_number(tf.strings.split(bbox, ","), out_type=tf.dtypes.int32)  # shape (4,)
        return image, bbox, label_text

    return proc


def testing_preprocess(table_label_test):
    def proc(image, bbox, label):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        x1 = tf.clip_by_value(x1, 0, w)
        y1 = tf.clip_by_value(y1, 0, h)
        x2 = tf.clip_by_value(x2, 0, w)
        y2 = tf.clip_by_value(y2, 0, h)

        face = tf.zeros([96, 96, 3])
        if (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):  # picture contain no face
            face = tf.zeros([96, 96, 3], dtype=tf.uint8)
        else:
            face = image[y1:y2, x1:x2]  # crop the region of the face
        face = tf.image.resize(face, [96, 96])

        mask = tf.numpy_function(mask_face, [image, bbox],
                                 tf.uint8)  # execute a numpy function with the inputs are tensors, because we cannot use eager tensor.numpy() in a dataset mapping
        # maybe it converts input from tensor -> numpy and converts back the output from numpy to tensor
        image = tf.multiply(image, mask)
        image = tf.image.encode_jpeg(
            image)  # convert the tensor to image to apply tf.image.resize, i dont know why must do this thing, otherwise it keep getting error
        image = tf.image.decode_jpeg(image)

        image = tf.image.convert_image_dtype(image, tf.float32)  # normalize to range [0,1]
        face = tf.image.convert_image_dtype(face, tf.float32) / 255  # normalize to range [0,1]

        image = tf.image.resize(image, [128, 171])
        image = image[8:120, 29:141:]
        label = table_label_test.lookup(label)
        return image, face, label

    return proc

def get_eval_dataset(kind='test'):
    table_label_test = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(config.class_names),
            values=tf.constant([0, 1, 2, 3, 4, 5, 6])
        ),
        default_value=tf.constant(-1),
    )

    if kind == 'train':
        image_path = config.train_images
        crop_path = config.train_crop
    elif kind == 'test':
        image_path = config.test_images
        crop_path = config.test_crop
    elif kind == 'val':
        image_path = config.val_images
        crop_path = config.val_crop
    else:
        raise ValueError('Wrong type of dataset ("train". "val" or "test" is acceptable)')

    ds_root = pathlib.Path(image_path)
    list_ds_test = tf.data.Dataset.list_files(str(ds_root / "*/*"))
    dataset_test = list_ds_test.map(parse_image_test(crop_path))
    dataset_test = dataset_test.map(testing_preprocess(table_label_test), num_parallel_calls=config.num_parallel_calls)
    dataset_test = dataset_test.batch(config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset_test



if __name__ == '__main__':
    #test_dataset = get_test_dataset('val')
    # test_dataset = get_train_dataset()
    # print(tf.data.experimental.cardinality(test_dataset))
    # res = 0
    # for a,b,y in test_dataset:
    #     res += a.shape[0]
    # print(res)
    extract_faces("val")