import tensorflow as tf
import os
import cv2
from config import config
import dlib
import argparse
from data_utils import *
from model import *

def parse_arg(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_weights", type=str,
                        default="weights/glamor-net/Model",
                        help="/path/to/model_weights in tf2 saved format")
    parser.add_argument("--input", type=str,
                        help="path to input image or folder of images", required=True)
    parser.add_argument("--output", type=str,
                        help="path to output folder")


    global args
    args = parser.parse_args(argv)

def predict_preprocess(image, bbox):
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
    image = tf.image.encode_jpeg(image)  # convert the tensor to image to apply tf.image.resize, i dont know why must do this thing, otherwise it keep getting error
    image = tf.image.decode_jpeg(image)

    image = tf.image.convert_image_dtype(image, tf.float32)  # normalize to range [0,1]
    face = tf.image.convert_image_dtype(face, tf.float32) / 255  # normalize to range [0,1]

    image = tf.image.resize(image, [128, 171])
    image = image[8:120, 29:141:]

    return image, face

def predict_image(in_path, out_path = None, conf_threshold=0.6):
    print("Predicting on single image")
    print("Load weight from: " + args.trained_weights)
    model = get_model()
    model.load_weights(args.trained_weights)

    f = os.path.split(in_path)[-1]
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("detector/mmod_human_face_detector.dat")

    img = cv2.imread(in_path)
    out_img = img.copy()
    h, w, _ = img.shape
    extract_results = dnnFaceDetector(img, 1)

    for rect in extract_results:
        x1 = rect.rect.left()
        y1 = rect.rect.top()
        x2 = rect.rect.right()
        y2 = rect.rect.bottom()
        if rect.confidence < conf_threshold:
            continue
        image, face = predict_preprocess(img[:, :, ::-1], [x1, y1, x2, y2])

        scores = model.call(tf.expand_dims(face,0),tf.expand_dims(image,0), training=False)
        y_pred = int(tf.argmax(scores, axis=1))
        class_conf = float(scores[0, y_pred])
        class_pred = config.class_names[y_pred]
        out_img = cv2.rectangle(out_img, (x1, y1), (x2, y2), [0, 0, 255], 2)

        out_img = cv2.putText(out_img, class_pred + ": {:.2}".format(class_conf), (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0, 0, 255), 2, cv2.LINE_AA)
    if out_path is None:
        cv2.imshow("image", out_img)
        cv2.waitKey(0)
    else:
        if not os.path.exists(out_path):
            os.makedirs((out_path))
        cv2.imwrite(os.path.join(out_path,f), out_img)
    print("DONE!")

def predict_folder(in_path, out_path = "out", conf_threshold=0.6):
    print(f"Predicting images in {in_path}")
    print("Load weight from: " + args.trained_weights)
    model = get_model()
    model.load_weights(args.trained_weights)
    if not os.path.exists(out_path):
        os.makedirs((out_path))

    dnnFaceDetector = dlib.cnn_face_detection_model_v1("detector/mmod_human_face_detector.dat")
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for f in os.listdir(in_path):
        fname, ext = os.path.splitext(f)
        if ext.lower() not in valid_images:
            continue
        img = cv2.imread(os.path.join(in_path, f))
        out_img = img.copy()
        h, w, _ = img.shape
        extract_results = dnnFaceDetector(img, 1)

        for rect in extract_results:
            x1 = rect.rect.left()
            y1 = rect.rect.top()
            x2 = rect.rect.right()
            y2 = rect.rect.bottom()
            if rect.confidence < conf_threshold:
                continue
            image,face = predict_preprocess(img[:,:,::-1], [x1,y1,x2,y2])

            scores = model.call(tf.expand_dims(face,0),tf.expand_dims(image,0), training=False)
            y_pred = int(tf.argmax(scores, axis=1))
            class_conf = float(scores[0,y_pred])
            class_pred = config.class_names[y_pred]

            out_img = cv2.rectangle(out_img, (x1, y1), (x2, y2), [0, 0, 255], 2)

            out_img = cv2.putText(out_img, class_pred + ": {:.2}".format(class_conf) , (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX ,
                                1,  (0, 0, 255) ,  2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_path, f),out_img)
    print("DONE!")




if __name__ == "__main__":
    parse_arg()
    ext = os.path.splitext(args.input)[-1]
    if ext: #predict on single image
        if args.output:
            predict_image(args.input, args.output)
        else:
            predict_image(args.input, None)
    else: #predict on images on a directory
        if args.output:
            predict_folder(args.input, args.output)
        else:
            predict_folder(args.input)