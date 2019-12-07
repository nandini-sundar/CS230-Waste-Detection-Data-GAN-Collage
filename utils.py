import glob
import os
import cv2
import pathlib
import random
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

def get_image_paths_labels(root_path):
    #'''Ref: CS230 TA Session 5 : Tesorflow vs Pytorch
    #Link: https://colab.research.google.com/drive/1HzN2f0Mypj0r2rKJdKYCjczM1WzJyoaV#scrollTo=lzBchLwtgHk7
    #
    print(root_path)

    root_path = pathlib.Path(root_path)
    all_image_paths = [f for f in root_path.glob("**/*.jpg")]
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    label_names = sorted(item.name for item in root_path.glob('*/') if item.is_dir())

    print("Label names are {}".format(label_names))
    num_classes = len(label_names)

    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print("Label to index are {}".format(label_to_index))

    all_image_label_text = [pathlib.Path(path).parent.name for path in all_image_paths]
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
    # print("All labels are {} : {}".format(all_image_words, all_image_labels))
    return all_image_paths, all_image_labels, all_image_label_text, num_classes


def forground_mask(img):
    '''Ref: Grab-cut using
        Link: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html

    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((10, 10), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    #     # Finding sure foreground area
    #     dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    #     ret, sure_fg = cv2.threshold(dist_transform,0.001*dist_transform.max(),255,0)

    cv2.normalize(sure_bg, sure_bg, 0, 1, cv2.NORM_MINMAX)

    return sure_bg

def reshape(train_images):

    # Convert list to numpy
    train_images = np.asarray(train_images)

    # Reshape Images
    num_train_samples, img_rows, img_cols = train_images.shape

    if K.image_data_format() == 'channels_first':
        x_train = train_images.reshape(num_train_samples, 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = train_images.reshape(num_train_samples, img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # Because we have to divide by 255. We have change the type
    x_train = x_train.astype('float32')
    x_train /= 255

    return x_train, input_shape, num_train_samples

