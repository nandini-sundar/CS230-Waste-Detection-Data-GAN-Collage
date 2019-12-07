#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
import random
import pathlib
import argparsedd
from tensorflow import keras


from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint


def set_random_seed():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


set_random_seed()

tf.__version__

def get_image_paths_labels(root_path):
    '''Ref: CS230 TA Session 5 : Tesorflow vs Pytorch 
    Link: https://colab.research.google.com/drive/1HzN2f0Mypj0r2rKJdKYCjczM1WzJyoaV#scrollTo=lzBchLwtgHk7
    '''
    all_image_paths = [f for f in root_path.glob("**/*.jpg")]
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths) 

    label_names = sorted(item.name for item in root_path.glob('*/') if item.is_dir())
    
    print("Label names are {}".format(label_names))
    num_classes = len(label_names)

    label_to_index = dict((name, index) for index,name in enumerate(label_names))
    print("Label to index are {}".format(label_to_index))
    
    all_image_label_text = [pathlib.Path(path).parent.name for path in all_image_paths]
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
    #print("All labels are {} : {}".format(all_image_words, all_image_labels))
    return all_image_paths, all_image_labels, all_image_label_text, num_classes

def random_select_images_labels(image_paths, image_labels, k):
    '''Select different images and labels'''
    l1 = len(image_paths)
    l2 = len(image_labels)
    if l1 != l2: # Make sure the lengths are same before going forward 
        print("Number of images not equal to number of labels")
        raise
        
    rdm_idx = np.random.choice(range(len(image_labels)), k)
    
    rdm_image_paths = [image_paths[i] for i in rdm_idx]
    rdm_image_labels = [image_labels[i] for i in rdm_idx] 
    
    return rdm_image_paths, rdm_image_labels

'''
We first verify that the dataset was loaded correctly
by viewing the images using opencv. Takes list of paths as input
Ref: CS230 TA Session 5 : Tesorflow vs Pytorch 
Link: https://colab.research.google.com/drive/1HzN2f0Mypj0r2rKJdKYCjczM1WzJyoaV#scrollTo=lzBchLwtgHk7
'''
def view_dataset_paths(paths, labels, method='cv2'):
    N = len(paths)
    cols = 2
    rows = int(np.ceil(N / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    flatted_axs = [item for one_ax in axs for item in one_ax]
    for ax, path, label in zip(flatted_axs, paths, labels):
        if method == 'cv2':
            img = cv2.imread(path, 3)
            ax.imshow(img)
            ax.set_title(label)
            ax.axis('off')
    plt.show() 

'''
We verify that the dataset was loaded correctly
by viewing the images using opencv.
Takes list of images as input
Ref: CS230 TA Session 5 : Tesorflow vs Pytorch 
Link: https://colab.research.google.com/drive/1HzN2f0Mypj0r2rKJdKYCjczM1WzJyoaV#scrollTo=lzBchLwtgHk7
'''

def view_dataset_images(list_imgs, labels, method='cv2'):
    N = len(list_imgs)
    cols = 2
    rows = int(np.ceil(N / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    flatted_axs = [item for one_ax in axs for item in one_ax]
    
    for ax, img, label in zip(flatted_axs, list_imgs, labels):
        img = np.squeeze(img)
        ax.imshow(img, cmap='gray')
        ax.set_title(label)
        ax.axis('off')
    plt.show() 


def forground_mask(img):
    '''Ref: Grab-cut using 
        Link: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html
        
    '''
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # noise removal
    kernel = np.ones((10, 10),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

#     # Finding sure foreground area
#     dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
#     ret, sure_fg = cv2.threshold(dist_transform,0.001*dist_transform.max(),255,0)

    cv2.normalize(sure_bg, sure_bg, 0, 1, cv2.NORM_MINMAX)
    
    return sure_bg

def transform(mask, hm, verbose=False):
    '''Given a mask, it transforms using the homogeneous matrix given'''
    '''hm = [tx, ty, s, theta]'''
    tx, ty, theta, s = hm
    
    if len(mask.shape) != 3:
        print("Image of rows * col * ch is needed. Dimension {} given".
             format(len(mask.shape)))
        
    rows,cols, ch = mask.shape
    ## Translate
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Setting the border to reflect so that we do not waste data - borderMode=cv2.BORDER_WRAP
    dst_translate = cv2.warpAffine(mask, M_translate, (cols,rows))
    
    if verbose == True:
        print("Translating by \n{}".format(M_translate))
    
    #Rotate and Scale
    M_rotate = cv2.getRotationMatrix2D((cols/2,rows/2),theta,s)
    
    # Setting the border to reflect so that we do not waste data - borderMode=cv2.BORDER_WRAP
    dst_rotate = cv2.warpAffine(dst_translate, M_rotate, (cols,rows) )
    if verbose == True:
        print("Rotating by \n{}".format(M_rotate))
    
    if ch == 1: # Normalize only if gray image
        cv2.normalize(dst_rotate, dst_rotate, 0, 1, cv2.NORM_MINMAX)
        
    if verbose == True:
        print("Max: {}".format(np.max(dst_rotate)))
        
    dst_rotate = np.reshape(dst_rotate, mask.shape)
    return dst_rotate

def mask_union(masks_to_merge):
    print(masks_to_merge.shape)
    num_training, train_rows, train_cols, channels = masks_to_merge.shape
    
    merge_resulting = np.zeros((train_rows, train_cols, channels), dtype='float32')
    print(merge_resulting.shape)
    for i in range(num_training):
        #print(masks_to_merge[i].shape)
        merge_resulting = cv2.bitwise_or(merge_resulting, masks_to_merge[i])
        
    return merge_resulting

def mask_intersection(masks_to_merge):
    print(masks_to_merge.shape)
    num_training, train_rows, train_cols, channels = masks_to_merge.shape
    
    merge_resulting = np.ones((train_rows, train_cols, channels), dtype='float32')
    print(merge_resulting.shape)
    for i in range(num_training):
        #print(masks_to_merge[i].shape)
        merge_resulting = cv2.bitwise_and(merge_resulting, masks_to_merge[i])
        
    return merge_resulting


# In[24]:


def merge_masks(masks_to_merge):
    '''Merge list of masks into one mask'''
    merge_resulting = mask_union(masks_to_merge=masks_to_merge)
    return merge_resulting


def collage_loss(list_masks):
    merge_union = mask_union(masks_to_merge=list_masks)
    merge_intersection = mask_intersection(masks_to_merge=list_masks)
    
    intersection_scr = np.sum(merge_intersection)
    print("Intersection Score {}".format(intersection_scr))
    
    union_scr = np.sum(merge_union)
    print("Union score {}".format(union_scr))
    
    individual_sums = [np.sum(lm) for lm in list_masks]
    print("individual_sums {}".format(individual_sums))
    
    individual_sum_of_sums = np.sum(individual_sums)
    print("individual_sum_of_sums {}".format(individual_sum_of_sums))
    
    combined_loss = (individual_sum_of_sums - union_scr) - intersection_scr
    print("combined_loss {}".format(combined_loss))
    
    return combined_loss


# Define custom loss
def model_collage_loss(model_input_left_resize, model_input_right_resize):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        left_translate = tf.keras.layers.Lambda(lambda image: tfa.image.translate(model_input_left_resize,
                                                                                  y_pred,
                                                                                  interpolation='NEAREST'))(
            model_input_left_resize)

        right_translate = tf.keras.layers.Lambda(lambda image: tfa.image.translate(model_input_right_resize,
                                                                                   y_pred,
                                                                                   interpolation='NEAREST')) \
            (model_input_right_resize)

        left_translate = tf.keras.layers.ThresholdedReLU(theta=0.5)(left_translate)
        left_translate = K.cast(left_translate, dtype='uint8')

        right_translate = tf.keras.layers.ThresholdedReLU(theta=0.5)(right_translate)
        right_translate = K.cast(right_translate, dtype='uint8')
        combined_mask_union = tf.keras.layers.Lambda(lambda image: tf.bitwise.bitwise_or(left_translate,
                                                                                         right_translate, name='union')) \
            ([left_translate, right_translate])

        combined_mask_intersection = tf.keras.layers.Lambda(lambda image: tf.bitwise.bitwise_and(left_translate,
                                                                                                 right_translate,
                                                                                                 name='intersection')) \
            ([left_translate, right_translate])

        val = K.sum(combined_mask_union) - K.sum(combined_mask_intersection)
        val = K.cast(val, dtype='float64')

        return val

    # Return a function
    return loss



def run(data_filepath, verbose=False):


    x_train_left, x_train_left_label_text, y_left,  \
    x_train_right, x_train_right_label_text, y_right, input_shape, num_classes = get_dataset(data_filepath)

    train_model(input_shape, x_train_left, x_train_right)



    print("Done")


def train_model(input_shape, x_train_left, x_train_right):
    model_input_left = tf.keras.Input(shape=input_shape, name='img_left')
    model_input_left_resize = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (int(input_shape[0] / 2),
                                                                                           int(input_shape[1] / 2))))(
        model_input_left)
    model_input_right = tf.keras.Input(shape=input_shape, name='img_right')
    model_input_right_resize = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (int(input_shape[0] / 2),
                                                                                            int(input_shape[1] / 2))
                                                                                    ))(model_input_right)
    # Left Branch
    x_left = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(model_input_left_resize)
    x_left = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x_left)
    x_left = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_left)
    x_left = tf.keras.layers.Dropout(0.25)(x_left)
    x_left = tf.keras.layers.Flatten()(x_left)
    x_left = tf.keras.layers.Dense(128, activation='relu')(x_left)
    x_left = tf.keras.layers.Dropout(0.5)(x_left)
    left_dx_dy = tf.keras.layers.Dense(2, activation='softmax', name="left_translation")(x_left)
    # Right Branch
    x_right = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(model_input_right_resize)
    x_right = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x_right)
    x_right = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_right)
    x_right = tf.keras.layers.Dropout(0.25)(x_right)
    x_right = tf.keras.layers.Flatten()(x_right)
    x_right = tf.keras.layers.Dense(128, activation='relu')(x_right)
    x_right = tf.keras.layers.Dropout(0.5)(x_right)
    right_dx_dy = tf.keras.layers.Dense(2, activation='softmax', name="right_translation")(x_right)
    model = tf.keras.Model(inputs=[model_input_left, model_input_right], outputs=[left_dx_dy, right_dx_dy],
                           name='collage_model')
    model.compile(loss=model_collage_loss(model_input_left_resize=model_input_left_resize,
                                          model_input_right_resize=model_input_right_resize),
                  optimizer=keras.optimizers.Adadelta())
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    training_history = \
        model.fit(x={'img_left': x_train_left,
                     'img_right': x_train_right},
                  epochs=512, verbose=2)
    model.save('collage_model.h5')
    # The returned "history" object holds a record
    # of the loss values and metric values during training
    print('\nhistory dict:', training_history.history)


def get_left_right_img_masks(all_fgms, all_images, all_labels, image_label_text, verbose):
    # Make Left Images
    left_all_fgms, left_all_images, left_all_labels, left_image_label_text = shuffle_images_and_labels(all_fgms,
                                                                                                       all_images,
                                                                                                       all_labels,
                                                                                                       image_label_text)



    if verbose:
        k = 4  # Get 4 Images
        view_random_images(left_all_fgms, left_all_images, left_image_label_text, k)
    # Make Right Images
    right_all_fgms, right_all_images, right_all_labels, right_image_label_text = shuffle_images_and_labels(all_fgms,
                                                                                                           all_images,
                                                                                                           all_labels,
                                                                                                           image_label_text)
    if verbose:
        k = 4  # Get 4 Images
        view_random_images(right_all_fgms, right_all_images, right_image_label_text, k)

    #Convert to np-asarray
    left_all_fgms = np.asarray(left_all_fgms)
    right_all_fgms = np.asarray(right_all_fgms)

    return left_all_fgms, left_image_label_text, left_all_labels,  right_all_fgms, right_image_label_text, right_all_labels


def shuffle_images_and_labels(all_fgms, all_images, all_labels, image_label_text):
    idx = np.arange(len(all_labels))
    np.random.shuffle(idx)
    rdm_all_fgms = [all_fgms[i] for i in idx]
    rdm_all_images = [all_images[i] for i in idx]
    rdm_all_labels = [all_labels[i] for i in idx]
    rdm_image_label_text = [image_label_text[i] for i in idx]
    return rdm_all_fgms, rdm_all_images, rdm_all_labels, rdm_image_label_text


def view_random_images(all_fgms, all_images, image_label_text, k):
    rdm_idx = np.random.choice(range(len(all_fgms)), k)
    rdm_images = [all_images[i] for i in rdm_idx]
    rdm_fgms = [all_fgms[i] for i in rdm_idx]
    rdm_image_labels = [image_label_text[i] for i in rdm_idx]
    view_dataset_images(rdm_images, rdm_image_labels, method='cv2')
    view_dataset_images(rdm_fgms, rdm_image_labels, method='cv2')


def get_dataset(data_filepath):
    ## Load Paths
    data_root = pathlib.Path(data_filepath)
    image_paths, image_labels, image_label_text, num_classes = get_image_paths_labels(data_root)
    # Get Images
    all_images = [cv2.imread(path, 3) for path in image_paths]
    all_fgms = [forground_mask(img) for img in all_images]

    x_train_left, x_train_left_label_text,  x_train_left_label, \
    x_train_right, x_train_right_label_text, x_train_right_label = get_left_right_img_masks(all_fgms, all_images, image_labels, image_label_text,
                                                             verbose=False)

    #Reshape Images
    num_train_samples, img_left_rows, img_left_cols = x_train_left.shape
    _, img_right_rows, img_right_cols = x_train_right.shape

    if tf.keras.backend.image_data_format() == 'channels_first':
        x_train_left = x_train_left.reshape(num_train_samples, 1, img_left_rows, img_left_cols)
        x_train_right = x_train_right.reshape(num_train_samples, 1, img_right_rows, img_right_cols)
        input_shape = (1, img_left_rows, img_left_cols)
    else:
        x_train_left = x_train_left.reshape(num_train_samples, img_left_rows, img_left_cols, 1)
        x_train_right = x_train_right.reshape(num_train_samples, img_right_rows, img_right_cols, 1)
        input_shape = (img_left_rows, img_left_cols, 1)

    # Because we have to divide by 255. We have change the type
    x_train_left = x_train_left.astype('float32')
    x_train_right = x_train_right.astype('float32')

    x_train_left /= 255
    x_train_right /= 255



    print("x_train_left.shape {}".format(x_train_left.shape))
    print("x_train_right.shape {}".format(x_train_right.shape))

    # convert to categorical

    y_left = tf.keras.utils.to_categorical(x_train_left_label, num_classes=num_classes)
    y_right = tf.keras.utils.to_categorical(x_train_right_label, num_classes=num_classes)

    return x_train_left, x_train_left_label_text, y_left, x_train_right, \
           x_train_right_label_text, y_right, input_shape, num_classes

def get_args():
    parser = argparse.ArgumentParser(description='Collage as a Optimization Problem')
    parser.add_argument('--path', '-p', type=str, dest='path', help='path where the data lives')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    run(args.path)
