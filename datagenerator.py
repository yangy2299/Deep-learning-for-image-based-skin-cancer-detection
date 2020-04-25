import numpy as np
import cv2
import utils
import pandas as pd
from os import walk
from os.path import join
import os
from sklearn.model_selection import StratifiedShuffleSplit

class ImageDataGenerator:
    def __init__(self, path, horizontal_flip=False, shuffle=False,
                 mean=np.array([104., 117., 124.]), scale_size=(224, 224),
                 nb_classes=2):

        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        # self.mean = mean
        self.scale_size = scale_size
        self.train_pointer = 0
        self.test_pointer = 0
        self.read_class_list(path)

        if self.shuffle:
            self.shuffle_data()

    def y_value(self, label):
        # if text == 'relevant':
        if label == 'bkl':
        # if text == 'on-topic':
        # if text == 'Related and informative':
            return 0
        elif label == 'nv':
            return 1
        elif label == 'df':
            return 2
        elif label == 'mel':
            return 3
        elif label == 'vasc':
            return 4
        elif label == 'bcc':
            return 5
        else:
            return 6

    def x_path(self, x):
        # if text == 'relevant':
        return 'data_image/{}.jpg'.format(x)

    def read_class_list(self, path):
        """
        Scan the image file and get the image paths and labels
        """
        # file_path = path
        # im_names = []
        # for root, dirs, files in os.walk(file_path, topdown=False):
        #     for name in files:
        #         if os.path.splitext(os.path.join(root, name))[1].lower() == ".jpeg":
        #             # if name.split('.')[0].split('-')[-1].lower()=='5x':
        #             im_names.append(os.path.join(root, name))

        input_file = os.path.join(path)
        # with open(input_file, 'r') as f:
        #     data = f.read()
        # df = pd.read_csv(input_file, sep='\t')
        df = pd.read_csv(input_file)#, lineterminator='\n')
        #df = df[1:2000]
        # df = df.loc[(df['text_human'].isin(['affected_individuals','infrastructure_and_utility_damage',
        #     'injured_or_dead_people','rescue_volunteering_or_donation_effort'])) | 
        #       (df['text_info'].isin(['not_informative']))]
        # df = df.loc[df[' Informativeness'].isin(['Related and informative','Related - but not informative'])]

        # x_raw = df['text'].apply(lambda x: self.clean_str(x)).tolist()
        # x_raw = df[' tweet'].apply(lambda x: p.tokenize(x)).tolist()
        self.images = df['image_id'].apply(lambda x: self.x_path(x)).tolist()
        self.labels = df['dx'].apply(lambda x: self.y_value(x)).tolist()

        x_img_raw_array = np.array(self.images)
        y_raw_array = np.array(self.labels)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
        sss.get_n_splits(x_img_raw_array, y_raw_array)
        for train_idx, test_idx in sss.split(x_img_raw_array, y_raw_array):
            x_img_train_array = x_img_raw_array[train_idx]
            x_img_test_array = x_img_raw_array[test_idx]

            y_train = y_raw_array[train_idx]
            y_test = y_raw_array[test_idx]

        #train_imgs, test_imgs, train_labels, test_labels = train_test_split(x_raw_array, y_raw_array, test_size=0.3, random_state=1)

        self.train_images = list(x_img_train_array)
        self.train_labels = list(y_train)

        self.test_images = list(x_img_test_array)
        self.test_labels = list(y_test)

        #im_names = next(walk(path))[2]
        #num_files = len(im_names)
        # self.images = []
        # self.labels = []
        # #filenames=filenames[2:]
        # for i, filename in enumerate(im_names):
        #     label = filename.split("/")[2]
        #     if label != 'unlabelled':
        #         #self.images.append(join(path, filename))
        #         if label == 'trainA':
        #             self.images.append(join(filename))
        #             self.labels.append(1)
        #         elif label == 'trainB':
        #             self.images.append(join(filename))
        #             self.labels.append(0)
        self.train_data_size = len(self.train_labels)
        self.test_data_size = len(self.test_labels)

    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = list(self.train_images)
        labels = list(self.train_labels)
        self.train_images = []
        self.train_labels = []

        # create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.train_images.append(images[i])
            self.train_labels.append(labels[i])

    def reset_train_pointer(self):
        """
        reset train_pointer to begin of the list
        """
        self.train_pointer = 0

        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        paths = self.train_images[self.train_pointer:self.train_pointer + batch_size]
        labels = self.train_labels[self.train_pointer:self.train_pointer + batch_size]
        # update train_pointer
        self.train_pointer += batch_size

        # Read images
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            #print(paths[i])
            img = utils.load_image(paths[i])
            #img = cv2.imread(paths[i])
            # flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)
            # rescale image
            #img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            #utils.load_image()
            #img = img.astype(np.float32)

            # subtract mean
            #img -= self.mean

            images[i] = img

        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        # return array of images and labels
        return images, one_hot_labels


    def test_next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        paths = self.test_images[self.test_pointer:self.test_pointer + batch_size]
        labels = self.test_labels[self.test_pointer:self.test_pointer + batch_size]
        # update train_pointer
        self.test_pointer += batch_size

        # Read images
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            #print(paths[i])
            img = utils.load_image(paths[i])
            #img = cv2.imread(paths[i])
            # flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)
            # rescale image
            #img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            #utils.load_image()
            #img = img.astype(np.float32)

            # subtract mean
            #img -= self.mean

            images[i] = img

        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        # return array of images and labels
        return images, one_hot_labels