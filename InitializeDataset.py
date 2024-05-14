import os
import random
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import csv
import math
from TrackNetDataset import TrackNetDataset

class InitializeDataset():
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.x = []
        self.x_train = []
        self.x_validate = []
        self.x_test = []
        self.y_train = []
        self.y_validate = []
        self.y_test = []

    def initialize_data(self):
        for root, dirs, files in os.walk(self.dataset_path):
            self.x.extend([os.path.join(root, f) for f in files if f.endswith(".jpg")])

    def read_labels(self):
        if not self.x_train:
            print("Data hasn't been divided to train, validate and test!!!")
            return
        #reading all labels
        labels = {}
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".csv"):
                    labels.update(self.read_csv(os.path.join(root, file), root))
        #dividing labels to train, validate, test datasets
        y_lists = [self.y_train, self.y_validate, self.y_test]
        x_lists = [self.x_train, self.x_validate, self.x_test]
        for x_list, y_list in zip(x_lists, y_lists):
            for x in x_list:
                y_list.append(labels[x])

    def read_csv(self, file, path):
        labels = {}
        with open(file, mode='r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)    #skip header line
            for row in csv_reader:
                labels[os.path.join(path, row[0])] = (row[2], row[3])
        return labels

    def random_split(self, size):
        random.shuffle(self.x)
        self.x_train = self.x[:size[0]]
        self.x_validate = self.x[size[0]:size[0]+size[1]]
        self.x_test = self.x[size[0]+size[1]:]


    def write_split(self, file):
        if not self.x_train:
            print("Data hasn't been divided to train, validate and test!!!")
            return
        with open(file, "w") as file:
            for item in self.x_train:
                tmp = Path(item)
                tmp = Path(*tmp.parts[len(self.dataset_path.parts):])
                file.write(str(tmp) + "\n")
            file.write("\n")
            for item in self.x_validate:
                tmp = Path(item)
                tmp = Path(*tmp.parts[len(self.dataset_path.parts):])
                file.write(str(tmp) + "\n")
            file.write("\n")
            for item in self.x_test:
                tmp = Path(item)
                tmp = Path(*tmp.parts[len(self.dataset_path.parts):])
                file.write(str(tmp) + "\n")
            file.write("\n")

    def read_split(self, file):
        data = [self.x_train, self.x_validate, self.x_test]
        data_i = 0
        with open(file, "r") as file:
            for line in file:
                if line.strip(): #Non-empty line
                    data[data_i].append(os.path.join(self.dataset_path, line.strip()))
                else:
                    data_i += 1

    def stats(self):
        if not self.x:
            print("Data hasn't been read!!!")
        else:
            print("Number of data:\t" + str(len(self.x)))

        if not self.x_train:
            print("Data hasn't been divided to train, validate and test!!!")
        else:
            print("Number of train data:\t" + str(len(self.x_train)))
            print("Number of validate data:\t" + str(len(self.x_validate)))
            print("Number of test data:\t" + str(len(self.x_test)))

    def train_dataset(self, transform=None, target_transform=None):
        return TrackNetDataset(self.y_train, self.x_train, transform, target_transform)

    def validate_dataset(self, transform=None, target_transform=None):
        return TrackNetDataset(self.y_validate, self.x_validate, transform, target_transform)

    def test_dataset(self, transform=None, target_transform=None):
        return TrackNetDataset(self.y_test, self.x_test, transform, target_transform)