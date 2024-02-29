"""
Modified from https://github.com/KaiyangZhou/CoOp

Copyright (c) 2021 Kaiyang Zhou, licensed under the MIT License
cf. 3rd-party-licenses.txt file in the root directory of this source tree

"""

from dataloader.utils import *
import os
import pickle
from collections import OrderedDict



class Caltech101(DatasetBase):
    dataset_dir = "caltech-101"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "Caltech101"

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
        # self.split_fewshot_dir = os.path.join(self.dataset_dir,
        #                                       "split_fewshot")
        # mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
            # self.all_classnames = get_lab2cname(test)

        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)


class OxfordFlowers(DatasetBase):
    dataset_dir = "oxford_flowers"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "OxfordFlowers"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")
        # self.split_fewshot_dir = os.path.join(self.dataset_dir,
        #                                       "split_fewshot")
        # mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
            # self.all_classnames = get_lab2cname(test)
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


class EuroSAT(DatasetBase):
    dataset_dir = "eurosat"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "EuroSAT"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "2750")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
            # self.all_classnames = get_lab2cname(test)
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


class OxfordPets(DatasetBase):
    dataset_dir = "oxford_pets"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "OxfordPets"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
            # self.all_classnames = get_lab2cname(test)
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames

        super().__init__(train=train, val=val, test=test)


class FGVCAircraft(DatasetBase):
    dataset_dir = "fgvc_aircraft"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "FGVCAircraft"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train, train_cnames = self.read_data(cname2lab, "images_variant_train.txt")
        val, val_cnames = self.read_data(cname2lab, "images_variant_val.txt")
        test, test_cnames = self.read_data(cname2lab, "images_variant_test.txt")
        # self.all_classnames = get_lab2cname(test)
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items, cnames = [], []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                cnames.append(classname)
        return items, cnames


class Food101(DatasetBase):
    dataset_dir = "food-101"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "Food101"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
            # self.all_classnames = get_lab2cname(test)
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


class DescribableTextures(DatasetBase):
    dataset_dir = "dtd"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "DescribableTextures"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_DescribableTextures.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
            # self.all_classnames = get_lab2cname(test)
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


class UCF101(DatasetBase):
    dataset_dir = "ucf101"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "UCF101"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
            # self.all_classnames = get_lab2cname(test)
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


class StanfordCars(DatasetBase):
    dataset_dir = "stanford_cars"

    def __init__(self, cfg,available_classes=None,relabel=True):

        self.data_name = "StanfordCars"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")

        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.dataset_dir)
            # self.all_classnames = get_lab2cname(test)
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


class SUN397(DatasetBase):
    dataset_dir = "sun397"

    def __init__(self, cfg,available_classes=None,relabel=True):
        self.data_name = "SUN397"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "SUN397")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_SUN397.json")
        if os.path.exists(self.split_path):
            train, val, test, train_cnames, val_cnames, test_cnames = read_split(self.split_path, self.image_dir)
            # self.all_classnames = get_lab2cname(test)
        if available_classes is not None:
            output, cnames = subsample_classes(train, val, test, available_classes=available_classes,relabel=relabel)
            train, val, test = output[0], output[1], output[2]
            train_cnames, val_cnames, test_cnames = cnames[0], cnames[1], cnames[2]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, val_cnames, test_cnames
        super().__init__(train=train, val=val, test=test)


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


class ImageNet(DatasetBase):
    dataset_dir = "imagenet"

    def __init__(self, cfg,available_classes=None,relabel=True):
        self.data_name = 'ImageNet'
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
                train_cnames = preprocessed["train_cnames"]
                test_cnames = preprocessed["test_cnames"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train, train_cnames = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test, test_cnames = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test, "train_cnames": train_cnames, "test_cnames": test_cnames}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)
        # self.all_classnames = get_lab2cname(test)
        if available_classes is not None:
            output, cnames = subsample_classes(train, test, available_classes=available_classes,relabel=relabel)
            train, test = output[0], output[1]
            train_cnames, test_cnames = cnames[0], cnames[1]
        self.train_cnames, self.val_cnames, self.test_cnames = train_cnames, test_cnames, test_cnames

        print('Imagenet is loaded.')

        super().__init__(train=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []
        all_cnames = []
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                all_cnames.append(classname)
        return items, all_cnames
TO_BE_IGNORED = ["README.txt","class_to_idx.json",""]

class ImageNetA(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-adversarial"

    def __init__(self, cfg):
        self.data_name = 'ImageNetA'
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir =os.path.join(self.dataset_dir, "imagenet-a")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data,all_cnames = self.read_data(classnames)
        self.train_cnames, self.val_cnames, self.test_cnames = all_cnames, all_cnames, all_cnames
        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        TO_BE_IGNORED = ["README.txt"]
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []
        all_cnames = []
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                all_cnames.append(classname)
        return items, all_cnames

class ImageNetR(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-rendition"
    def __init__(self, cfg):
        self.data_name = 'ImageNetR'
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "imagenet-r")
        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data,all_cnames = self.read_data(classnames)
        self.train_cnames, self.val_cnames, self.test_cnames = all_cnames, all_cnames, all_cnames
        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        TO_BE_IGNORED = ["README.txt", "class_to_idx.json", "dataset.h5"]
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []
        all_cnames = []
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                all_cnames.append(classname)
        return items, all_cnames

class ImageNetSketch(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-sketch"
    def __init__(self, cfg):
        self.data_name = 'ImageNetSketch'
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data,all_cnames = self.read_data(classnames)
        self.train_cnames, self.val_cnames, self.test_cnames = all_cnames, all_cnames, all_cnames
        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        TO_BE_IGNORED = ["README.txt", "class_to_idx.json", "dataset.h5"]
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []
        all_cnames = []
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                all_cnames.append(classname)
        return items, all_cnames

class ImageNetV2(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """
    dataset_dir = "imagenetv2"

    def __init__(self, cfg):
        self.data_name = 'ImageNetV2'
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        image_dir = "imagenetv2-matched-frequency-format-val"
        self.image_dir = os.path.join(self.dataset_dir, image_dir)

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data,all_cnames = self.read_data(classnames)
        self.train_cnames, self.val_cnames, self.test_cnames = all_cnames, all_cnames, all_cnames
        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = list(classnames.keys())
        items = []
        all_cnames = []
        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            imnames = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                all_cnames.append(classname)
        return items,all_cnames