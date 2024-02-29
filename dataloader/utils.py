"""
Modified from https://github.com/KaiyangZhou/Dassl.pytorch

Copyright (c) 2020 Kaiyang, licensed under the MIT License
cf. 3rd-party-licenses.txt file in the root directory of this source tree

"""
import json
import os
from collections import defaultdict
import errno
import os.path as osp


def subsample_classes(*args, available_classes,relabel=True):
    """Divide classes into two groups. The first group
    represents base classes while the second group represents
    new classes.

    Args:
        args: a list of datasets, e.g. train, val and test.
        subsample (str): what classes to subsample.
    """
    if relabel:
        relabeler = {y: y_new for y_new, y in enumerate(available_classes)}
    else:
        relabeler = {y: y for y in available_classes}
    output, cnames = [], []
    for dataset in args:
        dataset_new, cname_new = [], []
        for item in dataset:
            if item.label not in available_classes:
                continue
            item_new = Datum(
                impath=item.impath,
                label=relabeler[item.label],
                classname=item.classname
            )
            dataset_new.append(item_new)
            cname_new.append(item.classname)

        output.append(dataset_new)
        cnames.append(cname_new)

    return output, cnames

def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    
def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj

class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, classname=""):
        # assert isinstance(impath, str)

        self._impath = impath
        self._label = label
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def classname(self):
        return self._classname



def read_split(filepath, path_prefix):
    def _convert(items):
        out,cnames = [],[]
        for impath, label, classname in items:
            impath = os.path.join(path_prefix, impath)
            item = Datum(impath=impath, label=int(label), classname=classname)
            out.append(item)
            cnames.append(classname)
        return out,cnames

    print(f"Reading split from {filepath}")
    split = read_json(filepath)
    train,train_cnames = _convert(split["train"])
    val,val_cnames = _convert(split["val"])
    test,test_cnames = _convert(split["test"])

    return train, val, test,train_cnames,val_cnames,test_cnames



class DatasetBase:
    """A unified dataset class
    """
    dataset_dir = ""  # the directory where the dataset is stored

    def __init__(self, train=None, val=None, test=None):
        self._train = train  # labeled training data
        self._val = val  # validation data (optional)
        self._test = test  # test data
        self._num_classes = self.get_num_classes(train)
        self._lab2cname, self._classnames,self._labels = self.get_lab2cname(train)

    @property
    def train(self):
        return self._train

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    @staticmethod
    def get_num_classes(data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    @staticmethod
    def get_lab2cname(data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping,classnames,labels

def split_dataset_by_label(data_source):
    """Split a dataset, i.e. a list of Datum objects,
    into class-specific groups stored in a dictionary.

    Args:
        data_source (list): a list of Datum objects.
    """
    output = defaultdict(list)

    for item in data_source:
        output[item.label].append(item)

    return output

