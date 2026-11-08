import pickle
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self, single_channel=False):
        print("loading data...")
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        raw = pickle.load(f)
        f.close()

        train = self.convert(raw['train'], single_channel)
        test = self.convert(raw['test'], single_channel)

        print("Train label distribution:", np.unique(train['y'], return_counts=True))
        print("Test label distribution:", np.unique(test['y'], return_counts=True))

        # labels start at 0
        offset = min(train['y'].min(), test['y'].min())
        train['y'] -= offset
        test['y'] -= offset
        return {'train': train, 'test': test}

    def convert(self, instances, single_channel):
        X = []
        y = []
        for instance in instances:
            X.append(instance['image'])
            y.append(instance['label'])
        X = np.array(X, dtype=np.float32) / 255.0
        y = np.array(y, dtype=np.int64)

        if single_channel and X.ndim == 4:
            X = X[:, :, :, 0]
        if X.ndim == 3:
            X = X[:, None, :, :]
        else:
            X = X.transpose(0, 3, 1, 2)
        return {'X': X, 'y': y}