import numpy as np
import pickle
import mindspore.dataset as ds
from torch.utils.data import Dataset


class FLAG3DTrainDatasetGenerator():
    """
    dataset_dir: where the dataset locate
    """
    def __init__(self, dataset_dir):
        with open(dataset_dir, "rb") as dataset:
            origin_dataset = pickle.load(dataset) # No changed format dataset

        self.dataset_len = len(origin_dataset['split']['train'])
        self.dataset = origin_dataset['annotations'][:self.dataset_len]

    def __getitem__(self, index):
        return self.dataset[index]['keypoint'], self.dataset[index]['label']

    def __len__(self):
        return self.dataset_len


class FLAG3DValDatasetGenerator():
    """
    dataset_dir: where the dataset locate
    """
    def __init__(self, dataset_dir):
        with open(dataset_dir, "rb") as dataset:
            origin_dataset = pickle.load(dataset) # No changed format dataset

        self.dataset_len = len(origin_dataset['split']['val'])
        self.dataset = origin_dataset['annotations'][len(origin_dataset['split']['train']):]

    def __getitem__(self, index):
        return self.dataset[index]['keypoint'], self.dataset[index]['label']

    def __len__(self):
        return self.dataset_len


if __name__=="__main__":
    dataset_generator = FLAG3DValDatasetGenerator("D:\\data\\FLAG3D\\data\\flag3d.pkl")
    dataset = ds.GeneratorDataset(dataset_generator, ["keypoint", "label"], shuffle=True).batch(32, True)
    for data in dataset.create_dict_iterator():
        print(data["label"])