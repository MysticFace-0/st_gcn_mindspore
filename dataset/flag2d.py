import numpy as np
import pickle
import mindspore.dataset as ds
from torch.utils.data import Dataset

from dataset.transform import PreNormalize2D, GenSkeFeat, UniformSampleFrames, PoseDecode, FormatGCNInput, Collect, \
    ToTensor


class FLAG2DTrainDatasetGenerator():
    """
    dataset_dir: where the dataset locate
    """
    def __init__(self, dataset_dir, clip_len = 500, num_clips = 1, test_mode=False):
        with open(dataset_dir, "rb") as dataset:
            self.dataset = pickle.load(dataset) # No changed format dataset

        self.dataset_len = len(self.dataset['split']['train'])
        self.dataset = self.dataset['annotations'][:self.dataset_len]#[:self.dataset_len]

        # origin: (1, 1045, 17, 2)
        self.PreNormalize2D = PreNormalize2D() # (1, 1045, 17, 2)
        self.GenSkeFeat = GenSkeFeat() # (1, 1045, 17, 3)
        self.UniformSampleFrames = UniformSampleFrames(clip_len, num_clips, test_mode) # (1, 1045, 17, 3)
        self.PoseDecode = PoseDecode() # (1, 500, 17, 3)
        self.FormatGCNInput = FormatGCNInput() # (1, 1, 500, 17, 3)
        self.Collect = Collect() # (1, 1, 500, 17, 3)
        self.ToTensor = ToTensor() # (1, 1, 500, 17, 3)

        for i in range(self.dataset_len):
            self.dataset[i] = self.PreNormalize2D.transform(self.dataset[i])
            self.dataset[i] = self.GenSkeFeat.transform(self.dataset[i])
            self.dataset[i] = self.UniformSampleFrames.transform(self.dataset[i])
            self.dataset[i] = self.PoseDecode.transform(self.dataset[i])
            self.dataset[i] = self.FormatGCNInput.transform(self.dataset[i])
            self.dataset[i] = self.Collect.transform(self.dataset[i])
            self.dataset[i] = self.ToTensor.transform(self.dataset[i])

    def __getitem__(self, index):
        return self.dataset[index]['keypoint'], self.dataset[index]['label']

    def __len__(self):
        return self.dataset_len


class FLAG2DValDatasetGenerator():
    """
    dataset_dir: where the dataset locate
    """
    def __init__(self, dataset_dir, clip_len = 500, num_clips = 10, test_mode=True):
        with open(dataset_dir, "rb") as dataset:
            self.dataset = pickle.load(dataset) # No changed format dataset

        self.dataset_len = len(self.dataset['split']['val'])
        self.dataset = self.dataset['annotations'][len(self.dataset['split']['train']):]

        # origin: (1, 745, 17, 2)
        self.PreNormalize2D = PreNormalize2D() # (1, 745, 17, 2)
        self.GenSkeFeat = GenSkeFeat() # (1, 745, 17, 3)
        self.UniformSampleFrames = UniformSampleFrames(clip_len, num_clips, test_mode) # (1, 745, 17, 3)
        self.PoseDecode = PoseDecode() # (1, 5000, 17, 3)
        self.FormatGCNInput = FormatGCNInput() # (10, 1, 500, 17, 3)
        self.Collect = Collect() # (10, 1, 500, 17, 3)
        self.ToTensor = ToTensor() # (10, 1, 500, 17, 3)

        for i in range(self.dataset_len):
            self.dataset[i] = self.PreNormalize2D.transform(self.dataset[i])
            self.dataset[i] = self.GenSkeFeat.transform(self.dataset[i])
            self.dataset[i] = self.UniformSampleFrames.transform(self.dataset[i])
            self.dataset[i] = self.PoseDecode.transform(self.dataset[i])
            self.dataset[i] = self.FormatGCNInput.transform(self.dataset[i])
            self.dataset[i] = self.Collect.transform(self.dataset[i])
            self.dataset[i] = self.ToTensor.transform(self.dataset[i])

    def __getitem__(self, index):
        # Tensor:  int:
        return self.dataset[index]['keypoint'], self.dataset[index]['label']

    def __len__(self):
        return self.dataset_len


if __name__=="__main__":
    dataset_generator = FLAG2DValDatasetGenerator("D:\\data\\FLAG3D\\data\\flag2d.pkl")
    dataset = ds.GeneratorDataset(dataset_generator, ["keypoint", "label"], shuffle=True).batch(4, True)
    for data in dataset.create_dict_iterator():
        # Tensor(32, 10, 1, 500, 17, 3) Tensor(32)
        # (Batch_size, num_clips, num_person, frames, num_keypoint, keypoint_location+keypoint_score) (label)
        print(data["keypoint"].shape, data["label"])