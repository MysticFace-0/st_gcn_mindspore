import pickle
import mindspore.dataset as ds

from dataset.transform_posec3d import UniformSampleFrames, PoseDecode, PoseCompact, Resize, RandomResizedCrop
    # Resize, RandomResizedCrop, \
    # Flip, GeneratePoseTarget, FormatShape, Collect, ToTensor

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

PoseDecode = PoseDecode()
PoseCompact = PoseCompact(hw_ratio=1., allow_imgpad=True)
RandomResizedCrop = RandomResizedCrop(area_range=(0.56, 1.0))

# RandomResizedCrop = RandomResizedCrop()
# Flip = Flip(flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp)
# GeneratePoseTarget = GeneratePoseTarget(with_kp=True, with_limb=False)
#
# Collect = Collect()
# ToTensor = ToTensor()

class FLAG2DTrainDatasetGenerator():
    """
    dataset_dir: where the dataset locate
    """
    def __init__(self, dataset_dir, clip_len = 500, num_clips = 1, test_mode=False):
        with open(dataset_dir, "rb") as dataset:
            self.dataset = pickle.load(dataset) # No changed format dataset

        self.class_num = 60
        self.keypoint_num = 17
        self.dataset_len = len(self.dataset['split']['train'])
        self.dataset = self.dataset['annotations'][:20]#[:self.dataset_len]

        # origin: (1, 1045, 17, 2)
        self.UniformSampleFrames = UniformSampleFrames(clip_len, num_clips, test_mode) # (1, 1045, 17, 3)
        self.Resize1 = Resize(scale=(-1, 64))

        # for i in range(self.dataset_len):
        for i in range(20):
            self.dataset[i] = self.UniformSampleFrames.transform(self.dataset[i])
            self.dataset[i] = PoseDecode.transform(self.dataset[i])
            self.dataset[i] = PoseCompact.transform(self.dataset[i])
            self.dataset[i] = self.Resize1.transform(self.dataset[i])
            self.dataset[i] = RandomResizedCrop.transform(self.dataset[i])
            # self.dataset[i] = PreNormalize2D.transform(self.dataset[i])
            # self.dataset[i] = GenSkeFeat.transform(self.dataset[i])
            # self.dataset[i] = self.UniformSampleFrames.transform(self.dataset[i])
            # self.dataset[i] = PoseDecode.transform(self.dataset[i])
            # self.dataset[i] = FormatGCNInput.transform(self.dataset[i])
            # self.dataset[i] = Collect.transform(self.dataset[i])
            # self.dataset[i] = ToTensor.transform(self.dataset[i])

    def __getitem__(self, index):
        return self.dataset[index]['keypoint'], self.dataset[index]['label']

    def __len__(self):
        return 20

    def class_num(self):
        return self.class_num

if __name__=="__main__":
    dataset_generator = FLAG2DTrainDatasetGenerator("D:\\data\\flag2d.pkl")
    dataset = ds.GeneratorDataset(dataset_generator, ["keypoint", "label"], shuffle=True).batch(2, True)
    i=0
    for data in dataset.create_dict_iterator():
        i+=1
        if i>4:
            break
        # Tensor(32, 10, 1, 500, 17, 3) Tensor(32)
        # (Batch_size, num_clips, num_person, frames, num_keypoint, keypoint_location+keypoint_score) (label)
        print(data["keypoint"].shape, data["label"])