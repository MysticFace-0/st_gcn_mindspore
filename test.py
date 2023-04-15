import mindspore
from mindspore import ops

from evaluation.evaluation import top_k_accuracy
from model.posec3d.recognizer3d import Recognizer3d
from dataset import FLAG2DPoseC3DValDatasetGenerator
import mindspore.dataset as ds


def val(dataset_val, model):
    i = 0  # iteration num
    total_acc_top1 = 0.
    total_acc_top5 = 0.

    model.set_train(False)
    for data in dataset_val.create_dict_iterator():
        x = data["keypoint"]
        label = data["label"].to(mindspore.int32)
        y = model(x)
        batch_acc = top_k_accuracy(y.numpy(), label.numpy(), (1,5))
        print(label,ops.argmax(y,axis=1))

        total_acc_top1 += batch_acc[0]
        total_acc_top5 += batch_acc[1]

        i += 1


    accuracy_top1 = total_acc_top1 / i
    accuracy_top5 = total_acc_top5 / i
    print("accuracy_top1: ", accuracy_top1 ,"accuracy_top5: ", accuracy_top5)

dataset_val_generator = FLAG2DPoseC3DValDatasetGenerator("D:/data/flag2d.pkl", 500)
dataset_val = ds.GeneratorDataset(dataset_val_generator, ["keypoint", "label"], shuffle=True).batch(
        1, True)
print("dataset success")

model = Recognizer3d(
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2,),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1),
        num_classes=60,
        in_channels_head=512,
        spatial_type='avg',
        dropout_ratio=0.5
    )
print("model success")
# for param in model.trainable_params():
#     print(param)
model.set_train(False)
for param in model.trainable_params():
    print(param)

param_dict = mindspore.load_checkpoint("./chpk_resume/posec3d_2d.ckpt")
for i in param_dict:
    print(param_dict[i])
mindspore.load_param_into_net(model, param_dict)
print("chpk success")

val(dataset_val, model)
print("val success")

