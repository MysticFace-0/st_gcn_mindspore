import argparse
import os

import mindspore

from dataset import FLAG2DTrainDatasetGenerator, FLAG2DValDatasetGenerator, FLAG2DTestDatasetGenerator

from model.stgcn.st_gcn import STGCN

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig


def main(args: argparse):
    # dataloader
    dataset_train_generator = FLAG2DTrainDatasetGenerator(args.data_path, args.num_frames)
    dataset_val_generator = FLAG2DValDatasetGenerator(args.data_path, args.num_frames)
    dataset_test_generator = FLAG2DTestDatasetGenerator(args.data_path, args.num_frames)
    dataset_train = ds.GeneratorDataset(dataset_train_generator, ["keypoint", "label"], shuffle=True).batch(
        args.batch_size, True)
    dataset_val = ds.GeneratorDataset(dataset_val_generator, ["keypoint", "label"], shuffle=True).batch(args.batch_size,
                                                                                                        True)
    dataset_test = ds.GeneratorDataset(dataset_test_generator, ["keypoint", "label"], shuffle=True).batch(
        args.batch_size, True)

    # model
    model = STGCN(args.in_channels, args.num_frames, args.num_class, args.graph_args, args.edge_importance)

    # loss
    celoss = nn.CrossEntropyLoss()

    # optimizer
    optimizer = nn.SGD(params=model.trainable_params(),learning_rate=args.lr,momentum=args.momentum,
                       weight_decay=args.weight_decay, nesterov=args.nesterov)

    # scheduler

    if args.mode == 'test':
        test_acc = val(dataset_test, model, celoss, optimizer, args)

        return


    if args.mode == 'train':
        best_acc = 0
        for i in range(args.epoch):
            train(dataset_train, model, celoss, optimizer, args)
            val_acc = val(dataset_val, model, celoss, optimizer, args)

            if val_acc>val_acc:
                best_acc = val_acc
                mindspore.save_checkpoint(model, args.logs_path+"/"+"best.ckpt")

        test_acc = val(dataset_val, model, celoss, optimizer, args)

        return


def train(dataset_train, model, celoss, optimizer, args):
    model.set_train()

    for data in dataset_train.create_dict_iterator():
        x = data["keypoint"]
        label = data["label"]
        y = model(x)

        loss = celoss(y,label)
        loss.backward()

def val(dataset_val, model, celoss, optimizer, args):
    pass

def test(dataset_test, model, celoss, optimizer, args):
    pass

if __name__ == "__main__":
    logs_path = "../logs"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    parser = argparse.ArgumentParser(description='stgcn for flag2d')
    # dataset parameter
    parser.add_argument('--data_path', default="D:\\data\\FLAG3D\\data\\flag2d.pkl", type=str,
                        help='where dataset locate')
    parser.add_argument('--logs_path', default=logs_path, type=str, help='where logs and ckpt locate')
    # dataloader parameter
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--in_channels', default=3, type=int, help='Number of channels in the input data')
    parser.add_argument('--num_frames', default=500, type=int, help='Number of frames for the single video')
    parser.add_argument('--num_class', default=60, type=int, help='Number of classes for the classification task')
    parser.add_argument('--graph_args', default=dict(layout='coco', mode='stgcn_spatial'), type=dict,
                        help='The arguments for building the graph')
    parser.add_argument('--edge_importance', default=False, type=bool,
                        help='If ``True``, adds a learnable importance weighting to the edges of the graph')
    # optimizer parameter
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov')
    # training parameter
    parser.add_argument('--epochs', default=30, type=int, help='epochs')
    parser.add_argument('--mode', default="train", type=str, help='train or test')

    args = parser.parse_args()
    # main(args)
