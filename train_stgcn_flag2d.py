import argparse
import datetime
import os
import sys

import mindspore
from mindspore import ops

from dataset import FLAG2DTrainDatasetGenerator, FLAG2DValDatasetGenerator, FLAG2DTestDatasetGenerator
from logs.logger import Logger

from model.stgcn.st_gcn import STGCN

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig


def main(args: argparse):
    # dataloader
    print("loading train generator ...")
    dataset_train_generator = FLAG2DTrainDatasetGenerator(args.data_path, args.num_frames)
    print("loading val generator ...")
    dataset_val_generator = FLAG2DValDatasetGenerator(args.data_path, args.num_frames)
    print("loading test generator ...")
    dataset_test_generator = FLAG2DTestDatasetGenerator(args.data_path, args.num_frames, args.test_num_clip)
    dataset_train = ds.GeneratorDataset(dataset_train_generator, ["keypoint", "label"], shuffle=True).batch(
        args.batch_size, True)
    dataset_val = ds.GeneratorDataset(dataset_val_generator, ["keypoint", "label"], shuffle=True).batch(args.batch_size,
                                                                                                        True)
    dataset_test = ds.GeneratorDataset(dataset_test_generator, ["keypoint", "label"], shuffle=True).batch(
        args.batch_size, True)

    # model
    model = STGCN(args.in_channels, args.num_frames, args.num_class, args.graph_args, args.edge_importance)
    dataset_train_len = dataset_train_generator.dataset_len

    # loss
    celoss = nn.CrossEntropyLoss()

    # scheduler
    lr_scheduler = nn.cosine_decay_lr(args.min_lr, args.max_lr, args.epoch * args.iter, args.iter, args.epoch)

    # optimizer
    optimizer = nn.SGD(params=model.trainable_params(),learning_rate=lr_scheduler,momentum=args.momentum,
                       weight_decay=args.weight_decay, nesterov=args.nesterov)

    if args.reusme != None :
        param_dict = mindspore.load_checkpoint(args.reusme)
        mindspore.load_param_into_net(model, param_dict)

    if args.mode == 'test':
        test_acc = val(dataset_test, model, celoss, optimizer, args)
        return


    if args.mode == 'train':
        best_acc = 0
        for i in range(args.epoch):
            print(f"Epoch {i}-------------------------------\n")
            train(dataset_train, model, celoss, optimizer, args)
            val_acc = val(dataset_val, model, celoss, optimizer, args)

            if val_acc>val_acc:
                best_acc = val_acc
                mindspore.save_checkpoint(model, args.logs_path+"/"+"best.ckpt")

        test_acc = val(dataset_val, model, celoss, optimizer, args)

        return


def train(dataset_train, model, celoss, optimizer, args):
    i=0 #iteration num
    total_loss = 0
    model.set_train()
    for data in dataset_train.create_dict_iterator():
        if(i>=args.iter):
            break
        x = data["keypoint"]
        label = data["label"]
        y = model(x)

        loss = celoss(y, label)

        (loss, _), grads = ops.value_and_grad((loss, y), None, optimizer.parameters, has_aux=True)
        loss = ops.depend(loss, optimizer(grads))
        total_loss+=loss.item()

        i += 1

    print("train_total_avg_loss: ",total_loss/(args.batch_size*args.iter))

def val(dataset_val, model, celoss, args):
    i = 0  # iteration num
    total_loss = 0
    total_acc_num = 0
    softmax = ops.Softmax()
    model.set_train(False)
    for data in dataset_val.create_dict_iterator():
        x = data["keypoint"]
        label = data["label"]
        y = model(x)
        # 求loss
        loss = celoss(y, label)
        # 求预测值（先做softmax）
        y = softmax(y)
        pred = ops.argmax(y, axis=1)

        total_acc_num += (pred == label).sum().item()
        total_loss += loss.item()

        i += 1

    print("val_total_avg_loss: ", total_loss / (args.batch_size * i), "accuracy: ", total_acc_num/ (args.batch_size * i))

def test(dataset_test, model, celoss, args):
    i = 0  # iteration num
    total_loss = 0
    total_acc_num = 0
    softmax = ops.Softmax()
    model.set_train(False)
    for data in dataset_test.create_dict_iterator():
        x = data["keypoint"]
        label = data["label"]
        y = model(x)
        # 对10个clip求平均再softmax
        y = y.view(args.batch_size, args.test_num_clip, -1)
        y = ops.mean(y, 1, keep_dims=False)

        # 求loss
        loss = celoss(y, label)

        # 求预测值（先做softmax）
        y = softmax(y)
        pred = ops.argmax(y, axis=1)

        total_acc_num += (pred == label).sum().item()
        total_loss += loss.item()

        i += 1

    print("test_total_avg_loss: ", total_loss / (args.batch_size * i), "accuracy: ", total_acc_num / (args.batch_size * i))

if __name__ == "__main__":
    logs_path = "../logs"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    sys.stdout = Logger(logs_path + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+".txt")

    parser = argparse.ArgumentParser(description='stgcn for flag2d')
    # dataset parameter
    parser.add_argument('--data_path', default="../data/FLAG/flag2d.pkl", type=str,
                        help='where dataset locate')
    parser.add_argument('--logs_path', default=logs_path, type=str, help='where logs and ckpt locate')
    parser.add_argument('--reesume', default=None, type=str, help='where trained model locate')
    # dataloader parameter
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--in_channels', default=3, type=int, help='Number of channels in the input data')
    parser.add_argument('--num_frames', default=500, type=int, help='Number of frames for the single video')
    parser.add_argument('--test_num_clip', default=10, type=int, help='Number of num_clip for the test dataset')
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
    # scheduler parameter
    parser.add_argument('--min_lr', default=0., type=float, help='min learning rate')
    parser.add_argument('--max_lr', default=0.1, type=float, help='max learning rate')
    # training parameter
    parser.add_argument('--epochs', default=30, type=int, help='epochs')
    parser.add_argument('--iter', default=100, type=int, help='iteration')
    parser.add_argument('--mode', default="train", type=str, help='train or test')

    args = parser.parse_args()
    main(args)

# cd autodl-tmp/st-gcn-mindspore
# CUDA_VISIBLE_DEVICES=0 python train_stgcn_flag2d.py