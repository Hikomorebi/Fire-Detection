import torch
import torch.nn as nn
import torch.optim as optim
import sys

from preprocess import load_data
from attention_augmented_wide_resnet import Wide_ResNet
from utils import  read_data
import argparse
from tqdm import tqdm
import time
import os
import shutil
from torchvision import transforms
from my_dataset import MyDataSet
from torch.nn.parallel import DataParallel
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
IMG_SIZE = 32

def get_args():
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--data-path", type=str, default='/home/hkb/MyFireNet/Datasets/MyDatasets')
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs, (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16, help="number of batch size, (default, 8)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="learning_rate, (default: 1e-1)")
    parser.add_argument("--depth", type=int, default=28, help="wide-ResNet depth, (default: 28)")
    parser.add_argument("--widen_factor", type=int, default=10, help="wide_ResNet widen factor, (default: 10)")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate, (default: 0.3)")
    parser.add_argument("--load-pretrained", type=bool, default=False)

    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()
    step = 0
    train_loss = 0
    train_acc = 0
    for data, target, img_paths in tqdm(train_loader, desc="epoch " + str(epoch), mininterval=3):
        adjust_learning_rate(optimizer, epoch, args)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        step += 1
        if step % 500 == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc), end='')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))

    length = len(train_loader.dataset) // args.batch_size
    return train_loss / length, train_acc / length


def get_test(model, test_loader):
    model.eval()
    correct = 0
    FN_imgs = []
    FP_imgs = []
    with torch.no_grad():
        for data, target, img_paths in tqdm(test_loader, desc="evaluation", mininterval=2):
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    return acc

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    FP = torch.zeros(1).to(device)
    FN = torch.zeros(1).to(device)
    TP = torch.zeros(1).to(device)
    TN = torch.zeros(1).to(device)
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    FN_imgs = []
    FP_imgs = []
    for step, data in enumerate(data_loader):
        images, labels, img_paths = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        zes=torch.zeros(labels.size(0)).type(torch.LongTensor).to(device)
        ons=torch.ones(labels.size(0)).type(torch.LongTensor).to(device)
        # zz = torch.nonzero(((pred_classes==ons)&(labels.to(device)==zes)) == 1)
        # zz1 = zz.squeeze()

        FN_indices = torch.nonzero(((pred_classes==ons)&(labels.to(device)==zes)) == 1).squeeze(1).tolist()
        FP_indices = torch.nonzero(((pred_classes==zes)&(labels.to(device)==ons)) == 1).squeeze(1).tolist()
        for i in FN_indices:
            FN_imgs.append(img_paths[i])
        for i in FP_indices:
            FP_imgs.append(img_paths[i])
        

        train_correct01 = ((pred_classes==zes)&(labels.to(device)==ons)).sum() #原标签为1，预测为0
        train_correct10 = ((pred_classes==ons)&(labels.to(device)==zes)).sum() #原标签为0，预测为1
        train_correct00 = ((pred_classes==zes)&(labels.to(device)==zes)).sum() #原标签为0，预测为0
        train_correct11 = ((pred_classes==ons)&(labels.to(device)==ons)).sum() #原标签为1，预测为1
        FP += train_correct01
        FN += train_correct10
        TP += train_correct00
        TN += train_correct11

        accu_num += (train_correct00 + train_correct11)
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, TN.item(), FN.item(), FP.item(), TP.item(), FN_imgs, FP_imgs

def main():
    acc_list_val = []
    FP_list = []
    FN_list = []
    Recall_list = []
    Precision_list = []
    F_list = []
    args = get_args()
    index = -1
    best_acc = 0
    train_images_path, train_images_label, val_images_path, val_images_label = read_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(IMG_SIZE),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(72),
                                   transforms.CenterCrop(IMG_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    test_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    num_classes = 2

    # if args.load_pretrained:
    #     model = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes=num_classes, shape=32).to(device)
    #     filename = "best_model_"
    #     checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
    #     model.load_state_dict(checkpoint['model'])
    #     epoch = checkpoint['epoch']
    #     acc = checkpoint['acc']
    #     max_test_acc = acc
    #     print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)
    # else:
    # model = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes=num_classes, shape=IMG_SIZE).to(device)
    model = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes=num_classes, shape=IMG_SIZE)
    model = DataParallel(model).to(device)
    best_acc = 0
    # if device is "cuda":
    #     model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    if not os.path.isdir("reporting"):
        os.mkdir("reporting")

    start_time = time.time()
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, epoch, args)
        val_loss, val_acc, TN, FN, FP, TP, FN_imgs_temp, FP_imgs_temp = evaluate(model=model,data_loader=test_loader,device=device,epoch=epoch)
        if val_acc > best_acc:
            index = epoch
            best_acc = val_acc
            FN_imgs = FN_imgs_temp
            FP_imgs = FP_imgs_temp
        acc_list_val.append(val_acc)
        FP_list.append(round(FP/(FP+TN+TP+FN),4))
        FN_list.append(round(FN/(TP+FN+FP+TN),4))
        P = TP/(TP+FP+0.0001)
        R = TP/(TP+FN+0.0001)
        Recall_list.append(round(R,4))
        Precision_list.append(round(P,4))
        F_list.append(round(((2*P*R)/(P+R+0.0001)),4))
        # if max_test_acc < test_acc:
        #     print('Saving..')
        #     state = {
        #         'model': model.state_dict(),
        #         'acc': test_acc,
        #         'epoch': epoch,
        #     }
        #     if not os.path.isdir('checkpoint'):
        #         os.mkdir('checkpoint')
        #     filename = "best_model_"
        #     torch.save(state, './checkpoint/' + filename + 'ckpt.t7')
        #     max_test_acc = test_acc

        time_interval = time.time() - start_time
        time_split = time.gmtime(time_interval)
        print("Training time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ", time_split.tm_sec)
        print("Test acc:", best_acc, "time: ", time.time() - start_time)
        # with open("./reporting/" + "best_model.txt", "w") as f:
        #     f.write("Epoch: " + str(epoch) + " " + "Best acc: " + str(max_test_acc) + "\n")
        #     f.write("Training time: " + str(time_interval) + "Hour: " + str(time_split.tm_hour) + "Minute: " + str(
        #         time_split.tm_min) + "Second: " + str(time_split.tm_sec))
        #     f.write("\n")
    x = [i for i in range(args.epochs)]
    plt.figure(figsize = (10,12))
    plt.plot(x,acc_list_val,'g')
    plt.title('val(green)')
    plt.ylabel('accuracy')
    plt.xlabel("epoch")
    if not os.path.exists('./results'):
        os.mkdir('./results')
    plt.savefig('./results/figure.png')
    print("index:\t\t",index)
    print("Accuracy:\t %.4f"%acc_list_val[index])
    print("False Positives:",FP_list[index])
    print("Fales Negatives:",FN_list[index])
    print("Recall:\t\t",Recall_list[index])
    print("Precision:\t",Precision_list[index])
    print("F-measure:\t",F_list[index])

    # 存储错误分类的图像
    if not os.path.exists('./error'):
        os.mkdir('./error')
    FN_img_path = './error/FN'
    FP_img_path = './error/FP'
    if not os.path.exists(FN_img_path):
        os.mkdir(FN_img_path)
    if not os.path.exists(FP_img_path):
        os.mkdir(FP_img_path)
    for img_path in FN_imgs:
        shutil.copy(img_path, FN_img_path)
    for img_path in FP_imgs:
        shutil.copy(img_path, FP_img_path)


if __name__ == "__main__":
    main()
