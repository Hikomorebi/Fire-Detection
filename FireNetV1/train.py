import os
import math
import argparse
import shutil
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms


from my_dataset import MyDataSet
from firenet_model import ConvNet
from utils import read_split_data, read_data, train_one_epoch, evaluate



def main():
    learning_rate = 0.001
    num_classes = 2
    epochs = 200
    batch_size = 32
    IMG_SIZE = 64
    data_path = '/home/hkb/MyFireNet/Datasets/MyDatasets/'
    acc_list_train = []
    acc_list_val = []
    FP_list = []
    FN_list = []
    Recall_list = []
    Precision_list = []
    F_list = []
    best_acc = 0
    index = -1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    train_images_path, train_images_label, val_images_path, val_images_label = read_data(data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(IMG_SIZE),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.441, 0.351, 0.288], [0.286, 0.261, 0.261])]),
        "val": transforms.Compose([transforms.Resize(72),
                                   transforms.CenterCrop(IMG_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.441, 0.351, 0.288], [0.286, 0.261, 0.261])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = ConvNet(num_classes).to(device)
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)


        # validate
        val_loss, val_acc, TN, FN, FP, TP, FN_imgs_temp, FP_imgs_temp = evaluate(model=model,
                                                     data_loader=val_loader,
                                                     device=device,
                                                     epoch=epoch)
        if val_acc > best_acc:
            index = epoch
            best_acc = val_acc
            FN_imgs = FN_imgs_temp
            FP_imgs = FP_imgs_temp
        acc_list_train.append(train_acc)
        acc_list_val.append(val_acc)
        FP_list.append(round(FP/(FP+TN+TP+FN),4))
        FN_list.append(round(FN/(TP+FN+FP+TN),4))
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        Recall_list.append(round(R,4))
        Precision_list.append(round(P,4))
        F_list.append(round(((2*P*R)/(P+R)),4))

        #torch.save(model.state_dict(), "./models/model-{}.pth".format(epoch))
    if not os.path.exists('./results'):
        os.mkdir('./results')
    assert index != -1
    x = [i for i in range(epochs)]
    plt.figure(figsize = (10,12))
    plt.plot(x,acc_list_train,'r',x,acc_list_val,'g')
    plt.title('train(red) val(green)')
    plt.ylabel('accuracy')
    plt.xlabel("epoch")
    plt.savefig('./results/figure.png')


    print(index)
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



if __name__ == '__main__':
    main()
