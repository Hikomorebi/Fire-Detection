import os
import sys
import json
import pickle
import random
import torch
from tqdm import tqdm


def read_data(root: str):
    test_path = os.path.join(root,'Test')
    assert os.path.exists(test_path), "test dataset root: {} does not exist.".format(test_path)
    fire_class = [cla for cla in os.listdir(test_path)]
    # 排序，保证各平台顺序一致
    fire_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(fire_class))

    test_images_path = []  # 存储验证集的所有图片路径
    test_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    for cla in fire_class:
        cla_path = os.path.join(test_path, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(test_path, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        for img_path in images:
            test_images_path.append(img_path)
            test_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for test.".format(len(test_images_path)))
    assert len(test_images_path) > 0, "number of test images must greater than 0."
    return test_images_path, test_images_label

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
    data_loader = tqdm(data_loader, file=sys.stdout,mininterval=8)
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

        FN_indices = torch.where(((pred_classes==ons)&(labels.to(device)==zes)) == 1)[0].tolist()
        FP_indices = torch.where(((pred_classes==zes)&(labels.to(device)==ons)) == 1)[0].tolist()
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
