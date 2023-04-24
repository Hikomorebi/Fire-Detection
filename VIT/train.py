import os
import math
import argparse
import shutil
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torchvision import transforms


from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, read_data, train_one_epoch, evaluate


def main(args):
    acc_list_train = []
    acc_list_val = []
    FP_list = []
    FN_list = []
    Recall_list = []
    Precision_list = []
    F_list = []
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    best_acc = 0
    index = -1

    if os.path.exists("./models") is False:
        os.makedirs("./models")


    train_images_path, train_images_label, val_images_path, val_images_label = read_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.444, 0.385, 0.348], [0.286, 0.275, 0.283])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.444, 0.385, 0.348], [0.286, 0.275, 0.283])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
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

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

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
            if os.listdir('./models'):
                for filename in os.listdir('./models'):
                    file_path = os.path.join('./models', filename)
                    os.remove(file_path)
            torch.save(model.state_dict(), "./models/model-{}.pth".format(epoch))
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
    x = [i for i in range(args.epochs)]
    plt.figure(figsize = (10,12))
    plt.plot(x,acc_list_train,'r',x,acc_list_val,'g')
    plt.title('train(red) val(green)')
    plt.ylabel('accuracy')
    plt.xlabel("epoch")
    plt.savefig('./results/figure.png')
    print("index:\t\t",index)
    print("Accuracy:\t %.4f"%acc_list_val[index])
    print("False Positives:",FP_list[index])
    print("Fales Negatives:",FN_list[index])
    print("Recall:\t\t",Recall_list[index])
    print("Precision:\t",Precision_list[index])
    print("F-measure:\t",F_list[index])
    
    if not os.path.exists('./error'):
        os.mkdir('./error')
    # 存储错误分类的图像
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str,default="/home/hkb/Fire-Detection/Datasets/BigDatasets")
    #parser.add_argument('--data-path', type=str,default="/home/hkb/Fire-Detection/Datasets/LightDataset")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/home/hkb/Fire-Detection/VIT/weights/vit_base_patch16_224_in21k.pth',help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
