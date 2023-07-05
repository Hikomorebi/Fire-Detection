import argparse
from utils import read_data,evaluate
from my_dataset import MyDataSet
from torchvision import transforms
from resnet50 import MutilScaleViT
import torch
import os
import shutil

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    test_images_path, test_images_label = read_data(args.data_path)
    # 实例化测试数据集
    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.444, 0.385, 0.348], [0.286, 0.275, 0.283])]))
                                   #transforms.Normalize([0.441, 0.351, 0.288], [0.286, 0.261, 0.261])])})
    nw = 4
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=test_dataset.collate_fn)
    model = MutilScaleViT(num_classes=args.num_classes).to(device)

    # 加载已经预训练好的权重参数
    model.load_state_dict(torch.load('/home/hkb/Fire-Detection/MultiScaleViT/models/model-82.pth'))
    test_loss, test_acc, TN, FN, FP, TP, FN_imgs_temp, FP_imgs_temp = evaluate(model=model,
                                                     data_loader=test_loader,
                                                     device=device,
                                                     epoch=0)
    print("Accuracy:\t %.4f"%test_acc)
    print("False Positives:",round(FP/(FP+TN+TP+FN),4))
    print("Fales Negatives:",round(FN/(TP+FN+FP+TN),4))
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    print("Recall:\t\t",round(R,4))
    print("Precision:\t",round(P,4))
    print("F-measure:\t",round(((2*P*R)/(P+R)),4))

    # 存储错误分类的图像
    if not os.path.exists('./error'):
        os.mkdir('./error')
    FN_img_path = './error/FN'
    FP_img_path = './error/FP'
    if not os.path.exists(FN_img_path):
        os.mkdir(FN_img_path)
    if not os.path.exists(FP_img_path):
        os.mkdir(FP_img_path)
    for img_path in FN_imgs_temp:
        shutil.copy(img_path, FN_img_path)
    for img_path in FP_imgs_temp:
        shutil.copy(img_path, FP_img_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data-path', type=str, default="/home/hkb/Fire-Detection/Datasets/RealDatasets")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--batch-size', type=int, default=32)

    opt = parser.parse_args()

    main(opt)
