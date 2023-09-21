import os   # os库是Python标准库，包含几百个函数,常用路径操作、进程管理、环境参数等几类
import numpy as np
import random
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image  # PIL，全称 Python Imaging Library，是 Python 平台一个功能非常强大而且简单易用的图像处理库。
from torchvision.transforms import InterpolationMode  # pytorch中resize使用到的一个参数
from utils import compute_loader_otsu_mask
import koila

class ImageSet(data.Dataset):
    # 这里自定义了一个dataset类，名为imageset
    # __init__（）用于类的初始化，几乎在任何框架定义类时都避免不了使用它，因为它负责创建类的实例属性并进行赋值等重要操作
    # 在pytorch中，如需自定义Dataset，就需实现__getitem__（）和__len__（）方法
    # __len__（）返回容器中元素的个数
    # __getitem__函数是数据获取函数，在这个函数里你可以写数据怎么读，怎么处理，并且一些数据预处理、数据增强都可以在这里进行。
    def __init__(self, set_path, set_type, transforms_=None):
        self.transform = transforms.Compose(transforms_)

        clean_path_dir = '{}/{}/{}_C'.format(set_path, set_type, set_type)

        self.clean_images_path = []
        self.smats_path = []
        self.original_images_path = []
        self.num_samples = 0

        for dirpath, dnames, fnames in os.walk("{}/{}/{}_A/".format(set_path, set_type, set_type)):
            # os.walk遍历
            for f in fnames:
                if f.endswith(".png"):
                    orig_path = os.path.join(dirpath, f)
                    clean_path = os.path.join(clean_path_dir, f)

                    self.clean_images_path.append(clean_path)
                    self.original_images_path.append(orig_path)
                    self.num_samples += 1

    def __len__(self):
        # 返回dataset大小
        return self.num_samples

    def __getitem__(self, index):
        # 根据index读取图片
        shadow_data = Image.open(self.original_images_path[index])
        clean_data = Image.open(self.clean_images_path[index])
        
        return self.transform(clean_data), self.transform(shadow_data)


class PairedImageSet(data.Dataset):
    # 也定义了一个dataset
    def __init__(self, set_path, set_type, size=(256, 256), use_mask=True, aug=False):
        self.augment = aug

        self.size = size
        self.use_mask = use_mask

        self.to_tensor = transforms.ToTensor()  # to_tensor函数将原格式转化为可被pytorch快速处理的张量
        if size is not None:
            self.resize = transforms.Resize(self.size, interpolation=InterpolationMode.BICUBIC)
            # Bicubic interpolation双三次插值
            # 将图像的大小调整为指定的size，指定为train.py中262行target size
        else:
            self.resize = None
            
        if use_mask:
            # 例如ISTD数据集还有shadow map的图片，训练时可以使用
            smat_path_dir = '{}/{}/{}_B'.format(set_path, set_type, set_type)
            # 它创建一个字符串，该字符串是一个路径，由三部分组成：set_path、set_type和set_type + '_B'。这三部分用斜杠（/）分隔。
            # 例如，如果set_path是'my_directory'，set_type是'test'，那么smat_path_dir将会是'my_directory/test/test_B'。
        
        clean_path_dir = '{}/{}/{}_C'.format(set_path, set_type, set_type)  # _C/B什么的后缀只是为了区分不同角色的data（inp,gt）

        self.gt_images_path = []  # 初始化4个实例变量
        self.masks_path = []
        self.inp_images_path = []  # input
        self.num_samples = 0

        for dirpath, dnames, fnames \
                in os.walk("{}/{}/{}_A/".format(set_path, set_type, set_type)):
            for f in fnames:
                if f.endswith(".zip"):
                    continue
                # 如果当前文件名以“.zip”结尾，那么就跳过当前循环的剩余部分，直接开始下一次循环。也就是说，它会忽略所有.zip文件。
                orig_path = os.path.join(dirpath, f)
                # 如果当前文件名不以“.zip”结尾，那么这行代码会执行。它使用os.path.join()函数将当前目录的路径（dirpath）和当前文件名（f）连接起来，
                # 形成一个完整的文件路径，并将其存储在变量orig_path中。
                if use_mask:
                    smat_path = os.path.join(smat_path_dir, f)
                    self.masks_path.append(smat_path)

                clean_path = os.path.join(clean_path_dir, os.path.splitext(f)[0] + '.jpg')
                # _A应该是input图像文件夹，但有一个input就有一个gt，所以也在clean路径中添加一个，同时注意格式为jpg
                self.gt_images_path.append(clean_path)
                
                self.inp_images_path.append(orig_path)
                self.num_samples += 1

    def __len__(self):
        return self.num_samples

    # def augs(self, gt, mask, inp):
    #     # 数据增强函数
    #     w, h = gt.size
    #     print(w, h)
    #     tl = np.random.randint(0, h - self.size[0])   # 该函数作用是返回一个随机整数
    #     tt = np.random.randint(0, w - self.size[1])
    #
    #     # PyTorch自定义Transform类————使用transforms.functional模块
    #     # crop裁切干嘛？？？？
    #     gt = torchvision.transforms.functional.crop(gt, tt, tl, self.size[0], self.size[1])
    #     mask = torchvision.transforms.functional.crop(mask, tt, tl, self.size[0], self.size[1])
    #     inp = torchvision.transforms.functional.crop(inp, tt, tl, self.size[0], self.size[1])
    #
    #     if random.random() < 0.5:
    #         inp = torchvision.transforms.functional.hflip(inp)  # 水平翻转
    #         gt = torchvision.transforms.functional.hflip(gt)
    #         mask = torchvision.transforms.functional.hflip(mask)
    #     if random.random() < 0.5:
    #         inp = torchvision.transforms.functional.vflip(inp)  # 垂直翻转
    #         gt = torchvision.transforms.functional.vflip(gt)
    #         mask = torchvision.transforms.functional.vflip(mask)
    #     if random.random() < 0.5:
    #         angle = random.choice([90, 180, 270])
    #         inp = torchvision.transforms.functional.rotate(inp, angle)  # 旋转
    #         gt = torchvision.transforms.functional.rotate(gt, angle)
    #         mask = torchvision.transforms.functional.rotate(mask, angle)
    #
    #     return gt, mask, inp

    def __getitem__(self, index):
        # __getitem__方法定义如何根据索引index获取数据
        inp_data = Image.open(self.inp_images_path[index])  # 给定一个索引index,就可以返回对应图像，index就是对应文件是第几个的数字
        # print(inp_data.mode)
        gt_data = Image.open(self.gt_images_path[index])

        if self.use_mask:
            smat_data = Image.open(self.masks_path[index])
        else:
            smat_data = compute_loader_otsu_mask(inp_data, gt_data)

        # if self.augment:
        #     gt_data, smat_data, inp_data = self.augs(gt_data, smat_data, inp_data)
        # else:
        if self.resize is not None:
            gt_data = self.resize(gt_data)
            smat_data = self.resize(smat_data)
            inp_data = self.resize(inp_data)
        # print(inp_data.mode)

        tensor_gt = self.to_tensor(gt_data)
        tensor_msk = self.to_tensor(smat_data)
        tensor_inp = self.to_tensor(inp_data)
        # 从下往上找，确认三个data的路径地址，这样方便自己布置相似文件结构
        # smat_path_dir = path/type/type_B = masks_path = AB_img
        # clean_path_dir = path/type/type_C = gt_images_path = B_img
        # orig_path_dir = path/type/type_A = inp_images_path = A_img
         (tensor_inp, tensor_msk) = koila.lazy(tensor_inp, tensor_msk, batch=0)
        return tensor_gt, tensor_msk, tensor_inp





