import argparse  # argparse是python用于解析命令行参数和选项的标准模块
import os  # os库是Python标准库，包含几百个函数,常用路径操作、进程管理、环境参数等几类。
import torch
from dconv_model import DistillNet
from initializer import weights_init_normal
from ImageLoaders import PairedImageSet
from loss import PerceptualLossModule # , custom_mse_loss
from torch.autograd import Variable
# autograd包是PyTorch中神经网络的核心, 为基于tensor的的所有操作提供自动微分。是一个逐个运行框架, 意味反向传播是根据代码运行的, 且每次的迭代运行都可能不同.
# Variable包裹着Tensor, 支持几乎所有Tensor的操作,并附加额外的属性
# https://zhuanlan.zhihu.com/p/34298983
from torch.nn.functional import interpolate  # 可以實現上采樣
from torch.optim.lr_scheduler import MultiStepLR  # MultiStepLR按需调整学习率，按照设定间隔调整学习率
from torch.utils.data import DataLoader
from torchvision.utils import save_image
# from UNet import UNetTranslator
from utils import analyze_image_pair, compute_shadow_mask_otsu # analyze_image_pair_rgb, analyze_image_pair_lab # compute_shadow_mask, \
import gc

# import wandb
# wandb.init(project="DISTILL-NET-WSRD2-TEST-REPORT")


os.environ['TORCH_HOME'] = "./loaded_models/"
# 这行代码是在设置 PyTorch 的环境变量 TORCH_HOME。TORCH_HOME 是一个环境变量，用于指定 PyTorch 应该在哪个目录下查找预训练模型和其他资源。
# 当你使用 PyTorch 的一些功能（例如，下载预训练模型）时，PyTorch 会将这些资源保存在 TORCH_HOME 指定的目录中。
# ./表示为当前工程目录下"D:\PyCharm\PycharmProjects\WSRD-DNSR\"


if __name__ == '__main__':
    # parse CLI arguments

    # argparse 模块可以让人轻松编写用户友好命令行接
    # 以代码定义它需要的参数，然后argparse将弄清如何从sys.argv(获取运行python文件的时候命令行参数)解析出这些参数
    # argparse 模块还可自动生成用户手册，即下面的help
    parser = argparse.ArgumentParser()
    # 创建解析器，即创建一个argumentparser对象，其中包含将命令行解析成python数据类型所需的全部信息
    parser.add_argument("--model_type", type=int, default=2, help="[0]UNet [else]DistillNet")
    parser.add_argument("--fullres", type=int, default=1, help="[0]inference with hxwx3 [1]fullres inference")
    # 这个参数--fullres是一个命令行参数，它的值可以是0或1，具体含义如下：
    # 当--fullres的值为0时，模型将以hxwx3的方式进行推理。这可能意味着模型将在一个具有特定高度（h）、宽度（w）和3个颜色通道的输入上进行推理。
    # 当--fullres的值为1时，模型将进行全分辨率（full resolution）推理。这可能意味着模型将在输入的原始分辨率上进行推理，而不是在缩小或修改后的分辨率上进行推理。
    # 默认情况下，--fullres的值为1，即默认进行全分辨率推理。
    # 调用add.argument()方法创建参数信息

    parser.add_argument("--n_epochs", type=int, default=15, help="number of epochs of training")
    parser.add_argument("--resume_epoch", type=int, default=0, help="epoch to resume training")  # 重载训练，从之前中断处接着
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument("--decay_epoch", type=int, default=8, help="epoch from which to start lr decay")
    # decay_epoch 就是经过多少个epoch后学习率衰减，这样有助于收敛
    parser.add_argument("--decay_steps", type=int, default=2, help="number of step decays")
    # 衰减速度，相当于iteration数值，训练一个batch就是一个迭代

    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=2048, help="size of image height")
    parser.add_argument("--img_width", type=int, default=2048, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")

    parser.add_argument("--pixelwise_weight", type=float, default=1.0, help="Pixelwise loss weight")
    parser.add_argument("--perceptual_weight", type=float, default=0.1, help="Perceptual loss weight")

    parser.add_argument("--valid_checkpoint", type=int, default=1, help="checkpoint for validation")
    parser.add_argument("--save_checkpoint", type=int, default=3, help="checkpoint for visual inspection")
    parser.add_argument("--model_dir", default="./model",
                        help="Path of destination directory for the trained models")  # 保存训练模型
    parser.add_argument("--image_dir", default="./savepoint_gallery",
                        help="Path for the directory used to save the output test images")
    opt = parser.parse_args()
    print(opt)  # .parse_args()方法把参数提取并放到opt中print

    print('CUDA: ', torch.cuda.is_available(), torch.cuda.device_count())

    os.makedirs(opt.model_dir, exist_ok=True)  # exist_ok只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    criterion_pixelwise = torch.nn.MSELoss()  # 均方误差
    pl = PerceptualLossModule()

    cuda = torch.cuda.is_available()
    if cuda:
        device = "cuda"
    else:
        device = "cpu"

    if opt.model_type == 0:  # 根据上面参数设定0为UNet
        translator = UNetTranslator(in_channels=3, out_channels=3)
        translator.apply(weights_init_normal)
    else:
        translator = DistillNet(num_iblocks=6, num_ops=4)
        # 通常用torch.nn.DataParallel()函数来用多个gpu加速训练

    if cuda:
        print("USING CUDA FOR MODEL TRAINING")
        translator.cuda()
        criterion_pixelwise.cuda()

    optimizer_G = torch.optim.Adam(translator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # adam优化算法；lr学习率控制了权重的更新比例
    decay_step = (opt.n_epochs - opt.decay_epoch) // opt.decay_steps
    # //表示只保留除后结果的整数部分
    milestones = [me for me in range(opt.decay_epoch, opt.n_epochs, decay_step)]  # decay_epoch正好是lr开始衰减的数
    # milestones参数表示学习率更新的起止区间 https://zhuanlan.zhihu.com/p/93624972
    # range（start, stop, step）start起始步，默认为0；stop停止步但不含其本身；step步长
    scheduler = MultiStepLR(optimizer_G, milestones=milestones, gamma=0.2)
    # 有时希望不同的区间采用不同的更新频率，或者是有的区间更新学习率，有的区间不更新学习率，这就需要MultiStepLR实现动态区间长度控制
    # 假设初始lr=0.5，range（200，400，100），则scheduler意为利用优化算法optimizer_G，在区间[0,200]lr=0.5/[200,300]lr=0.5*gamma/[300.400]lr=0.5*gamma^2
    # optimizer是指定使用哪个优化器，scheduler是对优化器的学习率进行调整

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    train_set = PairedImageSet('./dataset', 'train',
                               size=(opt.img_height, opt.img_width), use_mask=False, aug=True)
    validation_set = PairedImageSet('./dataset', 'validation', size=None,
                                    use_mask=False, aug=False)
    # size=none表示不做改变

    dataloader = DataLoader(
        train_set,  # 第一个参数表示dataset
        batch_size=opt.batch_size,
        shuffle=True,  # 用于打乱数据集的顺序
        num_workers=opt.n_cpu,  # 用于数据加载的子进程数
        drop_last=True  # 样本数不能被batchsize整除时是否舍弃最后一批，true舍弃
    )

    val_dataloader = DataLoader(
        validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu
    )

    num_samples = len(dataloader)
    val_samples = len(val_dataloader)

    translator_train_loss = []  # 创建一个名为 translator_train_loss 的空列表,可用来存储一些值
    translator_valid_loss = []

    translator_train_mask_loss = []
    translator_valid_mask_loss = []

    translator_train_pix_loss = []
    translator_valid_pix_loss = []

    translator_train_perc_loss = []
    translator_valid_perc_loss = []

    best_rmse = 1e3

    for epoch in range(opt.resume_epoch, opt.n_epochs):
        # 这个for语句一直贯穿到结束，下面的0应该是一开始初始设置
        # 这个for语句一直执行到结束，算一次循环，再回过头来重新再来下一个epoch，再来下一个epoch这些损失/误差初始均为0
        train_epoch_loss = 0
        train_epoch_pix_loss = 0
        train_epoch_perc_loss = 0
        train_epoch_mask_loss = 0

        valid_epoch_loss = 0
        valid_mask_loss = 0
        valid_perc_loss = 0
        valid_pix_loss = 0

        epoch_err = 0
        epoch_p = 0

        rmse_epoch = 0
        psnr_epoch = 0

        # lab_rmse_epoch = 0
        # lab_psnr_epoch = 0

        # lab_shrmse_epoch = 0
        # lab_shpsnr_epoch = 0

        # lab_frmse_epoch = 0
        # lab_fpsnr_epoch = 0

        translator = translator.cuda()
        translator = translator.train()

        for i, (B_img, AB_mask, A_img) in enumerate(dataloader):
            # i是啥——>https://blog.csdn.net/HiSi_/article/details/108127173
            # dataloader其中设置好batchsize等各种遍历参数，该for语句完成相当于完成一个epoch，即所有batch循环一遍
            # 如果batch size=4，那么在每次循环中，dataloader会返回一个包含4个图像的批次。也就是说，(B_img, AB_mask, A_img)每个元素都会包含4个图像。
            # 对于mask,原始数据集中并没,但class PairedImageSet的__getitem__方法说明,若本身不含mask那么compute_loader_otsu_mask即可
            # 所以说加载dataloader≠train_dataset,还有class PairedImageSet中定义的其他部分
            # B_img, AB_mask, A_img对应pairedimageset中getitem方法最终返回的三个值“return tensor_gt, tensor_msk, tensor_inp”

            inp = Variable(A_img.type(Tensor))  # input
            gt = Variable(B_img.type(Tensor))
            mask = Variable(AB_mask.type(Tensor))
            # print(f"通道数：{mask.shape[1]}")
            # print(f"通道数：{inp.shape[1]}")
            # print(f"通道数：{gt.shape[1]}")

            optimizer_G.zero_grad()
            # gc.collect()
            # torch.cuda.empty_cache()
            out = translator(inp, mask)

            synthetic_mask = compute_shadow_mask_otsu(inp, out.clone().detach())

            mask_loss = criterion_pixelwise(synthetic_mask, mask)
            loss_pixel = criterion_pixelwise(out, gt)
            perceptual_loss = pl.compute_perceptual_loss_v(out.detach(), gt.detach())

            loss_G = opt.pixelwise_weight * loss_pixel + opt.perceptual_weight * perceptual_loss +\
                     opt.mask_weight * mask_loss

            loss_G.backward()
            optimizer_G.step()
            # 在pytorch训练模型时，常会用到的一组函数依次为optimizer.zero_grad()，loss.backward()，optimizer.step()
            # 作用依次为将梯度归零，反向传播计算得到的每个参数的梯度值，通过梯度下降执行参数更新
            # 用了optimizer.step()模型才会更新

            train_epoch_loss += loss_G.detach().item()
            train_epoch_pix_loss += loss_pixel.detach().item()
            train_epoch_perc_loss += perceptual_loss.detach().item()
            train_epoch_mask_loss += mask_loss.detach().item()  # item()取出单元素张量元素值并返回该值,保持其类型不变，精度高多用于损失函数

        translator_train_loss.append(train_epoch_loss)  # 在之前定义的空列表后添加
        translator_train_mask_loss.append(train_epoch_mask_loss)
        translator_train_perc_loss.append(train_epoch_perc_loss)
        translator_train_pix_loss.append(train_epoch_pix_loss)

        # wandb.log({
             # "train_loss": train_epoch_loss / len(train_set),
             # "train_mask": train_epoch_mask_loss / len(train_set),
             # "train_pixelwise": train_epoch_pix_loss / len(train_set),
             # "train_perceptual": train_epoch_perc_loss / len(train_set)  # 记录train_epoch的各种参数
         # })

        scheduler.step()
        # 对lr进行调整，每个epoch结束进行设置，所以说到这完成了一个epoch
        # 以下是对于epoch进程的检验（if语句）

        if (epoch + 1) % opt.valid_checkpoint == 0 or epoch in [0, 1]:
            # Checkpoint是用于描述在每次训练后保存模型参数（权重）的惯例或术语，正如其名表示该阶段正是需要check的点
            # %取除法余数部分 ==判断两对象是否相等
            # 此句含义为：当前的epoch是opt.valid_checkpoint的倍数 OR epoch是0或1，即是需要验证的情况
            # 每隔checkpoint个epoch模型就会进行一次验证。同时为了在训练初期就能得到反馈，也会在第0/1个epoch后立即验证。这样做的好处是可及时发现模型是否在正确方向上进行学习，并及时调整训练策略。
            with torch.no_grad():
                translator = translator.eval()
        # 以上3个行的含义：在每个opt.valid_checkpoint的倍数epoch以及第0个和第1个epoch时，关闭自动梯度计算，并将模型设置为评估模式。这通常用于在训练过程中定期对模型进行验证。

                if (epoch + 1) % opt.save_checkpoint == 0:  # 按照save_checkpoint的默认，每100个epoch保存一下,+1是因为从0开始计数epoch
                    os.makedirs("{}/{}".format(opt.image_dir, epoch + 1))  # os.makedirs用于创建目录make directory
                # 在每个opt.save_checkpoint的倍数epoch时，创建一个新的目录，该目录位于opt.image_dir下，名为当前"epoch+1"的子目录
                # 通常用于在训练过程中定期保存模型生成的图像，以便后续观察分析。

                for idx, (B_img, AB_mask, A_img) in enumerate(val_dataloader):
                    # 此语句开始每个epoch的评估validation阶段
                    inp = Variable(A_img.type(Tensor))
                    gt = Variable(B_img.type(Tensor))
                    mask = Variable(AB_mask.type(Tensor))

                    if epoch > 0:
                        if opt.fullres == 0:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                out = translator(inp, mask)
                        # 含义：如果opt.fullres的值为0，那么使用半精度浮点数（bfloat16）在CUDA设备上执行模型推理，并将结果保存在out中。通常用于加速模型推理并减少内存使用。
                        else:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                b, c, h, w = inp.shape
                                target_size = (960, 1280)  # ？？？这个可以改吗
                                res_inp = interpolate(inp, target_size, mode='bicubic')  # resize
                                res_mask = interpolate(mask, target_size, mode='nearest')

                                dsz_out = translator(res_inp, res_mask)  # 把res_inp, res_mask送进translator中得到无阴影图像
                                out = interpolate(dsz_out, (h, w), mode='bicubic')
                    else:
                        out = inp
                    # 该else语句含义：首先epoch=0（else情况），也就是输入层
                    # 如果 opt.fullres 的值不为0（是1），那么会按照原始分辨率进行推理。
                    # 首先，把输入图像inp和mask调整到目标大小（960x1280），然后把调整后的res_inp和res_mask送进模型中得到无阴影图像dsz_out
                    # 最后再把dsz_out调整回原始的大小，并保存在out中。

                    # 该for语句范围中，以下为evaluate阶段各种损失的计算
                    synthetic_mask = compute_shadow_mask_otsu(inp, out)

                    mask_loss = criterion_pixelwise(synthetic_mask, mask)

                    loss_pixel = criterion_pixelwise(out, gt)
                    perceptual_loss = pl.compute_perceptual_loss_v(out.detach(), gt.detach())
                    # detach()方法用于返回一个新的Tensor，和原来的tensor共享相同的内存空间，但不会被计算图所追踪，也就是说它不会参与反向传播/不会影响到原有计算图

                    loss_G = opt.pixelwise_weight * loss_pixel + opt.perceptual_weight * perceptual_loss

                    valid_epoch_loss += loss_G.detach().item()
                    valid_mask_loss += mask_loss.detach()
                    valid_pix_loss += loss_pixel.detach()
                    valid_perc_loss += perceptual_loss.detach()

                    rmse, psnr = analyze_image_pair_rgb(out.squeeze(0), gt.squeeze(0))

                    # squeeze(0) 对数据的维度进行压缩，主要作用是去掉维数为1的维度，如果out或gt形状是(1, height, width)，那么该方法会将形状变为(height, width)
                    # rmse_lab, psnr_lab = analyze_image_pair_lab(out.squeeze(0), gt.squeeze(0))
                    # 两个无阴影图像之间的误差rmse & psnr
                    # shrmse_lab, shpsnr_lab = analyze_image_pair_lab((out * mask).squeeze(0), (gt * mask).squeeze(0))
                    # # mask区域之间的误差rmse & psnr  sh代表shadow？？？？？？？？
                    # frmse_lab, fpsnr_lab = analyze_image_pair_lab((out * (1 - mask)).squeeze(0), (gt * (1 - mask)).squeeze(0))
                    # # mask区域之外的来给你这误差rmse & psnr     f代表free（对应1-mask）？？？？？？？？？

                    re, _ = analyze_image_pair(out.squeeze(0), gt.squeeze(0))
                    # 在utilis文件中，analyze_image_pair返回两个值“return rmse_loss, psnr”，分别赋予re和_
                    epoch_err += re

                    rmse_epoch += rmse
                    psnr_epoch += psnr

                    # lab_rmse_epoch += rmse_lab
                    # lab_psnr_epoch += psnr_lab

                    # lab_shrmse_epoch += shrmse_lab
                    # lab_shpsnr_epoch += shpsnr_lab

                    # lab_frmse_epoch += frmse_lab
                    # lab_fpsnr_epoch += fpsnr_lab

                    if (epoch + 1) % opt.save_checkpoint == 0:
                        # 检验下一个epoch是否是save point
                        img_synth = out.detach().data
                        img_real = inp.detach().data
                        img_gt = gt.detach().data
                        img_sample = torch.cat((img_real, img_synth, img_gt), dim=-1)
                        save_image(img_sample, "{}/{}/{}_im.png".format(opt.image_dir, epoch + 1, idx + 1))
                        mask_sample = torch.cat((mask, compute_shadow_mask_otsu(inp, out)), dim=-1)
                        save_image(mask_sample, "{}/{}/{}_mask.png".format(opt.image_dir, epoch + 1, idx + 1))

                # wandb.log({  # 记录下该epoch evaluate结果
                #      "valid_loss": valid_epoch_loss / len(validation_set),
                #      "valid_mask": valid_mask_loss / len(validation_set),
                #      "valid_pixelwise": valid_pix_loss / len(validation_set),
                #      "valid_perceptual": valid_perc_loss / len(validation_set)
                # })

                translator_valid_loss.append(valid_epoch_loss)
                translator_valid_mask_loss.append(valid_mask_loss)

                epoch_err /= val_samples  # /=意为左➗右再赋值给左

                rmse_epoch /= val_samples
                # 之前是rmse_epoch += rmse,即每个样本计算出的rmse值都累加到rmse_epoch上
                # 最后再➗样本数量,即计算验证集上的平均RMSE
                psnr_epoch /= val_samples

                # lab_rmse_epoch /= val_samples
                # lab_psnr_epoch /= val_samples

                # lab_shrmse_epoch /= val_samples
                # lab_shpsnr_epoch /= val_samples

                # lab_frmse_epoch /= val_samples
                # lab_fpsnr_epoch /= val_samples

            # wandb.log({
            #     "rmse": lab_rmse_epoch,
            #     "sh_rmse": lab_shrmse_epoch,
            #     "sf_rmse": lab_frmse_epoch,
            #
            # })

            print("EPOCH: {} - GEN: {} | {} - MSK: {} | {} - RMSE {} - PSNR - {}".format(
                                                                                    epoch, train_epoch_loss,
                                                                                    valid_epoch_loss, train_epoch_mask_loss,
                                                                                    valid_mask_loss,
                                                                                    rmse_epoch,  # lab_rmse_epoch,
                                                                                    # lab_shrmse_epoch, lab_frmse_epoch,
                                                                                    psnr_epoch)) # lab_psnr_epoch))
                                                                                    # lab_shpsnr_epoch, lab_fpsnr_epoch))
            if rmse_epoch < best_rmse and epoch > 0:
                best_rmse = rmse_epoch
                print("Saving checkpoint for {}".format(best_rmse))
                torch.save(translator.cpu().state_dict(), "{}/gen_sh2f.pth".format(opt.model_dir))
                torch.save(optimizer_G.state_dict(), "{}/optimizer_ABG.pth".format(opt.model_dir))
