import argparse
import torch
from dconv_model import DistillNet
from ImageLoaders import PairedImageSet
from loss import PerceptualLossModule
# from torch.nn.functional import interpolate
from torch.optim.lr_scheduler import MultiStepLR  
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import analyze_image_pair, analyze_image_pair_rgb, compute_shadow_mask_otsu
import os  
import gc
from PIL import Image
from torchvision import transforms
import numpy as np

# import wandb
# wandb.init(project="WSRD-myversion-v2")

os.environ['TORCH_HOME'] = "./loaded_models/"

if __name__ == '__main__':
    # parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=int, default=1, help="[0]UNet [1]DistillNet")
    parser.add_argument("--n_epochs", type=int, default=15, help="number of epochs of training")
    parser.add_argument("--resume_epoch", type=int, default=1, help="epoch to resume training")  # 重载训练，从之前中断处接着
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument("--decay_epoch", type=int, default=8, help="epoch from which to start lr decay")
    parser.add_argument("--decay_steps", type=int, default=2, help="number of step decays")

    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=2048, help="size of image height")
    parser.add_argument("--img_width", type=int, default=2048, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")

    parser.add_argument("--pixelwise_weight", type=float, default=1.0, help="Pixelwise loss weight")
    parser.add_argument("--perceptual_weight", type=float, default=0.1, help="Perceptual loss weight")

    parser.add_argument("--valid_checkpoint", type=int, default=1, help="checkpoint for validation")
    parser.add_argument("--save_checkpoint", type=int, default=2, help="checkpoint for visual inspection")
    parser.add_argument("--image_dir", default="./savepoint_gallery",
                        help="Path for the directory used to save the output test images")
    parser.add_argument("--mask_weight", type=float, default=0.05, help="mask loss weight")
    parser.add_argument("--save_interval", type=int, default=1, help="save translator_train/valid_loss/error interval")
    # loss.py中的def compute_perceptual_loss_v(self, synthetic, real):其中三大部分前面的weight也可以更改
    opt = parser.parse_args()

    print('CUDA: ', torch.cuda.is_available(), torch.cuda.device_count())

    criterion_pixelwise = torch.nn.MSELoss() 
    pl = PerceptualLossModule()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.model_type == 0:  # 根据上面参数设定0为UNet
        translator = UNetTranslator(in_channels=3, out_channels=3)
        translator.apply(weights_init_normal)
    else:
        translator = DistillNet(num_iblocks=6, num_ops=4)
        translator = translator.to(device)
      
    print("USING CUDA FOR MODEL TRAINING")
    translator.cuda()
    criterion_pixelwise.cuda()

    optimizer_G = torch.optim.Adam(translator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    decay_step = (opt.n_epochs - opt.decay_epoch) // opt.decay_steps
    milestones = [me for me in range(opt.decay_epoch, opt.n_epochs, decay_step)] 
    scheduler = MultiStepLR(optimizer_G, milestones=milestones, gamma=0.2)
   
    Tensor = torch.cuda.FloatTensor

    train_set = PairedImageSet('./dataset', 'train',
                               size=(opt.img_height, opt.img_width), use_mask=False, aug=True)
    validation_set = PairedImageSet('./dataset', 'validation', size=None,
                                    use_mask=False, aug=False)

    dataloader = DataLoader(
        train_set,  
        batch_size=opt.batch_size,
        shuffle=False,  
        num_workers=opt.n_cpu,  
        drop_last=True  
    )

    val_dataloader = DataLoader(
        validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu
    )
    num_samples = len(dataloader)
    val_samples = len(val_dataloader)
   
    
    translator_train_loss = []  
    translator_valid_loss = []
    translator_train_mask_loss = []
    translator_valid_mask_loss = []
    translator_train_pix_loss = []
    translator_valid_pix_loss = []
    translator_train_perc_loss = []
    translator_valid_perc_loss = []
    translator_valid_error = []

    best_rmse=1e3
    
    for epoch in range(opt.resume_epoch, opt.n_epochs):
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

        translator = translator.cuda()
        translator = translator.train()

        for i, (B_img, AB_mask, A_img) in enumerate(dataloader):  # i其实表示的是第几批batch，从0开始
            B_img = B_img.to(device)
            AB_mask = AB_mask.to(device)
            A_img = A_img.to(device)
            
            # # 遍历每一批中的每一张图像
            # for j in range(B_img.size(0)):
            #     # 获取当前图像
            #     B_img_j = B_img[j]
            #     AB_mask_j = AB_mask[j]
            #     A_img_j = A_img[j]
            #     print("B_img: ", B_img.shape)
            #     print("AB_mask: ", AB_mask.shape)
            #     print("A_img: ", A_img.shape)
                
                # 将图像分割为 16 个 512x512 的块
            for m in range(4):
                for n in range(4):
                    left = n * 512
                    upper = m * 512
                    right = (n + 1) * 512
                    lower = (m + 1) * 512

                    gt = B_img[:, :, upper:lower, left:right]
                    mask = AB_mask[:, :, upper:lower, left:right]
                    inp = A_img[:, :, upper:lower, left:right]
                    
                    # gt = transforms.ToTensor(B_img.crop((left, upper, right, lower)))
                    # mask = transforms.ToTensor(AB_mask.crop((left, upper, right, lower)))
                    # inp = transforms.ToTensor(A_img.crop((left, upper, right, lower)))

                    # 将每个块送入网络模型进行训练,输出结果     
                    optimizer_G.zero_grad()  
                    out = translator(inp, mask)
                        
                    # 模仿源文件，设计一系列loss计算
                    synthetic_mask = compute_shadow_mask_otsu(inp, out.clone().detach())
                    mask_loss = criterion_pixelwise(synthetic_mask, mask)
                    loss_pixel = criterion_pixelwise(out, gt)
                    perceptual_loss = pl.compute_perceptual_loss_v(out.detach(), gt.detach())
                    loss_G = opt.pixelwise_weight * loss_pixel + opt.perceptual_weight * perceptual_loss + opt.mask_weight * mask_loss

                    # 计算/累积梯度
                    loss_G.backward()
                        
                    # 计算每一块的tile_loss之和，遍历所有pic的所有16 tiles
                    train_epoch_loss += loss_G.detach().item()
                    train_epoch_pix_loss += loss_pixel.detach().item()
                    train_epoch_perc_loss += perceptual_loss.detach().item()
                    train_epoch_mask_loss += mask_loss.detach().item()

            # 一个batch后更新模型参数
            optimizer_G.step()   
            # 如果你是在每个小块上计算损失，那么你应该在处理完一个batch的所有小块后，再进行权重的更新。也就是说，你先计算出一个batch中所有小块的损失，然后将这些损失加起来得到整个batch的总损失，最后根据这个总损失来更新权重。
        
        translator_train_loss.append(train_epoch_loss)             
        translator_train_mask_loss.append(train_epoch_mask_loss)   # 每个训练周期的总损失，而不是平均损失，可以帮助我们更好地理解模型在整个训练周期中的表现，而不仅仅是单个样本的表现。
        translator_train_perc_loss.append(train_epoch_perc_loss)   
        translator_train_pix_loss.append(train_epoch_pix_loss)

        if epoch % opt.save_interval == 0:
            np.save(f"./logs/loss/translator_train_loss_epoch{epoch}.npy", np.array(translator_train_loss))
            np.save(f"./logs/loss/translator_train_mask_loss_epoch{epoch}.npy", np.array(translator_train_mask_loss))
            np.save(f"./logs/loss/translator_train_perc_loss_epoch{epoch}.npy", np.array(translator_train_perc_loss))
            np.save(f"./logs/loss/translator_train_pix_loss_epoch{epoch}.npy", np.array(translator_train_pix_loss))
        
        # wandb.log({
             # "train_epoch_loss_avg": train_epoch_loss / len(train_set),
             # "train_epoch_mask_avg": train_epoch_mask_loss / len(train_set),
             # "train_epoch_pixelwise_avg": train_epoch_pix_loss / len(train_set),
             # "train_epoch_perceptual_avg": train_epoch_perc_loss / len(train_set),  
             # "train_epoch_loss": train_epoch_loss,
             # "train_epoch_mask": train_epoch_mask_loss,
             # "train_epoch_pixelwise": train_epoch_pix_loss,
             # "train_epoch_perceptual": train_epoch_perc_loss  
         # })

        scheduler.step()
      
        # if (epoch + 1) % opt.valid_checkpoint == 0 or epoch in [0, 1]:
        if epoch % opt.valid_checkpoint == 0 or epoch in [0, 1]:
            with torch.no_grad():
                translator = translator.eval()

                # if (epoch + 1) % opt.save_checkpoint == 0:
                if epoch % opt.save_checkpoint == 0:
                    os.makedirs("{}/{}".format(opt.image_dir, epoch), exist_ok=True)

                for idx, (B_img, AB_mask, A_img) in enumerate(val_dataloader):
                    B_img = B_img.to(device)
                    AB_mask = AB_mask.to(device)
                    A_img = A_img.to(device)
                    # # 遍历每一批中的每一张图像
                    # for j in range(B_img.size(0)):
                    #     # 获取当前图像
                    #     B_img_j = B_img[j]
                    #     AB_mask_j = AB_mask[j]
                    #     A_img_j = A_img[j]

                    # 将图像分割为 16 个 512x512 的块
                    for m in range(4):
                        for n in range(4):
                            left = n * 512
                            upper = m * 512
                            right = (n + 1) * 512
                            lower = (m + 1) * 512

                            gt = B_img[:, :, upper:lower, left:right]
                            mask = AB_mask[:, :, upper:lower, left:right]
                            inp = A_img[:, :, upper:lower, left:right]
                            # gt = transforms.ToTensor(B_img.crop((left, upper, right, lower)))
                            # mask = transforms.ToTensor(AB_mask.crop((left, upper, right, lower)))
                            # inp = transforms.ToTensor(A_img.crop((left, upper, right, lower)))

                            # 将每个块送入网络模型进行训练,输出结果
                            optimizer_G.zero_grad()
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                out = translator(inp, mask)

                            # if (epoch + 1) % opt.save_checkpoint == 0:
                            if epoch % opt.save_checkpoint == 0:
                                out_img = transforms.ToPILImage()(out[0])
                                inp_img =  transforms.ToPILImage()(inp[0])
                                # A_img_name = A_img.split('.')[0]
                                # 保存图像到文件
                                out_img.save(
                                    "{}/{}/out_{}_{}_{}.png".format(opt.image_dir, epoch, idx, m, n))
                                inp_img.save(
                                    "{}/{}/inp_{}_{}_{}.png".format(opt.image_dir, epoch, idx, m, n))

                                # 接下来就是Poisson image editing的合一部分，当保存了最后一块out时，把之前保存的16个小块进行拼接
                                # if m == 3 and n == 3:

                            # 模仿源文件，设计一系列loss计算
                            synthetic_mask = compute_shadow_mask_otsu(inp, out.clone().detach())
                            mask_loss = criterion_pixelwise(synthetic_mask, mask)
                            loss_pixel = criterion_pixelwise(out, gt)
                            perceptual_loss = pl.compute_perceptual_loss_v(out.detach(), gt.detach())
                            loss_G = opt.pixelwise_weight * loss_pixel + opt.perceptual_weight * perceptual_loss + opt.mask_weight * mask_loss

                            rmse, psnr = analyze_image_pair_rgb(out.squeeze(0), gt.squeeze(0))
                            re, _ = analyze_image_pair(out.squeeze(0), gt.squeeze(0))
                            # 在 analyze_image_pair 函数中，输入的图像数据被假定为在 [0, 1] 范围内，这是因为在许多深度学习框架中，图像数据通常被归一化到这个范围。因此，计算 PSNR 时使用的最大可能像素值是 1。
                            # 而在 analyze_image_pair_rgb 函数中，输入的图像数据被乘以 255，这意味着这个函数假定输入数据在 [0, 255] 范围内，这是一个未归一化的 RGB 图像的典型范围。因此，计算 PSNR 时使用的最大可能像素值是 255。

                            # 计算每一块的tile_loss之和，遍历所有val_pic的所有16 tiles
                            valid_epoch_loss += loss_G.detach().item()
                            valid_mask_loss += mask_loss.detach().item()
                            valid_pix_loss += loss_pixel.detach().item()
                            valid_perc_loss += perceptual_loss.detach().item()

                            epoch_err += re
                            rmse_epoch += rmse
                            psnr_epoch += psnr

            epoch_err /= val_samples    # /=意为左➗右再赋值给左
            rmse_epoch /= val_samples
            psnr_epoch /= val_samples
            
            translator_valid_error.append((epoch_err, rmse_epoch, psnr_epoch))
            translator_valid_loss.append(valid_epoch_loss)
            translator_valid_mask_loss.append(valid_mask_loss)
            translator_valid_pix_loss.append(valid_pix_loss)
            translator_valid_perc_loss.append(valid_perc_loss)

            if epoch % opt.save_interval == 0:
                np.save(f"./logs/loss/translator_valid_loss_epoch{epoch}.npy", np.array(translator_valid_loss))
                np.save(f"./logs/loss/translator_valid_mask_loss_epoch{epoch}.npy", np.array(translator_valid_mask_loss))
                np.save(f"./logs/loss/translator_valid_perc_loss_epoch{epoch}.npy", np.array(translator_valid_perc_loss))
                np.save(f"./logs/loss/translator_valid_pix_loss_epoch{epoch}.npy", np.array(translator_valid_pix_loss))
                np.save(f"./logs/error/translator_valid_error_epoch{epoch}.npy", np.array(translator_valid_error))
            
            # wandb.log({
            #      "valid_epoch_loss_avg": valid_epoch_loss / len(validation_set),
            #      "valid_epoch_mask_avg": valid_mask_loss / len(validation_set),
            #      "valid_epoch_pixelwise_avg": valid_pix_loss / len(validation_set),
            #      "valid_epoch_perceptual_avg": valid_perc_loss / len(validation_set),
            #      "valid_epoch_loss": valid_epoch_loss,
            #      "valid_epoch_mask": valid_mask_loss,
            #      "valid_epoch_pixelwise": valid_pix_loss,
            #      "valid_epoch_perceptual": valid_perc_loss,
            #      "epoch_err_avg":  epoch_err,
            #      "rmse_epoch_avg":  rmse_epoch,
            #      "psnr_epoch_avg":  psnr_epoch
            # })

            print("EPOCH: {} - GEN: {:.3f} | {:.3f} - MSK: {:.3f} | {:.3f} - RMSE {:.3f} - PSNR - {:.3f}".format(
                                                                                        epoch, train_epoch_loss,
                                                                                        valid_epoch_loss, train_epoch_mask_loss,
                                                                                        valid_mask_loss,
                                                                                        rmse_epoch,  # lab_rmse_epoch,
                                                                                        # lab_shrmse_epoch, lab_frmse_epoch,
                                                                                        psnr_epoch)) # lab_psnr_epoch))
                                                                                        # lab_shpsnr_epoch, lab_fpsnr_epoch))
            
            if rmse_epoch < best_rmse and epoch > 1:  # >1是因为第一个epoch模型通常不好，不要保存
                    best_rmse = rmse_epoch
                    print("   \rSaving checkpoint for epoch {} and RMSE {}".format(epoch, best_rmse))
                    torch.save(translator.cpu().state_dict(), "./best_rmse_model/distillnet_epoch{}.pth".format(epoch))
                    torch.save(optimizer_G.state_dict(), "./best_rmse_model/optimizer_epoch{}.pth".format(epoch))
                
            # torch.save(translator.cpu().state_dict(), "{}/gen_sh2f.pth".format(opt.model_dir))：
            # 将最佳模型的参数保存到文件 gen_sh2f.pth 中。这里的 translator.cpu().state_dict() 是一个字典，包含了模型的所有参数。
            # translator.cpu() 是将模型的参数从 GPU 移动到 CPU，这样可以确保无论是否有 GPU，都能加载模型。     
            # torch.save(optimizer_G.state_dict(), "{}/optimizer_ABG.pth".format(opt.model_dir))：
            # 将优化器的状态也保存到文件 optimizer_ABG.pth 中。这样在以后加载模型时，可以恢复优化器的状态，继续训练。

            
            # if epoch == 10 or epoch == opt.n_epochs:    模型保存，我的epoch太少了，之后正式弄的时候再写
            # torch.save(translator.cpu().state_dict(), "./logs/model/distillnet_epoch{}.pth".format(epoch))
            # torch.save(optimizer_G.state_dict(), "./logs/model/optimizer_epoch{}.pth".format(epoch))
            
            if epoch % opt.save_interval == 0:
                with open('./logs/config/hyperparameters.txt', 'w') as f:
                    f.write(str(opt))
                    f.write("\nbest_rmse: {}".format(best_rmse))
            

