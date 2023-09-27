import argparse
import torch
from dconv_model import DistillNet
from ImageLoaders import PairedImageSet
from loss import PerceptualLossModule
from torch.nn.functional import interpolate  
from torch.optim.lr_scheduler import MultiStepLR  
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import analyze_image_pair, compute_shadow_mask_otsu 
import os  
import gc

os.environ['TORCH_HOME'] = "./loaded_models/"

if __name__ == '__main__':
    # parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullres", type=int, default=1, help="[0]inference with hxwx3 [1]fullres inference")

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
    parser.add_argument("--save_checkpoint", type=int, default=3, help="checkpoint for visual inspection")
    parser.add_argument("--model_dir", default="./model",
                        help="Path of destination directory for the trained models")  # 保存训练模型
    parser.add_argument("--image_dir", default="./savepoint_gallery",
                        help="Path for the directory used to save the output test images")
    opt = parser.parse_args()
    print(opt)  

    print('CUDA: ', torch.cuda.is_available(), torch.cuda.device_count())

    criterion_pixelwise = torch.nn.MSELoss() 
    pl = PerceptualLossModule()

    cuda = torch.cuda.is_available()
    device = "cuda"

    translator = DistillNet(num_iblocks=6, num_ops=4)
      
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
        shuffle=True,  
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

    best_rmse = 1e3

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
            
            from PIL import Image
            from torchvision import transforms
            # 遍历每一批中的每一张图像
                for j in range(B_img.size(0)):
                    # 获取当前图像
                    B_img_j = B_img[j]
                    AB_mask_j = AB_mask[j]
                    A_img_j = A_img[j]

                    # 将图像分割为 16 个 512x512 的块
                    for m in range(4):
                        for n in range(4):
                            left = n * 512
                            upper = m * 512
                            right = (n + 1) * 512
                            lower = (m + 1) * 512

                            gt = transforms.ToTensor(B_img_j.crop((left, upper, right, lower)))
                            mask = transforms.ToTensor(AB_mask_j.crop((left, upper, right, lower)))
                            inp = transforms.ToTensor(A_img_j.crop((left, upper, right, lower)))

                            # 将每个块送入网络模型进行训练,输出结果     
                            optimizer_G.zero_grad()  
                            out = translator(inp, mask)
                            # 模仿源文件，设计一系列loss计算
                            synthetic_mask = compute_shadow_mask_otsu(inp, out.clone().detach())
                            mask_loss = criterion_pixelwise(synthetic_mask, mask)
                            loss_pixel = criterion_pixelwise(out, gt)
                            perceptual_loss = pl.compute_perceptual_loss_v(out.detach(), gt.detach())
                            loss_G = opt.pixelwise_weight * loss_pixel + opt.perceptual_weight * perceptual_loss + opt.mask_weight * mask_loss

                            
                            loss = criterion(output, A_img_chunk)

                            # 计算梯度并更新模型参数
                            loss_G.backward()
                            optimizer_G.step()

            
            inp = A_img.type(Tensor)  # input
            gt = B_img.type(Tensor)
            mask = AB_mask.type(Tensor)
           
            optimizer_G.zero_grad()  
           
            out = translator(inp, mask)
          
            synthetic_mask = compute_shadow_mask_otsu(inp, out.clone().detach())
            mask_loss = criterion_pixelwise(synthetic_mask, mask)
          
            loss_pixel = criterion_pixelwise(out, gt)
         
            perceptual_loss = pl.compute_perceptual_loss_v(out.detach(), gt.detach())
          
            loss_G = opt.pixelwise_weight * loss_pixel + opt.perceptual_weight * perceptual_loss +\
                     opt.mask_weight * mask_loss

            loss_G.backward()
            optimizer_G.step()
         
            train_epoch_loss += loss_G.detach().item()
            train_epoch_pix_loss += loss_pixel.detach().item()
            train_epoch_perc_loss += perceptual_loss.detach().item()
            train_epoch_mask_loss += mask_loss.detach().item()  

        translator_train_loss.append(train_epoch_loss)  
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
      
        if (epoch + 1) % opt.valid_checkpoint == 0 or epoch in [0, 1]:
           with torch.no_grad():
                translator = translator.eval()
    
                if (epoch + 1) % opt.save_checkpoint == 0:  
                    os.makedirs("{}/{}".format(opt.image_dir, epoch + 1))  
               
                for idx, (B_img, AB_mask, A_img) in enumerate(val_dataloader):
                    inp = A_img.type(Tensor)
                    gt = B_img.type(Tensor)
                    mask = AB_mask.type(Tensor)

                    if epoch > 0:
                        if opt.fullres == 0:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                out = translator(inp, mask)
                        else:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                b, c, h, w = inp.shape
                                target_size = (960, 1280)  # ？？？这个可以改吗
                                res_inp = interpolate(inp, target_size, mode='bicubic')  # resize
                                res_mask = interpolate(mask, target_size, mode='nearest')

                                dsz_out = translator(res_inp, res_mask)  
                                out = interpolate(dsz_out, (h, w), mode='bicubic')
                    else:
                        out = inp
                   
                    synthetic_mask = compute_shadow_mask_otsu(inp, out)

                    mask_loss = criterion_pixelwise(synthetic_mask, mask)

                    loss_pixel = criterion_pixelwise(out, gt)
                    perceptual_loss = pl.compute_perceptual_loss_v(out.detach(), gt.detach())

                    loss_G = opt.pixelwise_weight * loss_pixel + opt.perceptual_weight * perceptual_loss

                    valid_epoch_loss += loss_G.detach().item()
                    valid_mask_loss += mask_loss.detach()
                    valid_pix_loss += loss_pixel.detach()
                    valid_perc_loss += perceptual_loss.detach()

                    rmse, psnr = analyze_image_pair_rgb(out.squeeze(0), gt.squeeze(0))

                    re, _ = analyze_image_pair(out.squeeze(0), gt.squeeze(0))
              
                    epoch_err += re

                    rmse_epoch += rmse
                    psnr_epoch += psnr

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
            
                psnr_epoch /= val_samples
               
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
