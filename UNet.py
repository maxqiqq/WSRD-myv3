import torch.nn as nn
import torch.nn.functional as F
import torch


class UNetCompress(nn.Module):
    def __init__(self, in_size, out_size, normalize=True,  kernel_size=4, dropout=0.0):
        # torch.nn.dropout：在训练期间，使用伯努利分布中的样本以概率 p 随机归零输入张量的某些元素。每个通道将在每次转发呼叫时独立归零。
        super(UNetCompress, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        # 上两行就是encoder里面的步骤，那和encoder block的区是啥？???????
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)  # self.model = nn.Sequential(*layers) 表示将网络的层按照顺序组合成一个网络模型

    def forward(self, x):
        return self.model(x)


class UNetDecompress(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetDecompress, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        # 上三行就是decoder里面的步骤，那和它的区是啥？???????
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        f = self.model(x)
        o = torch.cat((f, skip_input), 1)  # concat信息融合
        return o


class UNetTranslator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, local=0, residual=True):
        self.local_transform = local
        super(UNetTranslator, self).__init__()

        self.res = residual

        self.down1 = UNetCompress(in_channels, 64, kernel_size=4, normalize=False)
        self.down2 = UNetCompress(64, 128)
        self.down3 = UNetCompress(128, 256)
        self.down4 = UNetCompress(256, 512, dropout=0.25)
        self.down5 = UNetCompress(512, 512, dropout=0.25)
        self.down6 = UNetCompress(512, 512, dropout=0.5)
        self.down7 = UNetCompress(512, 512, dropout=0.5)
        self.down8 = UNetCompress(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetDecompress(512, 512, dropout=0.5)
        self.up2 = UNetDecompress(1024, 512, dropout=0.5)
        self.up3 = UNetDecompress(1024, 512, dropout=0.25)
        self.up4 = UNetDecompress(1024, 512, dropout=0.25)
        self.up5 = UNetDecompress(1024, 256)
        self.up6 = UNetDecompress(512, 128)
        self.up7 = UNetDecompress(256, 64)
        # down有8层，up有7层？？？？？？

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 采用upsample插值的方法，scale_factor意为空间大小的乘数
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1)
        )

    def forward(self, x, mask):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        result = self.final(u7)

        if self.res:
            if x.shape[1] == 4:
                return mask * F.sigmoid(result) + (1 - mask) * x[:, 1:, :, :]
            else:
                return torch.clamp(torch.tanh(result + x), min=0, max=1)
                # torch.clamp(input, min, max, out=None)将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。
        else:
            return F.sigmoid(result)

