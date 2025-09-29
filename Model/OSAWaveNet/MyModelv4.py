import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import Model.OSAWaveNet.MobileNetV2 as MobileNetV2
from Model.OSAWaveNet.ResNet import resnet18,resnet34,resnet50,resnet101,resnet152
import math

# 改进1->上下文特征增强
from einops import rearrange
def pairwise_cos_sim(x1, x2):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim
def OrthogonalPriorGenerator(num_prototypes, feature_dim):    # 生成原型特征矩阵
    '''
    :param num_prototypes: 原型特征数量
    :param feature_dim: 原型特征维度
    :return:相互正交的原型特征矩阵
    '''
    # step 1: 随机初始化原型矩阵 W
    W = torch.randn(num_prototypes, feature_dim) * (2. / feature_dim) ** 0.5  # 使用 Kaiming 初始化
    # Step 2: 初始化临时变量
    beta = [w for w in W[:1]]  # 用于存储原型
    alpha = [w for w in W[1:]]  # 临时存储
    # Step 3: 处理每个 alpha_i  (Gram–Schmidt algorithm)
    for ai in alpha:
        temp = torch.zeros_like(ai)  # 初始化临时结果
        for bj in beta:
            # Step 4: 计算权重
            v = torch.matmul(ai, bj) / torch.matmul(bj, bj)
            temp += v * bj  # 更新临时结果
        beta.append(ai - temp)  # 将更新的结果添加到 beta 中
    # Step 5: 归一化 beta 的 L2 范数
    beta_tensor = torch.stack(beta)
    norm = torch.norm(beta_tensor, p=2, dim=1, keepdim=True)
    beta_normalized = beta_tensor / norm  # 归一化到 L2 范数为 1
    # # --- 验证是否相互正交 --- #
    # print('初始化原型特征:')
    # print(W)
    # print('Gram–Schmidt algorithm处理后原型特征:')
    # print(beta_normalized)
    # print('正交化后特征:')
    # print(torch.matmul(beta_normalized, beta_normalized.transpose(-2, -1)))
    # # --- 验证是否相互正交 --- #
    # Step 6: 返回归一化后的 beta
    return nn.Parameter(beta_normalized.float())
class ContextClusterModule(nn.Module):
    def __init__(self, in_channel, out_channel, proposal_h=2, proposal_w=2, heads=4, head_dim=24):
        """
        :param  in_channel: 输入通道数
        :param out_channel: 输出通道数，PS：在聚类过程中该参数不影响中间结果，仅在特征聚合后，用作context cluster模块的输出特征维度调整
        :param  proposal_w: 每个区域中聚类中心的数量，此处应当求根号值。例如，4个聚类中心，即h\w分别为2
        :param  proposal_h: 同上
        :param       heads: context cluster中注意力头数
        :param    head_dim: context cluster中注意力头的维度
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(in_channel, heads * head_dim, kernel_size=1)      # 将输入映射到相似度空间
        self.v = nn.Conv2d(in_channel, heads * head_dim, kernel_size=1)      # 将输入映射到特征空间
        self.proj = nn.Conv2d(heads * head_dim, out_channel, kernel_size=1)  # for projecting channel number
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        # 该版本的中心相似度不再使用自适应平均池化，而是直接生成正交矩阵向量作为不可学习的相似度矩阵
        self.centers = OrthogonalPriorGenerator(num_prototypes=proposal_h*proposal_w, feature_dim=head_dim)
        self.centers.requires_grad = False
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_h, proposal_w)) #

    def forward(self, x):  # [b,c,w,h]
        # 将特征映射到特征空间
        # 然后将多头特征并入batch维度, 即原本一个batch的样本是当前样本的前heads个
        value = self.v(x)                                                               # [b,c,h,w]->[b,heads*head_dim,h,w]
        value = rearrange(value, "b (e c) h w -> (b e) c h w", e=self.heads)    # [b,heads*head_dim,h,w]->[b*heads,head_dim,h,w]
        # 将输入映射到相似度空间
        # 然后将多头特征并入batch维度, 即原本一个batch的样本是当前样本的前heads个
        x = self.f(x)                                                                   # [b,c,h,w]->[b,heads*head_dim,h,w]
        x = rearrange(x, "b (e c) h w -> (b e) c h w", e=self.heads)            # [b,heads*head_dim,h,w]->[b*heads,head_dim,h,w]
        # 获得映射后空间尺度
        b, c, h, w = x.shape

        # 根据映射后的相似度与特征空间，利用平均池化得到初始簇类中心和初始簇中心特征
        # centers = self.centers_proposal(x)                                                       # [b*heads,head_dim,h_cluster,w_cluster]
        centers = self.centers
        value_centers = rearrange(self.centers_proposal(value), 'b c h w -> b (h w) c')  # [b*heads,num_cluster,head_dim]  num_cluster = h_cluster*w_cluster
        # 获得簇类空间信息
        # b, c, hh, ww = centers.shape
        # b, c, hh, ww = value_centers.shape
        # 计算映射后的输入特征与簇类中心的相似度
        # self.sim_beta、self.sim_alpha分别是可学习的全零、全一矩阵
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                # centers.reshape(b, c, -1).permute(0, 2, 1),
                centers,
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]
        # 根据相似度，将每一个点分配到一个簇类中心
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        # for debug --- 可视化像素分配结果
        assignment_map = sim_max_idx[0].view(h, w)
        assignment_map = assignment_map.detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.imshow(assignment_map, cmap='jet')  # cmap 可以换成你喜欢的
        # plt.show()
        # for debug --- 可视化像素分配结果
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        # 将映射后的输入特征变为点集
        value2 = rearrange(value, 'b c h w -> b (h w) c')  # [b,c,h,w] -> [b,h*w,c] 即 [b*heads,h*w,head_dim]

        # 聚合层
        # value2形如[b,h*w,head_dim]，指的是h*w个像素点特征
        # sim形如[b,num_cluster,h*w]，指的是num_cluster簇类与h*w个像素点的相似度
        # 在计算过程中value会被扩展到[b,num_cluster,h*w,head_dim]代表了为每一个簇保留所有像素点特征
        # 而sim会被扩展到[b,num_cluster,h*w,head_dim]
        # 最后会输出[b,num_cluster,h*w,head_dim]
        # 再经过(value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2)得到簇中心特征(PS:还存在残差与归一化操作)
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (sim.sum(dim=-1, keepdim=True) + 1.0)  # [b,num_cluster,head_dim]

        # dispatch step, return to each point in a cluster
        out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
        out = rearrange(out, "b (h w) c -> b c h w", w=w)
        out = rearrange(out, "(b e) c h w -> b (e c) h w", e=self.heads)
        out = self.proj(out)
        # return out
        return out,assignment_map

# 改进2->频率指导的空间差异感知
import pywt
from torch.autograd import Function
class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None
class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)
class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)   # 保存滤波器权重和输入尺寸，以便 backward 用。
        ctx.shape = x.shape

        dim = x.shape[1]    # 通道数
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)    # 用四组卷积参数，对输入x按通道做分组卷积（grouped conv，每个通道独立做滤波）
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)    # ...
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)    # ...
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)    # ...
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)  # 把四个子带沿 通道维 拼接，输出 [B, 4×C, H/2, W/2]
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors  # 把梯度拆成 4 个子带方向（因为 forward 时是通道拼接过的）
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None
class DWT_2D(nn.Module):
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])   # 低通滤波器系数（低频滤波核）
        dec_lo = torch.Tensor(w.dec_lo[::-1])   # 高通滤波器系数（高频滤波核）PS:[::-1]：滤波器系数翻转（因为 PyTorch 卷积核默认不做反向）

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)    # 构建2D小波滤波器,这是做一维滤波器外积扩展成二维卷积核
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)    # ...
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)    # ...
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)    # ...

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))    # 把这四个 2D 小波核注册为 不可训练的 buffer，这样模型保存或迁移时它们会一起保存，但不会被梯度更新。
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))    # ...
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))    # ...
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))    # ...

        self.w_ll = self.w_ll.to(dtype=torch.float32)   # 这边要和数据类型一直 torch.float32或者torch.float16
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
class LowFrequencyAttention(nn.Module):
    def __init__(self,in_channels,out_channels=None):
        super().__init__()
        if out_channels is None: out_channels = in_channels
        self.pred_norm = nn.InstanceNorm2d(num_features=in_channels, affine=True)
        self.proj_q = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.proj_o = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        self.CBR = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        B, C, H, W = x.shape
        # 自注意力部分
        x_norm = self.pred_norm(x)
        q = self.proj_q(x_norm)
        k = self.proj_k(x_norm)
        v = self.proj_v(x_norm)
        q = q.permute(0, 2, 3, 1).view(B, H*W, C)   # [B,C,H,W] -> [B,H*W,C]
        k = k.view(B, C, H*W)                       # [B,C,H,W] -> [B,C,H*W]
        attn = torch.bmm(q, k) * (int(C) **(-0.5))
        attn = F.softmax(attn, dim=-1)              # [B,H*W,H*W]
        v = v.permute(0, 2, 3, 1).view(B, H*W, C)
        o = torch.bmm(attn, v)                      # [B,H*W,C]
        o = o.view(B, H, W, C).permute(0, 3, 1, 2)
        o = self.proj_o(o)
        o = o + x

        # 特征提取部分
        o = o + self.CBR(o)
        return o
class HighFrequencyAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 方差压缩模块
        if in_channels < reduction: # 输入通道数小于缩减倍率
            reduction = in_channels
        self.compress = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.LayerNorm(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Step 1: 计算每个通道的空间方差
        channel_var = torch.var(x, dim=(2, 3), unbiased=False)  # 方差
        # Step 2: 通过可学习映射生成权重
        attn_weight = self.compress(channel_var)  # [B, C]
        # Step 3: Reshape成广播形式
        attn_weight = attn_weight.view(B, C, 1, 1)  # [B, C, 1, 1]
        # Step 4: 通道加权
        out = x * attn_weight + x
        return out
class FrequencyGuidedSpatialDifferencePerception(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.DWT = DWT_2D(wave='haar')
        self.IDWN = IDWT_2D(wave='haar')
        self.low_freq_attention = LowFrequencyAttention(in_channels=in_channels)     # 已包含残差无需再加
        self.high_freq_attention = HighFrequencyAttention(in_channels=in_channels*3)   # 已包含残差无需再加
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels // 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels*2, 1, bias=False)
        )

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1, stride=1, padding=0),
            nn.ReLU(),
        )


    def forward(self, x1, x2):
        B,C,H,W = x1.shape
        s_d = torch.abs(x1 - x2)        # 空间差异
        # 频域分支
        f_d_DWT = self.DWT.forward(s_d) # 小波变换
        Low,High = f_d_DWT[:,:C,:,:], f_d_DWT[:,C:,:,:] # 高底频拆分
        Low = self.low_freq_attention(Low)    # 已包含残差无需再加
        High = self.high_freq_attention(High) # 已包含残差无需再加
        Low_High = torch.cat([Low,High], dim=1)
        f_d = self.IDWN.forward(Low_High) # 小波逆变换
        # 空间分支
        fea = torch.cat([x1,x2], dim=1)
        diff = torch.cat([s_d,f_d],dim=1)
        aug = self.fusion(fea + diff)
        avgout = self.shared_MLP(self.avg_pool(aug))
        maxout = self.shared_MLP(self.max_pool(aug))
        w = F.sigmoid(avgout + maxout)
        fusion_feature = self.projection(w * fea + (1 - w) * diff)
        return fusion_feature

# from thop import profile
# model = FrequencyGuidedSpatialDifferencePerception(in_channels=3)
# a = torch.rand(3, 3, 256, 256)
# a1 = torch.rand(3, 3, 256, 256)
# pred = model(a,a1)
# print(pred.shape)
# Flops, params = profile(model, inputs=(a,a1))
# print('Flops: % .4fG' % (Flops / 1000000000))  # 计算量
# print('params参数量: % .4fM' % (params / 1000000))  # 参数量





# 原SEIFNet

# 跨阶段差异图合并
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_pool_out = self.avg_pool(x)
        max_out_out = self.max_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        max_out = self.fc2(self.relu1(self.fc1(max_out_out)))
        out = avg_out + max_out
        return self.sigmod(out)

class ACFF2(nn.Module):

    def __init__(self, channel_L, channel_H):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=channel_H,out_channels=channel_L,kernel_size=1, stride=1,padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Conv2d(in_channels=2*channel_L, out_channels=channel_L, kernel_size=1, stride=1, padding=0)
        self.BN = nn.BatchNorm2d(channel_L)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(in_channels=channel_L,ratio=16)

    def forward(self, f_low,f_high):
        f_high = self.relu(self.BN(self.conv1(self.up(f_high))))
        f_cat = f_high + f_low
        adaptive_w = self.ca(f_cat)
        out = f_low * adaptive_w+f_high*(1-adaptive_w) # B,C_l,h,w
        return out

# RM模块，本质上在做跨阶段特征通道对齐

# 上面还有一个ChannelAttention Module
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        self.cbam = CBAM(channel = self.mid_d)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        context = self.cbam(x)
        x_out = self.conv2(context)
        return x_out
#

class MyModel(nn.Module):
    def __init__(self, out_channels=1, pretrained_backbone_path=None):
        super().__init__()

        # 骨干网络
        self.backbone = resnet18(pretrained_path=pretrained_backbone_path)
        self.backbone_stage_dims = [64, 128, 256, 512]  # resnet18

        # 改进1->上下文特征增强
        self.context_cluster12 = ContextClusterModule(in_channel=128,out_channel=128,proposal_h=3,proposal_w=3, heads=4, head_dim=32)
        self.context_cluster13 = ContextClusterModule(in_channel=256, out_channel=256, proposal_h=3, proposal_w=3, heads=4, head_dim=64)
        self.context_cluster14 = ContextClusterModule(in_channel=512, out_channel=512, proposal_h=2, proposal_w=2, heads=4, head_dim=128)
        self.context_cluster22 = ContextClusterModule(in_channel=128, out_channel=128, proposal_h=3, proposal_w=3, heads=4, head_dim=32)
        self.context_cluster23 = ContextClusterModule(in_channel=256, out_channel=256, proposal_h=3, proposal_w=3, heads=4, head_dim=64)
        self.context_cluster24 = ContextClusterModule(in_channel=512, out_channel=512, proposal_h=2, proposal_w=2, heads=4, head_dim=128)


        # 改进2->频率指导的空间差异感知模块
        self.stage_dims = [64, 128, 256, 512]
        self.diff1  = FrequencyGuidedSpatialDifferencePerception(self.stage_dims[0])
        self.diff2  = FrequencyGuidedSpatialDifferencePerception(self.stage_dims[1])
        self.diff3  = FrequencyGuidedSpatialDifferencePerception(self.stage_dims[2])
        self.diff4  = FrequencyGuidedSpatialDifferencePerception(self.stage_dims[3])

        # 原SEIFNet模块
        # 跨阶段差异图融合
        self.ACFF3 = ACFF2(channel_L=self.stage_dims[2], channel_H=self.stage_dims[3])
        self.ACFF2 = ACFF2(channel_L=self.stage_dims[1], channel_H=self.stage_dims[2])
        self.ACFF1 = ACFF2(channel_L=self.stage_dims[0], channel_H=self.stage_dims[1])
        # RM模块本质上在做跨阶段通道对齐 用于上下阶段差异图合并
        self.sam_p4 = SupervisedAttentionModule(self.stage_dims[3])
        self.sam_p3 = SupervisedAttentionModule(self.stage_dims[2])
        self.sam_p2 = SupervisedAttentionModule(self.stage_dims[1])
        self.sam_p1 = SupervisedAttentionModule(self.stage_dims[0])
        # 跨阶段差异图合并
        self.conv4 = nn.Conv2d(512, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_final1 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upsample8 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.conv4 = nn.Conv2d(512, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_final1 = nn.Conv2d(64, out_channels, kernel_size=1)
        #




    def forward(self, x1, x2):
        # 骨干网络特征输出
        x1_list = self.backbone(x1)
        x2_list = self.backbone(x2)
        x11, x12, x13, x14 = x1_list    # [1, 64, 64, 64] -> [1, 128, 32, 32] -> [1, 256, 16, 16] -> [1, 512, 8, 8]
        x21, x22, x23, x24 = x2_list    # [1, 64, 64, 64] -> [1, 128, 32, 32] -> [1, 256, 16, 16] -> [1, 512, 8, 8]

        # 改进1->上下文特征增强
        x12 = self.context_cluster12(x12)[0] + x12
        x13 = self.context_cluster13(x13)[0] + x13
        x14 = self.context_cluster14(x14)[0] + x14
        x22 = self.context_cluster12(x22)[0] + x22
        x23 = self.context_cluster13(x23)[0] + x23
        x24 = self.context_cluster14(x24)[0] + x24
        # for debug --- 用于显示分配结果 --- 正常请使用上面的代码
        # t12,map12 = self.context_cluster12(x12)
        # t13,map13 = self.context_cluster13(x13)
        # t14,map14 = self.context_cluster14(x14)
        # t22,map22 = self.context_cluster12(x22)
        # t23,map23 = self.context_cluster13(x23)
        # t24,map24 = self.context_cluster14(x24)
        # x12 = t12 + x12
        # x13 = t13 + x13
        # x14 = t14 + x14
        # x22 = t22 + x22
        # x23 = t23 + x23
        # x24 = t24 + x24

        # 改进2->频率指导的空间差异感知模块
        d1 = self.diff1(x11, x21) # torch.Size([1,  64, 64, 64])
        d2 = self.diff2(x12, x22) # torch.Size([1, 128, 32, 32])
        d3 = self.diff3(x13, x23) # torch.Size([1, 256, 16, 16])
        d4 = self.diff4(x14, x24) # torch.Size([1, 512,  8,  8])

        # 原SEIFNet
        # 跨阶段差异图融合
        # 从深到浅层的聚合 sam就是论文中的RM模块
        p4 = self.sam_p4(d4)        # torch.Size([1, 512,  8,  8])
        ACFF_43 = self.ACFF3(d3,p4)
        p3 = self.sam_p3(ACFF_43)
        ACFF_32 =self.ACFF2(d2,p3)
        p2 = self.sam_p2(ACFF_32)
        ACFF_21 = self.ACFF1(d1,p2)
        p1 = self.sam_p1(ACFF_21)
        # 多阶段融合特征合并
        from Utlis.Inference_Tools import inference_tools
        it = inference_tools()
        p4_up = self.upsample8(p4)
        p4_up =self.conv4(p4_up)
        p3_up = self.upsample4(p3)
        p3_up = self.conv3(p3_up)
        p2_up = self.upsample2(p2)
        p2_up = self.conv2(p2_up)

        p= p1+p2_up+p3_up+p4_up
        p_up =self.upsample4(p)

        # 特征输出
        output = self.conv_final1(p_up)
        return output.sigmoid()


        # # 差异特征金字塔
        # dfps = self.DFP(diff_responses_list)
        # p4 = self.conv4(F.interpolate(dfps[3],scale_factor=8,mode='bilinear',align_corners=True))
        # p3 = self.conv4(F.interpolate(dfps[2],scale_factor=4,mode='bilinear',align_corners=True))
        # p2 = self.conv4(F.interpolate(dfps[1],scale_factor=2,mode='bilinear',align_corners=True))
        # p1 = dfps[0] +p2 +p3 +p4
        # output = self.conv_final1(F.interpolate(p1,scale_factor=4,mode='bilinear',align_corners=True))
        # return output.sigmoid(),[align_loss1,align_loss2,align_loss3,align_loss4]

if __name__ == "__main__":
    from thop import profile
    model = MyModel(out_channels=1, pretrained_backbone_path=r'/home/yc096/WorkSpace/BCD/BCD_Home/Pretrained_model/resnet18-5c106cde.pth').cuda()
    a = torch.rand(1, 3, 256, 256).cuda()
    a1 = torch.rand(1, 3, 256, 256).cuda()
    Flops, params = profile(model, inputs=(a,a1))
    print('Flops: % .4fG' % (Flops / 1000000000))  # 计算量
    print('params参数量: % .4fM' % (params / 1000000))  # 参数量