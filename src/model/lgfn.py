from model import common
import torch
import torch.nn as nn
from model.common import ResBlock
from torch.nn import functional as F
import numpy as np
from Deform_Conv.modules.modulated_deform_conv import ModulatedDeformConv_Sep as DeformConv


def make_model(args):
    return LGFN(args)
    
# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv=common.default_conv, n_feat=64, kernel_size=3, reduction=8, n_resblocks=20):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Local_Fusion(nn.Module):
    def __init__(self, n_frames=7, n_feat=64, conv=common.default_conv):
        super(Local_Fusion, self).__init__()

        self.n_frames = n_frames

        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv1 = nn.Conv2d(n_feat*2, n_feat, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)  

        kernel_size = 3
        res_body = [
            ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=self.act, res_scale=1) for _ in
            range(2)]
        self.res_block = nn.Sequential(*res_body)
        
        
    def neigh_fusion(self, fea1, fea2):

        fea_com = self.act(self.conv1( torch.cat( [fea1,fea2], dim=1 ) ))
        fea_com1 = self.act(self.conv2( torch.cat( [fea_com,fea2], dim=1 ) ))
        fea_fu = self.res_block( fea_com1 )

        return fea_fu + fea2
    
    def forward(self, fea):
        output = []
        lc_fu = fea[0]
        
        for i in range(self.n_frames):
            lc_fu = self.neigh_fusion( lc_fu,fea[i] )
            output.append( lc_fu )

        return output


class Global_Fusion(nn.Module):
    def __init__(self, n_frames=7, n_feat=64):
        super(Global_Fusion, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True) # 0.2 to 0.1

        self.s_num = n_feat // n_frames      # 64 // 7 = 9
        self.in_channel = self.s_num * n_frames      # 9 * 7 = 63
        self.mk_channel = self.in_channel // 2     # 63 // 2 = 31

        self.conv1 = nn.Conv2d(n_feat, self.in_channel, 3, 1, 1, bias=True)

        self.conv2c = nn.Conv2d(self.in_channel, self.mk_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        
        self.conv21 = nn.Conv2d(self.mk_channel, self.mk_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv22 = nn.Conv2d(self.mk_channel, self.mk_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        conv23_body = [
            nn.Conv2d(self.mk_channel, self.mk_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.mk_channel, self.mk_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
            ]
        self.conv23 = nn.Sequential(*conv23_body)
        
        self.conv2m = nn.Conv2d(self.mk_channel*3, self.in_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        
        self.conv3 = nn.Conv2d(self.in_channel, n_feat, 3, 1, 1, bias=True)

    def forward(self, x):
        batch_size, num, c, h, w = x.size()

        x2_list = []
        x4_list = []

        # conv1 
        x1 = self.act(self.conv1(x.view(-1, c, h, w)))
        x1 = x1.view(batch_size, num, self.in_channel, h, w)  

        # shuffle 
        for i in range(num):
            x1_part = x1[:, :, i * self.s_num:(i + 1) * self.s_num, :, :].contiguous()  
            x2_list.append(x1_part.view(batch_size, self.in_channel, h, w))  
        x2 = torch.stack(x2_list, dim=1)  
        x2 = x2.view(-1, self.in_channel, h, w)  

        # conv2
        x2 = self.act( self.conv2c(x2) )
        res21 = self.act(self.conv21(x2))
        res22 = self.act(self.conv22(x2))
        res23 = self.act(self.conv23(x2))
        res2 = self.act(self.conv2m( torch.cat([res21,res22,res23], dim=1) ))
        x3 = res2.view(batch_size, num, self.in_channel, h, w)  
        

        # inverse shuffle 
        for i in range(num):
            x3_part = x3[:, :, i * self.s_num:(i + 1) * self.s_num, :, :].contiguous()  
            x4_list.append(x3_part.view(batch_size, self.in_channel, h, w))  
        x4 = torch.stack(x4_list, dim=1)  
        x4 = x4.view(-1, self.in_channel, h, w)  

        # conv3
        x5 = self.conv3(x4)
        x5 = x5.view(batch_size, num, c, h, w)  

        return x + x5


class Dual_Fusion(nn.Module):
    def __init__(self,n_frames=7, n_feat=64):
        super(Dual_Fusion, self).__init__()

        self.lc_fusion_blocks = nn.ModuleList()
        self.gl_fusion_blocks = nn.ModuleList()

        self.n_frames = n_frames
        
        self.lc_bnum = 10
        self.gl_bnum = 10

        for i in range(self.lc_bnum):
            self.lc_fusion_blocks.append(Local_Fusion(n_frames=n_frames, n_feat=n_feat, conv=common.default_conv))

        for i in range(self.gl_bnum):
            self.gl_fusion_blocks.append(Global_Fusion(n_frames=n_frames, n_feat=n_feat))

        self.merge = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, stride=1, padding=0, dilation=1, bias=True) 
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        

    def forward(self, x):

        lc_fu = x
        gl_fu = torch.stack(x, dim=1)
        output= []

        for i in range(self.lc_bnum):
            lc_fu = self.lc_fusion_blocks[i](lc_fu)

        for i in range(self.gl_bnum):
            gl_fu = self.gl_fusion_blocks[i](gl_fu)

        for i in range(self.n_frames):
            output.append( self.act( self.merge( torch.cat( [lc_fu[i],gl_fu[:,i,:,:,:].contiguous()], dim=1) ) ) )

        return output


class MDCU(nn.Module):
    def __init__(self, n_feat=64, factor=8):
        super(MDCU, self).__init__()
        
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.factor = factor
        
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.d_list = nn.ModuleList()
        for i in range(1,factor+1 ):
            self.d_list.append( nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=i, dilation=i, bias=True) )
   
        self.conv2 = nn.Conv2d(n_feat*factor//2, n_feat, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)

    def forward(self, x):
        output1 = self.act(self.conv1(x))

        d_res = []
        for i in range(self.factor):
            d_res.append( self.act( self.d_list[i]( output1 ) ) )

        combine = torch.cat(d_res, dim=1)
        output2 = self.act(self.conv2(combine))
        output = x + output2

        return output


class CD_Align(nn.Module):

    def __init__(self, bn=4, n_feat=64, groups=8):
        super(CD_Align, self).__init__()    

        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.block_num = bn

        self.offset_conv1 = nn.ModuleList()
        self.offset_conv2 = nn.ModuleList()
        self.dc = nn.ModuleList()

        for i in range( self.block_num ):
            self.offset_conv1.append( nn.Conv2d(n_feat * 2, n_feat, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) )
            self.offset_conv2.append( MDCU( n_feat=n_feat, factor=(self.block_num-i)*2 ) )
            self.dc.append( DeformConv(n_feat, n_feat, 3, stride=1, padding=1, dilation=1, deformable_groups=groups) )

    def forward(self, nbr_fea_l, ref_fea_l):

        tmp_nbr_fea = nbr_fea_l

        for i in range( self.block_num ):
            offset = torch.cat([tmp_nbr_fea, ref_fea_l], dim=1)
            offset = self.act( self.offset_conv1[i]( offset ) )
            offset = self.offset_conv2[i]( offset )
            tmp_nbr_fea = self.dc[i]( tmp_nbr_fea, offset )

        return tmp_nbr_fea


class LGFN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(LGFN, self).__init__()

        self.n_feat = args.n_feats
        self.kernel_size = 3
        self.scale = args.scale[0]
        
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Feature Extraction
        self.feat0 = conv(args.n_colors, self.n_feat, self.kernel_size)

        res_body = [
            ResBlock(conv, self.n_feat, self.kernel_size, bias=True, bn=False, act=self.act, res_scale=1) for _ in
            range(5)]
        res_body.append(conv(self.n_feat, self.n_feat, self.kernel_size))
        self.res_feat = nn.Sequential(*res_body)

        # Deformable Alignment
        self.cd_align = CD_Align(n_feat=self.n_feat, groups=8)

        # Dual Fusion
        self.dual_fusion = Dual_Fusion(n_frames=args.n_frames, n_feat=self.n_feat)

        # Reconstruction
        self.merge = nn.Conv2d(self.n_feat * 7, self.n_feat, 3, 1, 1, bias=True)
        #RCAB
        self.rg = ResidualGroup()
        # Pixel Shuffle
        modules_tail = [common.Upsampler(conv, self.scale, self.n_feat, act=False),
                   conv(self.n_feat, args.n_colors, self.kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def align(self, res_fea, N):
        center = N // 2
        ref_fea_l = res_fea[:, center, :, :, :].clone()
        ref_fea = res_fea[:, center, :, :, :]
        aligned_fea = []

        for i in range(N):
            if i == center:
                aligned_fea.append(ref_fea)
            else:
                nbr_fea_l = res_fea[:, i, :, :, :].clone()
                nbr_fea = self.cd_align(nbr_fea_l, ref_fea_l)
                aligned_fea.append(nbr_fea)

        return aligned_fea

    def forward(self, x):
        B, N, C, H, W = x.size()

        bilinear = F.interpolate(x[:, N // 2, :, :, :], scale_factor=4, mode='bilinear', align_corners=False)

        flames = x.view(-1, C, H, W)
        flames_fea = self.act(self.feat0(flames))

        res_fea = self.res_feat(flames_fea)

        res_fea = res_fea.view(B, N, -1, H, W)

        aligned_feature = self.align(res_fea, N)

        fusion = torch.cat(self.dual_fusion(aligned_feature), dim=1)

        res = self.act(self.merge(fusion))

        res = self.rg(res)

        res = res + res_fea[:, N // 2, :, :, :]
        res = self.tail(res)

        res = res + bilinear

        return res