import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
from einops import rearrange
import pdb

class SwapAxes(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.transpose(1, 2)


class Conv1D1(nn.Module):
    def __init__(self):
        super(Conv1D1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=10, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=5, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x  


class Conv1D2(nn.Module):
    def __init__(self):
        super(Conv1D2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=3, kernel_size=10, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=5, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x  


class MLP_robot(nn.Module):
    def __init__(self):
        super(MLP_robot, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x  


class Point_Encoder(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, N_p):
        super().__init__()

        self.N_p = N_p
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(self.N_p, [0.2,0.4], [16, 32], 256+256, [[128, 128, 256], [128, 196, 256]])

    def forward(self, xyz):

        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  #[B, 3, npoint_sa1] --- [B, 320, npoint_sa1]

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  #[B, 3, npoint_sa2] --- [B, 512, npoint_sa2]

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #[B, 3, N_p]        --- [B, 512, N_p]

        return [[l0_xyz, l0_points], [l1_xyz, l1_points], [l2_xyz, l2_points], [l3_xyz, l3_points]]

class Img_Encoder(nn.Module):
    def __init__(self):
        super(Img_Encoder, self).__init__()

        self.model = models.resnet18(weights=None)
        self.model.relu = nn.ReLU()

    def forward(self, img):
        B, _, _, _ = img.size()
        out = self.model.conv1(img)
        out = self.model.relu(self.model.bn1(out))

        out = self.model.maxpool(out) 
        out = self.model.layer1(out)   
        down_1 = self.model.layer2(out)         
        down_2 = self.model.layer3(down_1)       
        down_3 = self.model.layer4(down_2)

        return down_3

class Decoder(nn.Module):
    def __init__(self, additional_channel, emb_dim, N_p, N_raw):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.N_p = N_p
        self.N = N_raw
        
        #upsample
        self.fp3 = PointNetFeaturePropagation(in_channel=512+self.emb_dim, mlp=[768, 512])  
        self.fp2 = PointNetFeaturePropagation(in_channel=832, mlp=[768, 512]) 
        self.fp1 = PointNetFeaturePropagation(in_channel=518+additional_channel, mlp=[512, 512]) 
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.out_head = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 8),
            SwapAxes(),
            nn.BatchNorm1d(self.emb_dim // 8),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(self.emb_dim // 8, 1),
        )


        self.sigmoid = nn.Sigmoid()

    def forward(self, point_feats):

        '''
        encoder_p: [Hierarchy feature]
        '''

        p_0, p_1, p_2, p_3 = point_feats 

        up_sample = self.fp3(p_2[0], p_3[0], p_2[1], p_3[1])
        up_sample = self.fp2(p_1[0], p_2[0], p_1[1], up_sample)
        up_sample = self.fp1(p_0[0], p_1[0], torch.cat([p_0[0], p_0[1]],1), up_sample)

        _3daffordance = self.out_head(rearrange(up_sample, 'b d n -> b n d')) #[B, N, 1]
        _3daffordance = self.sigmoid(_3daffordance)

        return _3daffordance

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv):
        B, N_q, C = q.shape
        _, N_kv, _ = kv.shape

        q = self.q(q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(kv).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class NaviAff(nn.Module):
    def __init__(self, pre_train = True,
                 N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, N_raw = 2048,
                 droprate=0., attention_dim=384):
        super().__init__()

        self.emb_dim = emb_dim
        self.N_p = N_p
        self.N_raw = N_raw
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.additional_channel_floor = 0
        self.additional_channel_global = 1

        self.fusion_attention = CrossAttention(attention_dim, attn_drop=droprate, proj_drop=droprate)
        
        self.point_encoder_global = Point_Encoder(self.emb_dim, True, self.additional_channel_global, self.N_p)
        self.point_encoder_floor = Point_Encoder(self.emb_dim, False, self.additional_channel_floor, self.N_p)

        self.decoder = Decoder(self.additional_channel_floor, self.emb_dim, self.N_p, self.N_raw)

        self.conv1 = Conv1D1()
        self.conv2 = Conv1D2()

        self.mlp_robot = MLP_robot()

      
    def forward(self, global_pc, floor_pc, robot_h_w, floor_mask):
        # pointnet++ process point cloud
        global_f = self.point_encoder_global(global_pc)
        floor_f =self.point_encoder_floor(floor_pc)
        robot_vector = self.mlp_robot(robot_h_w)

        # fusion global and floor
        floor_fuse_point = floor_f[-1][-1].permute(0, 2, 1)
        global_fuse_point = global_f[-1][-1].permute(0, 2, 1)
        additional_info = torch.cat((robot_vector.unsqueeze(1), global_fuse_point), dim=1) 

        fused_feature = self.fusion_attention(floor_fuse_point, additional_info)
        floor_f[-1][-1] = fused_feature.permute(0, 2, 1)

        affor = self.decoder(floor_f) # [B, N, 1]

        return affor