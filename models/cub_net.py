"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: ChengRung Wu <wu840407@gmail.com>
"""


import torch
import torch.nn as nn
import numpy as np
from utils.nn_util import Seq
import torch.nn.functional as F
from utils.pointnet2_util import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule, gather_nd
from utils_pytorch import quat2mat
BN_CONFIG = {"class": "GroupNorm", "num_groups": 4}

def quantize_flow(xyz: torch.Tensor, flow: torch.Tensor):
    dense_flow = xyz[:, 1].unsqueeze(1) - xyz[:, 0].unsqueeze(2)

    dist_mat = torch.cdist(xyz[:, 0] + flow[:, 0], xyz[:, 1])
    dist_mat = -dist_mat / 0.01

    flow01 = (dense_flow * torch.softmax(dist_mat, dim=-1).unsqueeze(-1)).sum(2)
    flow10 = -(dense_flow * torch.softmax(dist_mat, dim=-2).unsqueeze(-1)).sum(1)

    return torch.stack([flow01, flow10], dim=1)

##########################################################################################################################
class Feature_extract(nn.Module):
    def __init__(self, emb_dims, z_dims, k, num_cuboid, low_dim_idx):
        super(Feature_extract, self).__init__()
        self.emb_dims = emb_dims
        self.z_dims = z_dims
        self.k = k
        self.num_cuboid = num_cuboid
        self.low_dim_idx = low_dim_idx
        self.cuboid_vector = torch.eye(self.num_cuboid).float().cuda().detach()
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn3_1 = nn.BatchNorm1d(self.emb_dims)
        self.bn4_1 = nn.BatchNorm1d(self.z_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6 , 64, kernel_size=1, bias=False),
                                   self.bn1_1,
                                   nn.LeakyReLU(negative_slope=0.2, inplace = True),
                                   nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn1_2,
                                   nn.LeakyReLU(negative_slope=0.2, inplace = True))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2_1,
                                   nn.LeakyReLU(negative_slope=0.2, inplace = True),
                                   nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2_2,
                                   nn.LeakyReLU(negative_slope=0.2, inplace = True))
        self.conv3 = nn.Sequential(nn.Conv1d(128, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn3_1,
                                   nn.LeakyReLU(negative_slope=0.2, inplace = True))

        self.fc_mu  = nn.Linear(self.emb_dims, self.z_dims)
        self.fc_var = nn.Linear(self.emb_dims, self.z_dims)

        self.enc_cuboid_vec = nn.Sequential(nn.Conv1d(self.num_cuboid, 64, kernel_size=1, bias=False),
                                            nn.LeakyReLU(negative_slope=0.2, inplace = True))
        self.conv_cuboid = nn.Sequential(nn.Conv1d(self.z_dims + 64, 256, kernel_size=1, bias=False),
                                         nn.LeakyReLU(negative_slope=0.2, inplace = True),
                                         nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                         nn.LeakyReLU(negative_slope=0.2, inplace = True))

    def knn(self, x, k):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   
        return idx

    def get_graph_feature(self, x, k=20, idx=None, dim9=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if dim9 == False:
                idx = self.knn(x, k=k)   
            else:
                idx = self.knn(x[:, 6:], k=k)
        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()   
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.z_dims,1).cuda()
        cuboid_vec = self.cuboid_vector.unsqueeze(0).repeat(num_samples,1,1)            # (batch_size, num_cuboid, num_cuboid)
        cuboid_vec = self.enc_cuboid_vec(cuboid_vec)                                    # (batch_size, 64, num_cuboid)

        x_cuboid = torch.cat((z.repeat(1,1,self.num_cuboid),cuboid_vec),dim=1)          # (batch_size, emb_dims + 64, num_cuboid)
        x_cuboid = self.conv_cuboid(x_cuboid)                                           # (batch_size, 128, num_points)  
        return x_cuboid
    
    def interpolation(self, z1, z2, num_samples):
        delta = (z2 - z1) / (num_samples-1)
        z = torch.zeros(num_samples,self.z_dims).cuda()
        for i in range(num_samples):
            if i == (num_samples - 1):
                z[i,:] = z2
            else:
                z[i,:] = z1 + delta * i
        z = z.unsqueeze(-1)
        cuboid_vec = self.cuboid_vector.unsqueeze(0).repeat(num_samples,1,1)            # (batch_size, num_cuboid, num_cuboid)
        cuboid_vec = self.enc_cuboid_vec(cuboid_vec)                                    # (batch_size, 64, num_cuboid)

        x_cuboid = torch.cat((z.repeat(1,1,self.num_cuboid),cuboid_vec),dim=1)          # (batch_size, emb_dims + 64, num_cuboid)
        x_cuboid = self.conv_cuboid(x_cuboid)                                           # (batch_size, 128, num_points)  
        return x_cuboid

    def forward(self, xyz: torch.Tensor, flow: torch.Tensor):

        batch_size = xyz.size(0)
        x = xyz.transpose(2, 1)

        idx = self.knn(x, k=self.k)
        x = self.get_graph_feature(x, k=self.k, idx = idx if self.low_dim_idx == 1 else None)    # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                                                                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]                                                     # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        #x11=torch.cat((x1, flow.transpose(2, 1)), dim=1)                                         # (batch_size, 64, num_points) -> (batch_size, 67, num_points)
        x = self.get_graph_feature(x1, k=self.k ,idx = idx if self.low_dim_idx == 1 else None)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                                                                        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]                                                     # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x_per = torch.cat((x1, x2), dim=1)                 # (batch_size, 128, num_points)
        x_global = self.conv3(x_per)                       # (batch_size, 128, num_points) -> (batch_size, emb_dims, num_points)
        x_global = x_global.max(dim=-1, keepdim=True)[0]   # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
      
        
        mu = self.fc_mu(x_global.squeeze(-1))
    
        log_var = self.fc_var(x_global.squeeze(-1))
        
        z = self.reparameterize(mu, log_var).unsqueeze(-1)
        
        cuboid_vec = self.cuboid_vector.unsqueeze(0).repeat(batch_size,1,1)           # (batch_size, num_cuboid, num_cuboid)
        cuboid_vec = self.enc_cuboid_vec(cuboid_vec)                                  # (batch_size, 60, num_cuboid)

        x_cuboid = torch.cat((z.repeat(1,1,self.num_cuboid),cuboid_vec),dim=1)        # (batch_size, emb_dims + 60, num_cuboid)
        x_cuboid = self.conv_cuboid(x_cuboid)                                         # (batch_size, 120, num_cuboid)  

        return x_per, x_cuboid, z, mu, log_var

##########################################################################################################################
class Para_pred(nn.Module):
    def __init__(self):
        super(Para_pred, self).__init__()

        self.conv_scale = nn.Conv1d(128, 3, kernel_size=1)
        nn.init.zeros_(self.conv_scale.bias)

        self.conv_rotate = nn.Conv1d(128, 4, kernel_size=1)
        self.conv_rotate.bias.data = torch.Tensor([1, 0, 0, 0])

        self.conv_trans = nn.Conv1d(128, 3, kernel_size=1)
        nn.init.zeros_(self.conv_trans.bias)

        self.conv_ext =   nn.Sequential(nn.Conv1d(128, 30, kernel_size=1, bias=True),
                                        nn.LeakyReLU(negative_slope=0.2, inplace = True),
                                        nn.Conv1d(30, 1, kernel_size=1, bias=True))

    def forward(self, x_cuboid):

        scale = self.conv_scale(x_cuboid).transpose(2, 1)    # (batch_size, num_cuboid, 3)
        scale = torch.sigmoid(scale)                         # (batch_size, num_cuboid, 3)
        
        rotate = self.conv_rotate(x_cuboid).transpose(2, 1)  # (batch_size, num_cuboid, 4)
        rotate = quat2mat(F.normalize(rotate,dim=2,p=2))     # (batch_size, num_cuboid, 3, 3)

        trans = self.conv_trans(x_cuboid).transpose(2, 1)    # (batch_size, num_cuboid, 3)
        trans = torch.tanh(trans)                            # (batch_size, num_cuboid, 3)

        exist = self.conv_ext(x_cuboid).transpose(2, 1)
        return scale, rotate, trans, exist

##########################################################################################################################
class Attention_module(nn.Module):
    def __init__(self, attention_dim):
        super(Attention_module, self).__init__()
        self.attention_dim = attention_dim
        self.conv_Q = nn.Sequential(nn.Conv1d(128, self.attention_dim, kernel_size=1, bias=False))
        self.conv_K = nn.Sequential(nn.Conv1d(128, self.attention_dim, kernel_size=1, bias=False))


    def forward(self, x_per, x_cuboid, rotate, trans):
        
        Q = self.conv_Q(x_per) / (self.attention_dim ** 0.5)
        K = self.conv_K(x_cuboid).transpose(2, 1)


        assign_matrix = F.softmax(torch.matmul(K,Q), dim = 1).transpose(2, 1)  # (batch_size, num_point, num_cuboid*6)
        assign_rotate = torch.einsum('bmd,bdij->bmij', assign_matrix, rotate) # (batch_size, num_point, 3, 3)
        assign_trans = torch.einsum('bmd,bdk->bmk', assign_matrix, trans) # (batch_size, num_point, 3)
        
        
        
        return assign_matrix, assign_rotate, assign_trans

##########################################################################################################################




class MiniPointNets(nn.Module):
    def __init__(self, bn):
        super().__init__()
        self.u_pre_trans = Seq(12).conv2d(16, bn=bn).conv2d(64, bn=bn).conv2d(512, bn=bn)
        self.u_global_trans = Seq(512).conv2d(256, bn=bn).conv2d(256, bn=bn).conv2d(128, bn=bn)
        self.u_post_trans = Seq(512 + 256).conv2d(256, bn=bn).conv2d(64, bn=bn)\
            .conv2d(16, bn=bn).conv2d(1, activation=None)
    def forward(self, xyz, flow, sub_inds, assign_rotate, assign_trans):
        nsample = sub_inds.size(-1)
        
        sub_ind0, sub_ind1 = sub_inds[:, 0, ...], sub_inds[:, 1, ...]
        xyz0_down = gather_nd(xyz[:, 0, ...], sub_ind0)
        flow0_down = gather_nd(flow[:, 0, ...], sub_ind0)
        Rs0 = gather_nd(assign_rotate[:, 0, ...], sub_ind0).reshape(-1, nsample, 3, 3)
        ts0 = gather_nd(assign_trans[:, 0, ...], sub_ind0)

        xyz1_down = gather_nd(xyz[:, 1, ...], sub_ind1)
        flow1_down = gather_nd(flow[:, 1, ...], sub_ind1)
        Rs1 = gather_nd(assign_rotate[:, 1, ...], sub_ind1).reshape(-1, nsample, 3, 3)
        ts1 = gather_nd(assign_trans[:, 1, ...], sub_ind1)

        identity_mat = torch.eye(3, dtype=torch.float32, device=sub_inds.device).reshape(1, 1, 3, 3)

        xyz_ji0 = xyz1_down.unsqueeze(1) - xyz0_down.unsqueeze(2)
        rxji0 = torch.einsum('bnmd,bndk->bnmk', xyz_ji0, Rs0)
        rtsfi0 = torch.einsum('bnmd,bndk->bnmk', flow1_down.unsqueeze(1) + (ts0 + flow0_down).unsqueeze(2),
                              Rs0 + identity_mat)
        res0 = rxji0 - flow1_down.unsqueeze(1) - rtsfi0
        res0 = res0.permute(0, 3, 1, 2).contiguous()

        
        xyz_ji1 = -xyz_ji0.transpose(1, 2)
        rxji1 = torch.einsum('bnmd,bndk->bnmk', xyz_ji1, Rs1)
        rtsfi1 = torch.einsum('bnmd,bndk->bnmk', flow0_down.unsqueeze(1) + (ts1 + flow1_down).unsqueeze(2),
                              Rs1 + identity_mat)
        res1 = rxji1 - flow0_down.unsqueeze(1) - rtsfi1
        res1 = res1.permute(0, 3, 1, 2).contiguous()
        xyz_info = torch.cat([xyz0_down.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, nsample),
                              xyz1_down.transpose(1, 2).unsqueeze(-2).expand(-1, -1, nsample, -1)], dim=1)
        U = torch.cat([res0, res1.transpose(-1, -2), xyz_info], dim=1)
        
        U = self.u_pre_trans(U)
        U_global0, _ = U.max(3, keepdim=True)
        U_global0 = self.u_global_trans(U_global0)
        U_global1, _ = U.max(2, keepdim=True)
        U_global1 = self.u_global_trans(U_global1)
        U = torch.cat([U, U_global0.expand(-1, -1, -1, nsample), U_global1.expand(-1, -1, nsample, -1)], dim=1)
        U = self.u_post_trans(U)
        U = U.squeeze(1)

        return U, res0, res1


class CubNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.low_dim_idx = 0
        self.num_cuboid = 16
        self.k =20
        self.emb_dims = 1024
        self.z_dims = 512
        self.attention_dim = 64

        self.cube_vert = torch.FloatTensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]]).cuda().detach()
        self.cube_face = torch.FloatTensor([[0,2,3],[0,3,1],[0,1,2],[1,3,2],[4,6,7],[4,7,5],[4,5,6],[5,7,6],[0,4,5],[0,5,1],[0,1,4],[1,5,4],\
                                            [2,6,7],[2,7,3],[2,3,6],[3,7,6],[0,4,6],[0,6,2],[0,2,4],[2,6,4],[1,5,7],[1,7,3],[1,3,5],[3,7,5]]).cuda().detach()
                    
        self.Feature_extract = Feature_extract(emb_dims = self.emb_dims,z_dims = self.z_dims, k = self.k , num_cuboid = self.num_cuboid, low_dim_idx = self.low_dim_idx)
        self.Para_pred = Para_pred()
        self.Attention_module = Attention_module(attention_dim = self.attention_dim)
        self.verify_net = MiniPointNets(BN_CONFIG)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state'])

    def forward(self, xyz: torch.Tensor, flow: torch.Tensor, sub_inds: torch.Tensor):
        n_batch, _, n_point, _ = xyz.size()

        x_per, x_cuboid, z, mu, log_var = self.Feature_extract(xyz.reshape(2 * n_batch, n_point, 3),
                                    flow.reshape(2 * n_batch, n_point, 3))
        
        scale, rotate, trans, exist = self.Para_pred(x_cuboid)
        assign_matrix, assign_rotate, assign_trans= self.Attention_module(x_per, x_cuboid, rotate, trans)

        assign_rotate = assign_rotate.reshape(n_batch, 2, n_point, 9)
        assign_trans = assign_trans.reshape(n_batch, 2, n_point, 3)

        group_matrix, res0, res1 = self.verify_net(xyz, flow, sub_inds, assign_rotate, assign_trans)

        pc_assign = xyz.reshape(2 * n_batch, n_point, 3).unsqueeze(2).repeat(1,1,self.num_cuboid,1) * assign_matrix.unsqueeze(-1).repeat(1,1,1,3)
        # directly compute the cuboid center from segmentation branch
        pc_assign_mean = pc_assign.sum(1) / (assign_matrix.sum(1) + 1).unsqueeze(-1).repeat(1,1,3)

        verts_forward = self.cube_vert.unsqueeze(0).unsqueeze(0).repeat(n_batch*2,self.num_cuboid,1,1) * scale.unsqueeze(2).repeat(1,1,8,1)
        verts_forward = torch.einsum('abcd,abde->abce',rotate, verts_forward.permute(0,1,3,2)).permute(0,1,3,2)
        verts_forward = verts_forward + pc_assign_mean.unsqueeze(2).repeat(1,1,8,1)

        # predict the cuboid center
        verts_predict = verts_forward - pc_assign_mean.unsqueeze(2).repeat(1,1,8,1) + trans.unsqueeze(2).repeat(1,1,8,1)

        return group_matrix, res0, res1, {'scale':scale,
                'rotate':rotate,
                'trans':trans,
                'pc_assign_mean':pc_assign_mean,
                'assign_matrix':assign_matrix,
                'verts_forward':verts_forward,
                'verts_predict':verts_predict,
                'cube_face':self.cube_face,
                'x_cuboid':x_cuboid,
                'exist':exist,
                'z':z,
                'mu':mu,
                'log_var':log_var}
