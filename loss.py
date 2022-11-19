"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import torch
from torch import nn
import numpy as np
from utils.pointnet2_util import gather_nd
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class MultiScaleFlowLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred_flows, gt_flow, fps_idxs, **kwargs):
        num_scale = len(pred_flows)
        offset = len(fps_idxs) - num_scale

        gt_flows = [gt_flow]
        for i in range(1, len(fps_idxs)):
            fps_idx = fps_idxs[i]
            sub_gt_flow = gather_nd(gt_flows[0], fps_idx)
            gt_flows.append(sub_gt_flow)

        total_loss = torch.zeros(1).cuda()
        for i in range(num_scale):
            diff_flow = pred_flows[i].permute(0, 2, 1) - gt_flows[i + offset]
            total_loss += self.alpha[i] * torch.norm(diff_flow, dim=2).sum(dim=1).mean()

        return total_loss


class MultiwayFlowLoss(nn.Module):
    def __init__(self, n_view):
        super().__init__()
        self.n_view = n_view

    def forward(self, gt_full_flow, pd_flow_dict, **kwargs):
        loss = []
        for view_i in range(self.n_view):
            for view_j in range(self.n_view):
                if view_i == view_j:
                    continue
                gt_flow = gt_full_flow[:, view_i * self.n_view + view_j]
                pd_flow = pd_flow_dict[(view_i, view_j)]
                loss.append(torch.norm(gt_flow - pd_flow, dim=2).sum(dim=1).mean())
        return sum(loss) / len(loss)


class MultiwayRFlowLoss(MultiwayFlowLoss):
    def __init__(self, n_view):
        super().__init__(n_view)

    def forward(self, gt_full_flow, pd_rflow_dict, **kwargs):
        return super().forward(gt_full_flow, pd_rflow_dict)
        


class MotTransLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, trans_res0: torch.Tensor, trans_res1: torch.Tensor,
                group_sub_idx: torch.Tensor, segm: torch.Tensor, **kwargs):
        trans_res0 = (trans_res0 ** 2).sum(1)
        trans_res1 = (trans_res1 ** 2).sum(1)

        segm_down = torch.gather(segm, dim=-1, index=group_sub_idx)
        motion_mask0 = ((segm_down[:, 0, :].unsqueeze(2) -
                         segm_down[:, 1, :].unsqueeze(1)) == 0).float()
        motion_mask1 = motion_mask0.transpose(1, 2)
        motion_mask0 /= (motion_mask0.sum(2, keepdim=True) + 1e-8)
        motion_mask1 /= (motion_mask1.sum(2, keepdim=True) + 1e-8)

        trans_res0 *= motion_mask0
        trans_res1 *= motion_mask1
        return trans_res0.sum() / motion_mask0.sum() + trans_res1.sum() / motion_mask1.sum()


class MotGroupLoss(nn.Module):
    """
    Supervise the support matrix used for motion segmentation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_support: torch.Tensor, group_sub_idx: torch.Tensor, segm: torch.Tensor,
                view_pair: tuple = None, **kwargs):
        if view_pair is None:
            view_pair = (0, 1)

        vi, vj = view_pair
        segm_down = torch.gather(segm, dim=-1, index=group_sub_idx)
        motion_mask = ((segm_down[:, vi, :].unsqueeze(2) -
                        segm_down[:, vj, :].unsqueeze(1)) == 0).float()

        group_loss = F.binary_cross_entropy_with_logits(pred_support, motion_mask, reduction='mean')
        return group_loss


class FlowConfLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: list, is_pos: torch.Tensor):
        """
        :param logits: ([B, N])
        :param is_pos: (B, N) 0-1 values.
        :return: scalar loss
        """
        is_pos = is_pos.type(logits[0].type())
        is_neg = 1. - is_pos
        c = 2 * is_pos - 1.         # positive is 1, negative is -1
        num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
        num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0

        all_loss = []
        for ind_logits in logits:
            classif_losses = -torch.log(torch.sigmoid(c * ind_logits) + np.finfo(float).eps.item())
            classif_loss_p = torch.sum(classif_losses * is_pos, dim=1)
            classif_loss_n = torch.sum(classif_losses * is_neg, dim=1)
            classif_loss = torch.mean(classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg)
            all_loss.append(classif_loss)

        ind_logits = ind_logits.detach()
        precision = torch.mean(
            torch.sum((ind_logits > 0).type(is_pos.type()) * is_pos, dim=1) /
            torch.sum((ind_logits > 0).type(is_pos.type()), dim=1)
        )
        recall = torch.mean(
            torch.sum((ind_logits > 0).type(is_pos.type()) * is_pos, dim=1) /
            torch.sum(is_pos, dim=1)
        )

        return {
            "loss": sum(all_loss),
            "precision": precision.item(), "recall": recall.item()
        }


class IoULoss(nn.Module):

    def __init__(self, use_softmax=False):
        super().__init__()
        self.use_softmax = use_softmax
    
    @staticmethod
    def batch_hungarian_matching(gt_segm: torch.Tensor, pd_segm: torch.Tensor, iou: bool = True):
        """
        Get the matching based on IoU score of the Confusion Matrix.
            - Restriction: s must be larger/equal to all gt.
        :param gt_segm (B, N), this N should be n_view * n_point, also segmentation should start from 0.
        :param pd_segm (B, N, s), where s should be in the form of scores.
        :param iou: whether the confusion is based on IoU or simple accuracy.
        :return: (B, s, 2), Only the first n_gt_segms are valid mapping from gt to pd.
                 (B, s) gt mask
        """
        assert gt_segm.min() == 0

        n_batch, n_data, s = pd_segm.size()

        n_gt_segms = torch.max(gt_segm, dim=1).values + 1
        gt_segm = torch.eye(s, dtype=pd_segm.dtype, device=pd_segm.device)[gt_segm]

        matching_score = torch.einsum('bng,bnp->bgp', gt_segm, pd_segm)
        if iou:
            union_score = torch.sum(gt_segm, dim=1).unsqueeze(-1) + \
                          torch.sum(pd_segm, dim=1, keepdim=True) - matching_score
            matching_score = matching_score / (union_score + 1e-7)

        matching_idx = torch.ones((n_batch, s, 2), dtype=torch.long)
        valid_idx = torch.zeros((n_batch, s)).float()
        for batch_id, n_gt_segm in enumerate(n_gt_segms):
            assert n_gt_segm <= s
            
            row_ind, col_ind = linear_sum_assignment(matching_score[batch_id, :n_gt_segm, :].cpu().numpy(), maximize=True)
            assert row_ind.size == n_gt_segm
            matching_idx[batch_id, :n_gt_segm, 0] = torch.from_numpy(row_ind)
            matching_idx[batch_id, :n_gt_segm, 1] = torch.from_numpy(col_ind)
            valid_idx[batch_id, :n_gt_segm] = 1

        matching_idx = matching_idx.to(pd_segm.device)
        valid_idx = valid_idx.to(pd_segm.device)

        return matching_idx, gt_segm, valid_idx

    def forward(self, pd_segm: torch.Tensor, segm: torch.Tensor, **kwargs):
        """
        :param segm:    (B, ...), starting from 1.
        :param pd_segm: (B, ..., s)
        :return: (B,) meanIoU
        """
        n_batch = pd_segm.size(0)
        num_classes = pd_segm.size(-1)


        vi, vj = (0, 1)     
                
        gt_segm = segm.reshape(n_batch, -1)     
        pd_segm = pd_segm.reshape(n_batch, -1, num_classes)
        
        if self.use_softmax:
            pd_segm = torch.softmax(pd_segm, dim=-1)

        n_data = gt_segm.size(-1)
        matching_idx, gt_segm, valid_idx = \
            self.batch_hungarian_matching(gt_segm.detach() - 1, pd_segm.detach())

        gt_gathered = torch.gather(gt_segm, dim=-1,
                                   index=matching_idx[..., 0].unsqueeze(1).repeat(1, n_data, 1))
        pd_gathered = torch.gather(pd_segm, dim=-1,
                                   index=matching_idx[..., 1].unsqueeze(1).repeat(1, n_data, 1))

        matching_score = (pd_gathered * gt_gathered ).sum(dim=1)
        union_score = pd_gathered.sum(dim=1) + gt_gathered.sum(dim=1) - matching_score
        iou = matching_score / (union_score + 1e-7)

        matching_mask = (valid_idx > 0.0).float()
        assert not matching_mask.requires_grad
        iou = (iou * matching_mask).sum(-1) / matching_mask.sum(-1)
        iou = torch.mean(iou)

        return -iou


class CombinedLoss(nn.Module):
    def __init__(self, names: list, loss: list, weights: list):
        super().__init__()
        self.names = names
        self.loss = loss
        self.weights = weights
        assert len(names) == len(loss) == len(weights)

    def forward(self, **kwargs):
        loss_dict = {}
        loss_arr = []
        for nm, ls, w in zip(self.names, self.loss, self.weights):
            loss_res = ls(**kwargs)
            this_loss = None
            if isinstance(loss_res, torch.Tensor):
                this_loss = loss_res * w
            elif isinstance(loss_res, dict):
                if "loss" in loss_res.keys():
                    this_loss = loss_res["loss"] * w
                    del loss_res["loss"]
                loss_dict.update(loss_res)
            else:
                raise NotImplementedError
            if this_loss is not None:
                loss_arr.append(this_loss)
                loss_dict[nm] = this_loss.detach().cpu().numpy()
        loss_dict['sum'] = sum(loss_arr)
        return loss_dict



class CubLoss(nn.Module):
    def __init__(self, hypara):
        super(CubLoss, self).__init__()
        self.std = hypara['W']['W_std']

        self.mask_project = torch.FloatTensor([[0,1,1],[0,1,1],[1,0,1],[1,0,1],[1,1,0],[1,1,0]]).cuda().unsqueeze(0).unsqueeze(0).detach()

        self.mask_plane = torch.FloatTensor([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]]).cuda().unsqueeze(0).unsqueeze(0).detach()

        self.cube_normal = torch.FloatTensor([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]).cuda().unsqueeze(0).unsqueeze(0).detach()
        
        self.cube_planes = torch.FloatTensor([[-1,-1,-1],[1,1,1]]).cuda().unsqueeze(0).unsqueeze(0).detach()

    def compute_REC(self, idx_normals_sim_max, assign_matrix,\
                    scale, pc_inver, pc_sample_inver, planes_scaled, mask_project, mask_plane,\
                    batch_size, num_points, num_cuboids):
        planes_scaled = planes_scaled.unsqueeze(1).repeat(1,num_points,1,3,1).reshape(batch_size,num_points,num_cuboids*6,3)
        scale = scale.unsqueeze(1).repeat(1,num_points,1,6).reshape(batch_size,num_points,num_cuboids*6,3)
        pc_project = pc_sample_inver.permute(0,2,1,3).repeat(1,1,1,6).reshape(batch_size,num_points,num_cuboids*6,3) * mask_project + planes_scaled * mask_plane
        pc_project = torch.max(torch.min(pc_project, scale), -scale).view(batch_size, num_points, num_cuboids, 6, 3)  # [B * num_points * (N*6) * 3]
        pc_project = torch.gather(pc_project, dim=3, index = idx_normals_sim_max.unsqueeze(-1).repeat(1,1,1,1,3)).squeeze(3).permute(0,2,1,3)
        diff = ((pc_project - pc_inver) ** 2).sum(-1).permute(0,2,1)
        diff = torch.mean(torch.mean(torch.sum(diff * assign_matrix, -1), 1))

        return diff

    def compute_SPS(self, assign_matrix):
        num_points = assign_matrix.shape[1]
        norm_05 = (assign_matrix.sum(1)/num_points + 0.01).sqrt().mean(1).pow(2)
        norm_05 = torch.mean(norm_05)

        return norm_05

    def compute_EXT(self, assign_matrix, exist,
                    batch_size, num_points, num_cuboids):
        thred = 24
        loss = nn.BCEWithLogitsLoss().cuda()
        gt = (assign_matrix.sum(1) > thred).to(torch.float32).detach()
        exist = torch.clamp(exist, min=1e-7, max=1-1e-7)
        entropy = loss(exist.squeeze(-1), gt)
        

        return entropy

    def compute_KLD(self, mu , log_var):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - F.relu(log_var), dim = 1), dim = 0)
        return kld_loss

    def compute_CST(self, pc_assign_mean, trans):
        diff = torch.norm((pc_assign_mean.detach() - trans), p = 2, dim = -1)
        diff = torch.mean(torch.mean(diff, -1), -1)
        return diff

    def forward(self, pc, segm, out_dict_1, hypara, **kwargs):
        batch_size = out_dict_1['scale'].shape[0]
        num_cuboids = out_dict_1['scale'].shape[1]
        num_points = pc.shape[2]
        pc=pc.reshape(batch_size, num_points, 3)
        segm = segm.reshape(batch_size, num_points).unsqueeze(-1).repeat(1,1,3).float()


        randn_dis = (torch.randn((batch_size,num_points)) * self.std).cuda().detach()

        pc_sample = pc + randn_dis.unsqueeze(-1).repeat(1,1,3) * segm
        
        pc_sample_inver = pc_sample.unsqueeze(1).repeat(1,num_cuboids,1,1) - out_dict_1['pc_assign_mean'].unsqueeze(2).repeat(1,1,num_points,1)
        pc_sample_inver = torch.einsum('abcd,abde->abce', out_dict_1['rotate'].permute(0,1,3,2), pc_sample_inver.permute(0,1,3,2)).permute(0,1,3,2) #B * N * num_points * 3

        planes_scaled = self.cube_planes.repeat(batch_size,num_cuboids,1,1) * out_dict_1['scale'].unsqueeze(2).repeat(1,1,2,1)

        pc_inver = pc.unsqueeze(1).repeat(1,num_cuboids,1,1) - out_dict_1['pc_assign_mean'].unsqueeze(2).repeat(1,1,num_points,1)
        pc_inver = torch.einsum('abcd,abde->abce', out_dict_1['rotate'].permute(0,1,3,2), pc_inver.permute(0,1,3,2)).permute(0,1,3,2) #B * N * num_points * 3

        normals_inver = segm.unsqueeze(1).repeat(1,num_cuboids,1,1)
        normals_inver = torch.einsum('abcd,abde->abce', out_dict_1['rotate'].permute(0,1,3,2), normals_inver.permute(0,1,3,2)).permute(0,1,3,2) #B * N * num_points * 3

        mask_project = self.mask_project.repeat(batch_size,num_points,num_cuboids,1)
        mask_plane = self.mask_plane.repeat(batch_size,num_points,num_cuboids,1)
        cube_normal = self.cube_normal.unsqueeze(2).repeat(batch_size,num_points,num_cuboids,1,1)

        cos = nn.CosineSimilarity(dim=4, eps=1e-4)
        idx_normals_sim_max = torch.max(cos(normals_inver.permute(0,2,1,3).unsqueeze(3).repeat(1,1,1,6,1),cube_normal),dim=-1,keepdim=True)[1]

        loss_ins = 0
        loss_dict = {}

        # Loss REC
        if hypara['W']['W_REC'] != 0:
            REC = self.compute_REC(idx_normals_sim_max, out_dict_1['assign_matrix'],out_dict_1['scale'],\
                                    pc_inver, pc_sample_inver, planes_scaled, mask_project, mask_plane,\
                                    batch_size, num_points, num_cuboids)
            loss_ins = loss_ins + REC * hypara['W']['W_REC'] 
            loss_dict['REC'] = REC.data.detach().item()

        # Loss SPS
        if hypara['W']['W_SPS']  != 0:
            SPS = self.compute_SPS(out_dict_1['assign_matrix'])
            loss_ins = loss_ins + SPS * hypara['W']['W_SPS']
            loss_dict['SPS'] = SPS.data.detach().item()

        # Loss EXT
        if hypara['W']['W_EXT'] != 0:
            EXT = self.compute_EXT(out_dict_1['assign_matrix'], out_dict_1['exist'],
                                    batch_size, num_points, num_cuboids)
            loss_ins = loss_ins + EXT * hypara['W']['W_EXT']
            loss_dict['EXT'] = EXT.data.detach().item()

        # Loss KLD
        if hypara['W']['W_KLD'] != 0:
            KLD = self.compute_KLD(out_dict_1['mu'],out_dict_1['log_var'])
            loss_ins = loss_ins + KLD * hypara['W']['W_KLD'] 
            loss_dict['KLD'] = KLD.data.detach().item()

        # Loss CST
        if hypara['W']['W_CST'] != 0:
            CST = self.compute_CST(out_dict_1['pc_assign_mean'],out_dict_1['trans'])
            loss_ins = loss_ins + CST * hypara['W']['W_CST']
            loss_dict['CST'] = CST.data.detach().item()

        # loss_dict['ALL'] = loss_ins.data.detach().item()
        
        if hypara['W']['W_SPS']  != 0:
            loss_dict['eval'] = (REC * hypara['W']['W_REC']  + SPS * hypara['W']['W_SPS'] ).data.detach().item()
        else:
            loss_dict['eval'] = (REC * hypara['W']['W_REC']  + 0 * hypara['W']['W_SPS'] ).data.detach().item()
        loss_dict['mu'] = torch.mean(torch.mean(out_dict_1['mu'],1),0).data.detach().item()
        loss_dict['var'] = torch.mean(torch.mean(F.relu(out_dict_1['log_var']),1),0).data.detach().item()
       
        return loss_dict
