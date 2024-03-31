import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

def EculideanDistances(a, b):
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    sq_a = a**2
    sq_b = b**2
    sq_a_sum = torch.sum(sq_a, dim=1).unsqueeze(1)
    sq_b_sum = torch.sum(sq_b, dim=1).unsqueeze(0)
    bt = b.t()
    return torch.sqrt(sq_a_sum + sq_b_sum - 2*a.mm(bt))
def Inf_Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy
class MemoryTripletK_reuse(nn.Module):
    def __init__(self, num_classes, DIM):
        super(MemoryTripletK_reuse, self).__init__()
        self.margin = 0.3
        self.noise_rate = 0.75
        self.alpha = 1.0
        self.K = 5
        self.iteration = 0
        self.epoch = 0
        self.start_check_noise_iteration = 0
        self.class_num = num_classes
        self.Weight = torch.ones(DIM,).cuda()

    def forward(self, inputs_col, targets_col, idx_t, inputs_row_t, target_row_t, weight, image_t, model, p_t):
        _, l = torch.sort(target_row_t, descending=True, dim=1)
        target_row_t = l[0:, 0]
        n = inputs_col.size(0)
        W_t = torch.ones(n, ).cuda()  # add
        W_tt = torch.ones(n, ).cuda()

        epsilon = 1e-5

        loss = torch.tensor(0.).cuda()
        mix_loss = torch.tensor(0.).cuda()
        len = 0
        # if self.iteration < self.start_check_noise_iteration:
        #     self.iteration += 1
        #     return loss, W_tt, len, 0.,

        sim_mat_tt = EculideanDistances(inputs_col, inputs_row_t)
        simratio_score_tt = []
        for i in range(n):
            sim_mat_tt[i, idx_t[i]] = 10.  # 相同样本的特征距离为0
        for i in range(n):
            t_label = targets_col[i]
            nln_mask_tt = (target_row_t == t_label)
            nln_sim_all_tt = sim_mat_tt[i][nln_mask_tt]
            k = min(self.K, nln_sim_all_tt.size(0))
            nln_sim_r_tt = torch.narrow(torch.sort(nln_sim_all_tt, descending=False)[0], 0, 0, k)
            nun_mask_tt = (target_row_t != t_label)
            nun_sim_all_tt = sim_mat_tt[i][nun_mask_tt]
            k = min(self.K, nun_sim_all_tt.size(0))
            nun_sim_r_tt = torch.narrow(torch.sort(nun_sim_all_tt, descending=False)[0], 0, 0, k)
            conf_score_tt = (1.0 * torch.sum(nun_sim_r_tt) / torch.sum(nln_sim_r_tt)).item()
            simratio_score_tt.append(conf_score_tt)
            W_tt[i] = conf_score_tt
            W_t[i] = conf_score_tt

        # sort_ranking_score_tt, ind_tgt_tt = torch.sort(torch.tensor(simratio_score_tt), descending=True)
        # _, ind_tgt_tt_low = torch.sort(torch.tensor(simratio_score_tt))
        # top_n_tgt_ind_tt = torch.narrow(ind_tgt_tt, 0, 0, int(self.noise_rate * n))
        # len = torch.mean(torch.tensor(simratio_score_tt)[top_n_tgt_ind_tt])
        # low_n_tgt_ind_tt = torch.narrow(ind_tgt_tt_low, 0, 0, n - int(self.noise_rate * n))
        # W_tt[top_n_tgt_ind_tt] = 1.
        # W_tt[low_n_tgt_ind_tt] = W_tt[low_n_tgt_ind_tt] ** 2 / torch.sum(W_tt[low_n_tgt_ind_tt] ** 2)

        criterion = torch.nn.TripletMarginLoss(margin=self.margin, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None,
                                               reduction='mean') #TripletMarginLoss
        sum_weight = epsilon
        flag = torch.ones(n,)
        for i in range(n):
            t_label = targets_col[i]
            nln_mask_tt = (target_row_t == t_label)
            yuzhi_p = torch.mean(self.Weight[nln_mask_tt])
            if W_t[i] < yuzhi_p:
                flag[i] = 0.
            nln_mask_tt = nln_mask_tt & (self.Weight >= yuzhi_p)
            nln_sim_all_tt = sim_mat_tt[i][nln_mask_tt]
            k_p = min(self.K, nln_sim_all_tt.size(0))
            nln_sim_r_tt_idx = torch.narrow(torch.sort(nln_sim_all_tt, descending=True)[1], 0, 0, k_p) # True
            nun_mask_tt = (target_row_t != t_label)
            yuzhi_n = torch.mean(self.Weight[nun_mask_tt])
            nun_mask_tt = nun_mask_tt & (self.Weight >= yuzhi_n)
            nun_sim_all_tt = sim_mat_tt[i][nun_mask_tt]
            k_n = min(self.K, nun_sim_all_tt.size(0))
            nun_sim_r_tt_idx = torch.narrow(torch.sort(nun_sim_all_tt, descending=False)[1], 0, 0, k_n) # False
            if k_p == k_n == self.K:
                anchor = inputs_col[i].expand(k_p, inputs_col[i].size(0))
                loss = loss + criterion(anchor, (inputs_row_t[nln_mask_tt])[nln_sim_r_tt_idx],
                                        (inputs_row_t[nun_mask_tt])[nun_sim_r_tt_idx]) * weight[i]
                sum_weight = sum_weight + weight[i].item()
        idx_mix = (flag == 1)
        self.iteration += 1
        loss = loss / sum_weight
        self.Weight[idx_t] = W_t
        # reuse
        image_t = image_t[idx_mix]
        if image_t.size(0) <= 1:
            # res_lu = torch.tensor(0.).cuda()
            # res_hu = torch.tensor(0.).cuda()
            return loss, mix_loss, W_tt
        #信息熵检验
        # res_lu = Inf_Entropy(p_t[idx_mix])
        # res_hu = Inf_Entropy(p_t[~idx_mix])
        # res_hu = torch.mean(res_hu)
        # res_lu = torch.mean(res_lu)
        np.random.seed(seed=1)
        np.random.RandomState(seed=1)
        len_mix = image_t.size(0)
        t_batch = targets_col[idx_mix]
        lam = (torch.from_numpy(np.random.beta(self.alpha, self.alpha, [len_mix]))).float().cuda()#len(image_t)
        t_batch = torch.eye(self.class_num)[t_batch].cuda()
        shuffle_idx = torch.from_numpy(np.random.permutation(len_mix).astype('int64'))
        mixed_x = (lam * image_t.permute(1, 2, 3, 0) + (1 - lam) * image_t[shuffle_idx].permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
        mixed_t = (lam * t_batch.permute(1, 0) + (1 - lam) * t_batch[shuffle_idx].permute(1, 0)).permute(1, 0)
        mixed_x, mixed_t = map(torch.autograd.Variable, (mixed_x, mixed_t))
        mixed_outputs, _ = model(mixed_x)
        softmax = nn.Softmax(dim=1)(mixed_outputs)
        re_loss = (- mixed_t * torch.log(softmax + epsilon)).sum(dim=1)
        re_loss = re_loss.mean(dim=0)
        return loss, re_loss, W_tt

