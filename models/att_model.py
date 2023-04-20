import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import initialize_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


# My Net
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=256, D=512, dropout=False, n_classes=21, att=False):
        super(Attn_Net_Gated, self).__init__()
        self.att = att
        self.attention_a = [nn.Linear(L, D),
                            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)  # W

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)  # 点乘
        A = self.attention_c(A)  # N x n_classes => num_patch × n_classes
        if self.att:
            A = A.mean(dim=1)
        return A, x


class CLAM_SB(nn.Module):
    def __init__(self, size_arg="small", dropout=False, n_classes=21):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [2048, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]  # size = [1024,512,256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        classifiers = [nn.Linear(size[1], 256), nn.ReLU(), nn.Linear(256, n_classes)]
        self.classifiers = nn.Sequential(*classifiers)

        self.n_classes = n_classes

        initialize_weights(self)

    def relocate(self):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h, attention_only=False):

        A, h = self.attention_net(h)  # NxK     A:batch_size × num_patches × 512 , h: 原始输入
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        print(A.shape)
        A = F.softmax(A, dim=0)  # softmax over N
        # c = torch.sum(A, dim=2)
        # A = A.squeeze()
        # h = h.squeeze()
        M = torch.mm(A, h)  # 乘权重
        # M = M.mean(dim=0)
        # print(M.shape)
        logits = self.classifiers(M)

        return logits, A_raw


class CLAM_SB_Reg(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=10,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, freeze=False):
        super(CLAM_SB_Reg, self).__init__()
        self.size_dict = {"small": [2048, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)  # size[1]=512, n_classes=10
        self.reg = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(256, n_classes))
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)
        if freeze:
            self.attention_net.requires_grad_(False)

    def relocate(self, device):
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, attention_only=False):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)
        logits = self.classifiers(M)  # 返回分类层的输出 维度=n_classes
        # logits = self.reg(logits)
        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # Y_prob = F.softmax(logits, dim=1)
        # if instance_eval:
        #     results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
        #                     'inst_preds': np.array(all_preds)}
        # else:
        #     results_dict = {}
        # if return_features:
        #     results_dict.update({'features': M})
        del A, M, h
        gc.collect()
        return logits





