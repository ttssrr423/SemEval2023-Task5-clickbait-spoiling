import logging
import re
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
logger = logging.getLogger(__name__)

class FGM(object):
    """Reference: https://arxiv.org/pdf/1605.07725.pdf"""
    def __init__(self,
                 model,
                 emb_names=['word_embeddings', "encoder.layer.0"],        # emb_names 这个参数要换成你模型中embedding的参数名,可以是多组参数
                 epsilon=1.0):
        self.model = model
        self.emb_names = emb_names
        self.epsilon = epsilon
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self):
        """Add adversity."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                # 把真实参数保存起来
                self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self):
        """ restore embedding """
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if re.search("|".join(self.emb_names), name):
                    param.grad = self.grad_backup[name]
                else:
                    param.grad += self.grad_backup[name]


class PGD(object):
    """Reference: https://arxiv.org/pdf/1706.06083.pdf"""
    def __init__(self,
                 model,
                 emb_names=['word_embeddings', "encoder.layer.0"],
                 epsilon=1.0,
                 alpha=0.3):
        self.model = model
        self.emb_names = emb_names
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        """Add adversity."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.alpha * param.grad / norm
                    param.data.add_(r_adv)
                    param.data = self.project(name, param.data)

    def restore(self):
        """restore embedding"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r_adv = param_data - self.emb_backup[param_name]
        if torch.norm(r_adv) > self.epsilon:
            r_adv = self.epsilon * r_adv / torch.norm(r_adv)
        return self.emb_backup[param_name] + r_adv

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if re.search("|".join(self.emb_names), name):
                    param.grad = self.grad_backup[name]
                else:
                    param.grad += self.grad_backup[name]


class AWP:
    def __init__(
        self,
        model: Module,
        # criterion: _Loss,
        optimizer: Optimizer,
        apex: bool,
        adv_param: str="weight",
        adv_lr: float=1.0,
        adv_eps: float=0.01
    ) -> None:
        self.model = model
        # self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.apex = apex
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs: dict, label: Tensor) -> Tensor:
        # 直接调用这个函数（自定义书写）
        with torch.cuda.amp.autocast(enabled=self.apex):
            self._save()
            self._attack_step() # モデルを近傍の悪い方へ改変
            y_preds = self.model(inputs)
            adv_loss = self.criterion(
                y_preds.view(-1, 1), label.view(-1, 1))
            mask = (label.view(-1, 1) != -1)
            adv_loss = torch.masked_select(adv_loss, mask).mean()
            self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 直前に損失関数に通してパラメータの勾配を取得できるようにしておく必要あり
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}




# class AWP:
#     def __init__(
#         self,
#         model: Module,
#         criterion: _Loss,
#         optimizer: Optimizer,
#         apex: bool,
#         adv_param: str="weight",
#         adv_lr: float=1.0,
#         adv_eps: float=0.01
#     ) -> None:
#         self.model = model
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.adv_param = adv_param
#         self.adv_lr = adv_lr
#         self.adv_eps = adv_eps
#         self.apex = apex
#         self.backup = {}
#         self.backup_eps = {}
#
#     def attack_backward(self, inputs: dict, label: Tensor) -> Tensor:
#         # 直接调用这个函数（自定义书写）
#         with torch.cuda.amp.autocast(enabled=self.apex):
#             self._save()
#             self._attack_step() # モデルを近傍の悪い方へ改変
#             y_preds = self.model(inputs)
#             adv_loss = self.criterion(
#                 y_preds.view(-1, 1), label.view(-1, 1))
#             mask = (label.view(-1, 1) != -1)
#             adv_loss = torch.masked_select(adv_loss, mask).mean()
#             self.optimizer.zero_grad()
#         return adv_loss
#
#     def _attack_step(self) -> None:
#         e = 1e-6
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and param.grad is not None and self.adv_param in name:
#                 norm1 = torch.norm(param.grad)
#                 norm2 = torch.norm(param.data.detach())
#                 if norm1 != 0 and not torch.isnan(norm1):
#                     # 直前に損失関数に通してパラメータの勾配を取得できるようにしておく必要あり
#                     r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
#                     param.data.add_(r_at)
#                     param.data = torch.min(
#                         torch.max(
#                             param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
#                     )
#
#     def _save(self) -> None:
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and param.grad is not None and self.adv_param in name:
#                 if name not in self.backup:
#                     self.backup[name] = param.data.clone()
#                     grad_eps = self.adv_eps * param.abs().detach()
#                     self.backup_eps[name] = (
#                         self.backup[name] - grad_eps,
#                         self.backup[name] + grad_eps,
#                     )
#
#     def _restore(self) -> None:
#         for name, param in self.model.named_parameters():
#             if name in self.backup:
#                 param.data = self.backup[name]
#         self.backup = {}
#         self.backup_eps = {}
