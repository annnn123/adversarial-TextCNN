import torch


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class PGD:
    def __init__(self, model, eps, alpha):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='embedding', first=True):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if first:
                    self.emb_backup[name] = param.data.clone()
                    self.r_adv = torch.zeros_like(param)
                else:
                    self.r_adv = self.r_adv + self.alpha * param.grad.sign()
                self.r_adv = clamp(self.r_adv, torch.tensor(-self.eps), torch.tensor(self.eps))
                param.data = self.emb_backup[name] + self.r_adv

    def restore_emb(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
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
                param.grad = self.grad_backup[name]


class Free:
    def __init__(self, model, eps):
        self.model = model
        self.eps = eps
        self.emb_backup = {}
        self.grad_backup = {}
        self.r_adv = 0

    def attack(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.emb_backup[name] = param.data.clone()

                if name in self.grad_backup:
                    self.r_adv = self.r_adv + self.eps * self.grad_backup[name].sign()
                    self.r_adv = clamp(self.r_adv, torch.tensor(-self.eps), torch.tensor(self.eps))
                    self.grad_backup = {}
                param.data = param.data + self.r_adv

    def restore_emb(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
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
                param.grad = self.grad_backup[name]




class FGSM:
    def __init__(self, model, eps, alpha):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='embedding', first=True):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if first:
                    self.emb_backup[name] = param.data.clone()
                    self.r_adv = torch.zeros_like(param)
                    self.r_adv.uniform_(-self.eps, self.eps)
                else:
                    self.r_adv = self.r_adv + self.alpha * param.grad.sign()
                self.r_adv = clamp(self.r_adv, torch.tensor(-self.eps), torch.tensor(self.eps))
                param.data = self.emb_backup[name] + self.r_adv

    def restore_emb(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
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
                param.grad = self.grad_backup[name]