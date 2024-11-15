import os
import sys
sys.path.append('/Project/Yi/defense')
from CODE.Utils.package import *

class FGSM:
    def __init__(self, model, eps=0.1, device=None):
        """
        初始化FGSM攻击类。
        :param model: 被攻击的模型
        :param eps: 攻击强度参数，越大攻击越强
        """
        self.model = model
        self.eps = eps
        self.device = device

    def generate(self, x):
        """
        生成对抗样本。
        :param x: 原始输入样本
        :return: 对抗样本
        """
        x = x.to(self.device)
        # 确保模型的参数不会更新
        x.requires_grad = True
        x_adv = x.clone().detach().requires_grad_(True)
        x_adv = x_adv.to(self.device)
        # 前向传播
        y_pred = self.model(x)

        y_target = self.get_y_target(x, y_pred)

        # 计算损失
        loss = nn.functional.cross_entropy(y_pred, y_target, reduction="mean")
        
        # 反向传播
        self.model.zero_grad()
        loss.backward()
        
        # 生成对抗样本
        x_adv.data = x.data - self.eps * x.grad.sign()
        
        return x_adv

    def get_y_target(self, x, y_pred):
        """
        :param x: 输入张量
        :return: 目标预测值，未扰动数据的预测值
        """
        with torch.no_grad():
            y_target = torch.zeros_like(y_pred)
            _, c = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=y_pred.device)
                c_s = c_s[c_s != c[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0

        # 应用softmax函数获取最终的目标预测值
        return y_target

class BIM:
    def __init__(self, model, eps_init=0.001, eps=0.1, beta=0.0005, num_iters=1000, device=None):
        """
        Initialize the BIM attack class.
        :param model: The model to attack.
        :param eps: Maximum perturbation amount (clip bound).
        :param beta: Scaling factor for each iteration's step size.
        :param num_iters: Number of iterations to perform.
        """
        self.model = model
        self.eps = eps
        self.beta = beta
        self.num_iters = num_iters
        self.device = device
        self.eps_init = eps_init

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def generate(self, x):
        """
        Generate adversarial examples.
        :param x: Original input samples.
        :return: Adversarial examples.
        """
        x = x.to(self.device)
        x_adv = x.clone().detach().requires_grad_(True)
        # Forward pass
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)

        y_target = self.get_y_target(x, y_pred)
        for _ in range(self.num_iters):

            y_pred_adv = self.model(x+r)
            loss = nn.functional.cross_entropy(y_pred_adv, y_target, reduction="mean")
            print(f'{_} adversarial loss is {loss}')
            optimizer.zero_grad()
            loss.backward()
            # Here, we use the sign of the gradient for the update
            grad_sign = r.grad.sign()
            r.data = r.data - self.beta * grad_sign
            r.data = torch.clamp(r.data, -self.eps, self.eps)
            #print(self.num_iters)
        x_adv = x + r
        
        return x_adv

    def get_y_target(self, x, y_pred):
        """
        :param x: 输入张量
        :return: 目标预测值，未扰动数据的预测值
        """
        with torch.no_grad():
            y_target = torch.zeros_like(y_pred)
            _, c = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=y_pred.device)
                c_s = c_s[c_s != c[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0

        # 应用softmax函数获取最终的目标预测值
        return y_target
'''
class FGSM:
    def __init__(self, model, eps=0.1):
        """
        初始化FGSM攻击类。
        :param model: 被攻击的模型
        :param eps: 攻击强度参数，越大攻击越强
        """
        self.model = model
        self.eps = eps

    def generate(self, x, labels):
        """
        生成对抗样本。
        :param x: 原始输入样本
        :return: 对抗样本
        """
        # 确保模型的参数不会更新
        x.requires_grad = True
        x_adv = x.clone().detach().requires_grad_(True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        # 前向传播
        y_pred = self.model(x)
        # 计算损失
        loss = -criterion(y_pred, labels)
        
        # 反向传播
        self.model.zero_grad()
        loss.backward()
        
        # 生成对抗样本
        x_adv.data = x.data - self.eps * x.grad.sign()
        
        return x_adv

class BIM:
    def __init__(self, model, eps=0.1, beta=0.0005, num_iters=1000):
        """
        Initialize the BIM attack class.
        :param model: The model to attack.
        :param eps: Maximum perturbation amount (clip bound).
        :param beta: Scaling factor for each iteration's step size.
        :param num_iters: Number of iterations to perform.
        """
        self.model = model
        self.eps = eps
        self.beta = beta
        self.num_iters = num_iters

    def generate(self, x, labels):
        """
        Generate adversarial examples.
        :param x: Original input samples.
        :return: Adversarial examples.
        """
        x.requires_grad = True
        x_adv = x.clone().detach().requires_grad_(True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        # Forward pass
        y_pred = self.model(x)

        for _ in range(self.num_iters):

            self.model.zero_grad()
        
            # 重新对 x_adv 进行前向传播以获取当前的梯度
            y_pred = self.model(x_adv)
            # Compute loss
            loss = -criterion(y_pred, labels)
            # Backward pass
            
            loss.backward()

            # Update adversarial example with a small step size
            x_adv.data = x_adv.data - self.beta * x_adv.grad.sign()

            # Clip the adversarial example to be within the specified epsilon neighborhood of the original input
            x_adv.data = torch.clamp(x_adv.data, x.data - self.eps, x.data + self.eps)
        return x_adv
'''
class GM:
    def __init__(self, model, eps_init=0.001, eps=0.1, num_iters=1000, device=None):
        """
        Initialize the GM attack class.
        :param model: The model to attack.
        :param eps: Maximum perturbation amount (clip bound).
        :param beta: Scaling factor for each iteration's step size.
        :param num_iters: Number of iterations to perform.
        """
        self.model = model
        self.eps = eps
        self.num_iters = num_iters
        self.eps_init = eps_init
        self.device = device
    
    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def generate(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target = self.get_y_target(x, y_pred)
        # 这里看起来不需要to_device

        for epoch in range(self.num_iters):
            y_pred_adv = self.model(x+r)
            loss = nn.functional.cross_entropy(y_pred_adv, y_target, reduction="mean")
            print(f'{epoch} adversarial loss is {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        #y_adv = self.f(x_adv).argmax(1)

        return x_adv

    def get_y_target(self, x, y_pred):
        """
        :param x: 输入张量
        :return: 目标预测值，未扰动数据的预测值
        """
        with torch.no_grad():
            y_target = torch.zeros_like(y_pred)
            _, c = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=y_pred.device)
                c_s = c_s[c_s != c[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0

        # 应用softmax函数获取最终的目标预测值
        return y_target

class GM_l2:
    def __init__(self, model, eps_init=0.001, eps=0.1, num_iters=1000, device=None, beta=1):
        """
        Initialize the GM attack class.
        :param model: The model to attack.
        :param eps: Maximum perturbation amount (clip bound).
        :param beta: Scaling factor for each iteration's step size.
        :param num_iters: Number of iterations to perform.
        """
        self.model = model
        self.eps = eps
        self.num_iters = num_iters
        self.eps_init = eps_init
        self.device = device
        self.beta = beta

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def generate(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target = self.get_y_target(x, y_pred)
        # 这里看起来不需要to_device

        for epoch in range(self.num_iters):
            y_pred_adv = self.model(x+r)
            # 原始的交叉熵损失
            loss_ce = nn.functional.cross_entropy(y_pred_adv, y_target, reduction="mean")

            # 计算r的L2范数
            l2_norm = torch.norm(r, p=2)
            # 总损失 = 交叉熵损失 + lambda * L2范数
            loss_total = loss_ce + self.beta * l2_norm
            optimizer.zero_grad()
            # 反向传播总损失
            loss_total.backward()
            
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        #y_adv = self.f(x_adv).argmax(1)

        return x_adv

    def get_y_target(self, x, y_pred):
        """
        :param x: 输入张量
        :return: 目标预测值，未扰动数据的预测值
        """
        with torch.no_grad():
            y_target = torch.zeros_like(y_pred)
            _, c = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=y_pred.device)
                c_s = c_s[c_s != c[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0

        # 应用softmax函数获取最终的目标预测值
        return y_target

class PGD:
    def __init__(self, model, eps_init=0.001, eps=0.1, beta=0.0005, num_iters=1000, device=None):
        """
        Initialize the PGD attack class.
        :param model: The model to attack.
        :param eps: Maximum perturbation amount (clip bound).
        :param beta: Scaling factor for each iteration's step size.
        :param num_iters: Number of iterations to perform.
        """
        self.model = model
        self.eps = eps
        self.beta = beta
        self.num_iters = num_iters
        self.device = device
        self.eps_init = eps_init
        
    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def generate(self, x):
        """
        Generate adversarial examples.
        :param x: Original input samples.
        :return: Adversarial examples.
        """
        x = x.to(self.device)
        x_adv = x.clone().detach().requires_grad_(True)
        # Forward pass
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)

        y_target = self.get_y_target(x, y_pred)
        for _ in range(self.num_iters):
            # 在每个迭代步骤开始时添加随机扰动
            if _ > 0:  # 跳过第一次迭代，因为已经初始化过了
                random_noise = (torch.rand_like(x) * 2 - 1) * self.eps_init
                r.data = r.data + random_noise
                r.data = torch.clamp(r.data, -self.eps, self.eps)  # 确保扰动仍在允许的范围内

            y_pred_adv = self.model(x+r)
            loss = nn.functional.cross_entropy(y_pred_adv, y_target, reduction="mean")
            print(f'{_} adversarial loss is {loss}')
            optimizer.zero_grad()
            loss.backward()
            # Here, we use the sign of the gradient for the update
            grad_sign = r.grad.sign()
            r.data = r.data - self.beta * grad_sign
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        return x_adv

    def get_y_target(self, x, y_pred):
        """
        :param x: 输入张量
        :return: 目标预测值，未扰动数据的预测值
        """
        with torch.no_grad():
            y_target = torch.zeros_like(y_pred)
            _, c = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=y_pred.device)
                c_s = c_s[c_s != c[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0

        return y_target

class PGD_for_AT:
    def __init__(self, model, eps_init=0.001, eps=0.1, beta=0.0005, num_iters=1000, device=None):
        """
        Initialize the PGD attack class.
        :param model: The model to attack.
        :param eps: Maximum perturbation amount (clip bound).
        :param beta: Scaling factor for each iteration's step size.
        :param num_iters: Number of iterations to perform.
        """
        self.model = model
        self.eps = eps
        self.beta = beta
        self.num_iters = num_iters
        self.device = device
        self.eps_init = eps_init
        
    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def generate(self, x, y):
        """
        Generate adversarial examples.
        :param x: Original input samples.
        :return: Adversarial examples.
        """
        x = x.to(self.device)
        x_adv = x.clone().detach().requires_grad_(True)
       
       
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)

        for _ in range(self.num_iters):
            optimizer.zero_grad()
            y_pred_adv = self.model(x+r)
            loss = nn.functional.cross_entropy(y_pred_adv, y, reduction="mean")
            
            loss.backward()
            
            # Here, we use the sign of the gradient for the update
            grad_sign = r.grad.sign()
            r.data = r.data + self.beta * grad_sign
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        return x_adv

class CW:
    def __init__(self, model, eps_init=0.001, eps=0.1, c=1e-5, num_iters=1000, device=None):

        self.model = model
        self.eps = eps
        self.c = c
        self.num_iters = num_iters
        self.device = device
        self.eps_init = eps_init

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def __CW_loss_fun__(self, x, r, y_target, top1_index):
        y_pred_adv = self.model(x + r)
        y_target_labels = torch.max(y_target, 1)[1]  # 从one-hot转为索引
        loss = nn.functional.cross_entropy(y_pred_adv, y_target_labels, reduction="none")  # 改为none以保留每个样本的loss

        mask = torch.zeros_like(loss, dtype=torch.bool)
        _, top1_index_adv = torch.max(y_pred_adv, dim=1)

        for i in range(len(y_target)):
            if not top1_index_adv[i].item() == top1_index[i].item():  # 使用.item()来比较
                mask[i] = True
        loss[mask] = 0

        # Combine the attack loss with the L2 regularization
        l2_reg = torch.norm(r, p=2)

        return l2_reg * self.c + loss.mean()

    def generate(self, x):
        """
        Generate adversarial examples.
        :param x: Original input samples.
        :return: Adversarial examples.
        """
        x = x.to(self.device)
        #x_adv = x.clone().detach().requires_grad_(True)
        # Forward pass
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)

        y_target, top1_index = self.get_y_target(x, y_pred)
        for _ in range(self.num_iters):

            #y_pred_adv = self.model(x+r)
            loss = self.__CW_loss_fun__(x, r, y_target, top1_index)
            print(f'{_} adversarial loss is {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        return x_adv

    def get_y_target(self, x, y_pred):
        """
        :param x: 输入张量
        :return: 目标预测值，未扰动数据的预测值
        """
        with torch.no_grad():
            y_target = torch.zeros_like(y_pred)
            _, c1 = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=self.device)
                c_s = c_s[c_s != c1[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0

        # 应用softmax函数获取最终的目标预测值
        return y_target, c1

class SWAP:
    def __init__(self, model, eps_init=0.001, eps=0.1, gamma=0.01, num_iters=1000, device=None):
  
        self.model = model
        self.eps = eps
        self.gamma = gamma
        self.num_iters = num_iters
        self.device = device
        self.eps_init = eps_init

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def generate(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target = self.get_y_target(x, y_pred)
        # 这里看起来不需要to_device

        for epoch in range(self.num_iters):
            y_pred_adv = self.model(x+r)
            loss = nn.functional.cross_entropy(y_pred_adv, y_target, reduction="mean")
            print(f'{epoch} adversarial loss is {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        #y_adv = self.f(x_adv).argmax(1)

        return x_adv

    def get_y_target(self, x, y_pred):
        # 这里index用于挑选那一个预测用于swap
        with torch.no_grad():
            _, top2_indices = torch.topk(y_pred, 2, dim=1)
            y_target = y_pred.clone()

            for i in range(len(y_pred)):
                c_top2 = top2_indices[i]
                mean_ = (
                    y_pred[i, c_top2[0]] + y_pred[i, c_top2[1]]
                ) / 2  # 交换第一和第二项的值，保持原有分布
                y_target[i, c_top2[1]] = mean_ + self.gamma
                y_target[i, c_top2[0]] = mean_ - self.gamma  # 让原始的第二项比第一项稍大一点
        return y_target

class SWAP_l2:
    def __init__(self, model, eps_init=0.001, eps=0.1, gamma=0.01, num_iters=1000, device=None, beta=0.1):
  
        self.model = model
        self.eps = eps
        self.gamma = gamma
        self.num_iters = num_iters
        self.device = device
        self.eps_init = eps_init
        self.beta = beta

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)


    def generate(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.model(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target = self.get_y_target(x, y_pred)
        # 这里看起来不需要to_device

        for epoch in range(self.num_iters):
            y_pred_adv = self.model(x+r)
            # 原始的交叉熵损失
            loss_ce = nn.functional.cross_entropy(y_pred_adv, y_target, reduction="mean")

            # 计算r的L2范数
            l2_norm = torch.norm(r, p=2)
            # 总损失 = 交叉熵损失 + lambda * L2范数
            loss_total = loss_ce + self.beta * l2_norm
            optimizer.zero_grad()
            # 反向传播总损失
            loss_total.backward()
            
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)

        x_adv = x + r
        #y_adv = self.f(x_adv).argmax(1)

        return x_adv

    def get_y_target(self, x, y_pred):
        # 这里index用于挑选那一个预测用于swap
        with torch.no_grad():
            _, top2_indices = torch.topk(y_pred, 2, dim=1)
            y_target = y_pred.clone()

            for i in range(len(y_pred)):
                c_top2 = top2_indices[i]
                mean_ = (
                    y_pred[i, c_top2[0]] + y_pred[i, c_top2[1]]
                ) / 2  # 交换第一和第二项的值，保持原有分布
                y_target[i, c_top2[1]] = mean_ + self.gamma
                y_target[i, c_top2[0]] = mean_ - self.gamma  # 让原始的第二项比第一项稍大一点
        return y_target