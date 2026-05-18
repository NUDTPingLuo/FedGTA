import torch
from fed_baselines.client_base import FedClient
from torch.utils.data import DataLoader
import copy

class GTAClient(FedClient):
    # 使用 **kwargs 以兼容 fl_main.py 可能传来的多余参数，防止参数对齐报错
    def __init__(self, name, epoch, dataset_id, model_name,
                 lr, batch_size, momentum, beta=0.1, **kwargs):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum, **kwargs)

        # 这里的 beta 现在代表相对比例 λ (建议设置在 0.1 到 0.5 之间)
        self._beta = beta

    # =====================================================
    # utils
    # =====================================================
    def _flatten_model_tensors(self, tensor_dict):
        """将梯度字典展平为一个 1D Tensor"""
        flat = []
        for name, tensor in tensor_dict.items():
            if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                continue
            flat.append(tensor.view(-1))
        if not flat:
            return torch.tensor([]).to(self._device)
        return torch.cat(flat, dim=0)

    def _unflatten_to_model(self, flat_tensor, model):
        """将一维修正向量还原并写回到模型的 .grad.data 中"""
        pointer = 0
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                num_param = p.numel()
                # 切片并 reshape 为原始形状，原地覆盖 grad
                p.grad.data.copy_(flat_tensor[pointer: pointer + num_param].view_as(p.data))
                pointer += num_param

    # =====================================================
    # main training interface
    # =====================================================
    def train(self):
        train_loader = DataLoader(
            self.trainset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=False
        )

        self.model.to(self._device)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self._lr,
            momentum=self._momentum
        )
        loss_func = torch.nn.CrossEntropyLoss()

        normal = (self.global_round == 0)

        # =====================================================
        # [提速核心] 在循环外预计算全局参考梯度的 1D 向量与单位向量
        # =====================================================
        g_star_unit = None
        if not normal and getattr(self, 'global_rectify_gradient', None) is not None:
            g_star_flat = self._flatten_model_tensors(
                {k: v.to(self._device) for k, v in self.global_rectify_gradient.items()}
            )
            g_star_norm = g_star_flat.norm() + 1e-12
            g_star_unit = g_star_flat / g_star_norm

        # 用来替代 grad_list，防止数百个 Batch 导致内存(RAM)爆炸
        total_sum = {
            name: torch.zeros_like(p.data)
            for name, p in self.model.named_parameters() if p.requires_grad
        }
        first_grad = None
        epoch_loss = []
        step_count = 0

        # =====================================================
        # Training Loop
        # =====================================================
        for _ in range(self._epoch):
            for x, y in train_loader:
                b_x, b_y = x.to(self._device), y.to(self._device)

                self.model.train()
                optimizer.zero_grad()

                output = self.model(b_x)
                loss = loss_func(output, b_y.long())
                loss.backward()

                # =====================================================
                # Mode B: 梯度范数比例锚定 (Norm-Matching Alignment)
                # =====================================================
                if not normal and g_star_unit is not None:
                    # 1. 提取当前梯度的全局 1D 向量
                    grad_dict = {name: p.grad.data for name, p in self.model.named_parameters() if p.requires_grad}
                    local_flat = self._flatten_model_tensors(grad_dict)
                    local_norm = local_flat.norm() + 1e-12
                    local_unit = local_flat / local_norm

                    # 2. 计算正交的“对齐修正方向” (点积计算 cos 相似度)
                    cos_val = torch.dot(local_unit, g_star_unit)
                    align_dir_flat = g_star_unit - cos_val * local_unit
                    align_dir_norm = align_dir_flat.norm() + 1e-12

                    # 3. 核心：范数锚定缩放 (Scale Invariance)
                    scaled_align_flat = self._beta * local_norm * (align_dir_flat / align_dir_norm)

                    # 4. 加到原模型梯度上
                    final_grad_flat = local_flat + scaled_align_flat

                    # 5. 将一维修正向量还原回模型的各个参数矩阵中
                    self._unflatten_to_model(final_grad_flat, self.model)

                # =====================================================
                # 记录第一步的梯度作为 rectify_grad (即你原代码的 grad_list[0])
                # =====================================================
                if step_count == 0:
                    first_grad = {
                        name: p.grad.detach().clone()
                        for name, p in self.model.named_parameters() if p.requires_grad
                    }

                # =====================================================
                # OOM 安全的梯度累加 (替代 grad_list.append)
                # =====================================================
                for name, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        total_sum[name].add_(p.grad.detach())

                optimizer.step()
                epoch_loss.append(loss.item())
                step_count += 1

        # =====================================================
        # 构建并返回给 Server 的数据字典
        # =====================================================
        final_grad_dict = copy.deepcopy(self.model.state_dict())
        for name in final_grad_dict:
            if (
                "running_mean" not in name
                and "running_var" not in name
                and "num_batches_tracked" not in name
            ):
                if name in total_sum:
                    final_grad_dict[name] = total_sum[name] * self._lr

        avg_loss = sum(epoch_loss) / len(epoch_loss)

        # 返回顺序匹配 server_fedgta.py 中的: rec(name, final_grad, n_data, loss, first_grad)
        return final_grad_dict, self.n_data, avg_loss, first_grad