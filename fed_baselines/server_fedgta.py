from fed_baselines.server_base import FedServer
import copy
import torch
import os
import csv


class GTAServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name, batch_size, select_ratio=0.5, dirichlet_alpha=0.5,
                 prox_beta=0.1):
        super().__init__(client_list, dataset_id, model_name, batch_size)

        self.dataset_id = dataset_id
        self.dirichlet_alpha = dirichlet_alpha
        self.prox_beta = prox_beta

        self.round_grad_sum_history = []
        self.global_rectify_gradient = None

    def _flatten_grad_dict(self, grad_dict):
        """Flatten a gradient dict into a single 1-D tensor (skip BN buffers)."""
        flat = []
        for k, v in grad_dict.items():
            if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                continue
            flat.append(v.view(-1))
        return torch.cat(flat, dim=0)

    def agg(self):
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        self.model.to(self._device)
        global_model = self.model.state_dict()
        new_model = copy.deepcopy(global_model)
        avg_loss = 0

        grad_sum = {}
        first_grad_sum = {}

        # ===== 聚合 =====
        for i, name in enumerate(self.selected_clients):
            weight = self.client_n_data[name] / self.n_data
            client_grad = self.client_state[name]["final_grad"]
            first_grad = self.client_state[name]["first_grad"]

            for key in client_grad:
                if i == 0:
                    grad_sum[key] = client_grad[key].to(self._device) * weight
                    if "running_mean" not in key and "running_var" not in key and "num_batches_tracked" not in key:
                        first_grad_sum[key] = first_grad[key].to(self._device) / client_num
                else:
                    grad_sum[key] += client_grad[key].to(self._device) * weight
                    if "running_mean" not in key and "running_var" not in key and "num_batches_tracked" not in key:
                        first_grad_sum[key] += first_grad[key].to(self._device) / client_num

            avg_loss += self.client_loss[name] * weight

        # ============================================================
        # 动态构建分层目录
        # ============================================================
        csv_dir = os.path.join(".", "csv", str(self.dataset_id), f"alpha_{self.dirichlet_alpha}")
        os.makedirs(csv_dir, exist_ok=True)

        # ============================================================
        # [修复] 记录 Cosine 相似度 (静态全量表头，未选中填 NaN)
        # ============================================================
        csv_cos_path = os.path.join(csv_dir, f"cosine_vs_client_beta_{self.prox_beta}.csv")

        # 强制固定全量客户端作为表头
        all_fieldnames = ["round"] + self.client_list
        cos_row = {"round": self.round}

        if self.global_rectify_gradient is not None:
            g_star_flat = self._flatten_grad_dict(self.global_rectify_gradient).to(self._device)
            g_star_norm = g_star_flat.norm() + 1e-12

            for name in self.selected_clients:
                client_grad = self.client_state[name]["final_grad"]
                client_flat = self._flatten_grad_dict(client_grad).to(self._device)
                client_norm = client_flat.norm() + 1e-12
                cos_val = torch.dot(client_flat, g_star_flat) / (client_norm * g_star_norm)
                cos_row[name] = cos_val.item()

        # 补齐当轮未被选中的客户端，填入 NaN
        for name in self.client_list:
            if name not in cos_row:
                cos_row[name] = float("nan")

        write_header_cos = not os.path.exists(csv_cos_path)
        with open(csv_cos_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames)
            if write_header_cos:
                writer.writeheader()
            writer.writerow(cos_row)

        # ============================================================
        # [方案 A] 计算 客户端漂移 (Cosine Drift / Directional Variance)
        # 公式: Drift = sum_k p_k * (1 - cos(g_k, g_global))
        # ============================================================
        csv_drift_path = os.path.join(csv_dir, f"cosine_drift_beta_{self.prox_beta}.csv")

        g_global_flat = self._flatten_grad_dict(grad_sum).to(self._device)
        g_global_norm = g_global_flat.norm() + 1e-12

        drift_sum = 0.0
        for name in self.selected_clients:
            weight = self.client_n_data[name] / self.n_data
            client_grad = self.client_state[name]["final_grad"]

            # 展平客户端最终更新梯度
            client_flat = self._flatten_grad_dict(client_grad).to(self._device)
            client_norm = client_flat.norm() + 1e-12

            # 计算 Cosine 相似度
            cos_sim = torch.dot(client_flat, g_global_flat) / (client_norm * g_global_norm)

            # Cosine Drift = 1 - Cosine Similarity (值越小，代表漂移越小，方向越一致)
            client_drift = 1.0 - cos_sim.item()
            drift_sum += weight * client_drift

        write_header_drift = not os.path.exists(csv_drift_path)
        with open(csv_drift_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header_drift:
                writer.writerow(["round", "cosine_drift"])
            writer.writerow([self.round, drift_sum])
        # ============================================================

        # 更新 global_rectify_gradient
        self.global_rectify_gradient = first_grad_sum

        # 更新全局模型
        for key in global_model:
            if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
                new_model[key] = grad_sum[key]
            else:
                new_model[key] = global_model[key] - grad_sum[key]

        self.model.load_state_dict(new_model)
        self.round += 1

        return new_model, avg_loss, self.n_data

    def update_opt_gradient(self):
        return self.global_rectify_gradient

    def rec(self, name, final_grad, n_data, loss, first_grad):
        self.n_data += n_data
        self.client_state[name] = {"final_grad": final_grad, "first_grad": first_grad}
        self.client_n_data[name] = n_data
        self.client_loss[name] = loss

    def flush(self):
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}