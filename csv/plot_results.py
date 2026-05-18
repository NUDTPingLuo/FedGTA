import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# ==========================================
# 全局配置参数 (修改这里即可切换数据集和目录)
# ==========================================
DATASET = "FashionMNIST"  # 数据集名称
ALPHA = "1"  # Dirichlet Alpha 参数
BETA_AVG = "0"  # FedAvg 对应的 Beta
BETA_GTA = "0.1"  # FedGTA 对应的 Beta

# 设置全局绘图样式
sns.set_theme(style="whitegrid")
rcParams['font.family'] = 'DejaVu Sans'
# 10种高区分度标记符号
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']


def get_paths(dataset, alpha):
    """辅助函数：构建数据读取和图片保存的路径"""
    data_dir = os.path.join(dataset, f"alpha_{alpha}")
    plot_dir = data_dir
    os.makedirs(plot_dir, exist_ok=True)
    return data_dir, plot_dir


def plot_client_trajectories(dataset=DATASET, alpha=ALPHA):
    """
    功能 1: 分别绘制 FedAvg 和 FedGTA 的各个客户端 Cosine 轨迹图
    """
    data_dir, plot_dir = get_paths(dataset, alpha)

    file_avg = os.path.join(data_dir, f"cosine_vs_client_beta_{BETA_AVG}.csv")
    file_gta = os.path.join(data_dir, f"cosine_vs_client_beta_{BETA_GTA}.csv")

    if not os.path.exists(file_avg) or not os.path.exists(file_gta):
        print(f"⚠️ 找不到文件，请检查路径: {file_avg} 或 {file_gta}")
        return

    df_avg = pd.read_csv(file_avg)
    df_gta = pd.read_csv(file_gta)

    client_cols = [c for c in df_avg.columns if c != 'round']
    colors = sns.color_palette("tab10", len(client_cols))

    # --- 绘制 FedAvg ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, client in enumerate(client_cols):
        ax.plot(df_avg['round'], df_avg[client],
                marker=MARKERS[i % len(MARKERS)], color=colors[i],
                linestyle='-', linewidth=1.5, alpha=0.7, label=client)

    ax.set_title(f'FedAvg - Client Cosine Similarity [{dataset} $\\alpha$={alpha}]', fontsize=14, pad=10,
                 fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.12), ncol=5, fontsize=10, frameon=False)
    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)

    # 💡 修改点：文件名加入 alpha 值
    plt.savefig(os.path.join(plot_dir, f"{dataset}_alpha{alpha}_client_cosine_FedAvg.pdf"), dpi=300,
                bbox_inches='tight')
    plt.close()

    # --- 绘制 FedGTA ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, client in enumerate(client_cols):
        ax.plot(df_gta['round'], df_gta[client],
                marker=MARKERS[i % len(MARKERS)], color=colors[i],
                linestyle='-', linewidth=1.5, alpha=0.7, label=client)

    ax.set_title(f'FedGTA - Client Cosine Similarity [{dataset} $\\alpha$={alpha}]', fontsize=14, pad=10,
                 fontweight='bold')
    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)

    # 💡 修改点：文件名加入 alpha 值
    plt.savefig(os.path.join(plot_dir, f"{dataset}_alpha{alpha}_client_cosine_FedGTA.pdf"), dpi=300,
                bbox_inches='tight')
    plt.close()

    print(f"✅ 功能 1 (客户端轨迹) 绘图完成，已保存至 {plot_dir}")


def plot_cosine_drift(dataset=DATASET, alpha=ALPHA):
    """
    功能 2: 将 FedAvg 和 FedGTA 的 Cosine Drift 画在一张图中的两条线上
    """
    data_dir, plot_dir = get_paths(dataset, alpha)

    file_avg = os.path.join(data_dir, f"cosine_drift_beta_{BETA_AVG}.csv")
    file_gta = os.path.join(data_dir, f"cosine_drift_beta_{BETA_GTA}.csv")

    if not os.path.exists(file_avg): file_avg = os.path.join(data_dir, f"drift_beta_{BETA_AVG}.csv")
    if not os.path.exists(file_gta): file_gta = os.path.join(data_dir, f"drift_beta_{BETA_GTA}.csv")

    if not os.path.exists(file_avg) or not os.path.exists(file_gta):
        print(f"⚠️ 找不到 Drift 文件: {file_avg} 或 {file_gta}")
        return

    df_avg = pd.read_csv(file_avg)
    df_gta = pd.read_csv(file_gta)

    y_col_avg = 'cosine_drift' if 'cosine_drift' in df_avg.columns else 'drift_variance'
    y_col_gta = 'cosine_drift' if 'cosine_drift' in df_gta.columns else 'drift_variance'

    # 定义柔和高级配色
    soft_red = '#E15759'  # 柔和番茄红
    soft_green = '#59A14F'  # 莫兰迪绿

    plt.figure(figsize=(8, 5))
    plt.plot(df_avg['round'], df_avg[y_col_avg], label='FedAvg', color=soft_red, linewidth=2.5, alpha=0.85)
    plt.plot(df_gta['round'], df_gta[y_col_gta], label='FedGTA', color=soft_green, linewidth=2.5, alpha=0.85)

    plt.title(f'Directional Variance (Cosine Drift) - {dataset} ($\\alpha$={alpha})', fontsize=16, fontweight='bold')
    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Cosine Drift (Lower is better)', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(prop={'size': 14})
    plt.tight_layout()

    # 💡 修改点：文件名加入 alpha 值
    plt.savefig(os.path.join(plot_dir, f"{dataset}_alpha{alpha}_cosine_drift_comparison.pdf"), dpi=300,
                bbox_inches='tight')
    plt.close()

    print(f"✅ 功能 2 (Drift 对比) 绘图完成，已保存至 {plot_dir}")


def plot_client_mean_error_bar(dataset=DATASET, alpha=ALPHA):
    """
    功能 3: 统计客户端 Cosine 的平均值和误差棒，画分组直方图
    """
    data_dir, plot_dir = get_paths(dataset, alpha)

    file_avg = os.path.join(data_dir, f"cosine_vs_client_beta_{BETA_AVG}.csv")
    file_gta = os.path.join(data_dir, f"cosine_vs_client_beta_{BETA_GTA}.csv")

    if not os.path.exists(file_avg) or not os.path.exists(file_gta):
        print("⚠️ 找不到文件，无法绘制直方图。")
        return

    df_avg = pd.read_csv(file_avg).drop(columns=['round'], errors='ignore')
    df_gta = pd.read_csv(file_gta).drop(columns=['round'], errors='ignore')

    mean_avg = df_avg.mean(skipna=True)
    std_avg = df_avg.std(skipna=True)

    mean_gta = df_gta.mean(skipna=True)
    std_gta = df_gta.std(skipna=True)

    clients = mean_avg.index.tolist()
    x = np.arange(len(clients))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width / 2, mean_avg, width, yerr=std_avg,
           color='cornflowerblue', hatch='////', alpha=0.8,
           edgecolor='black', capsize=4, label='FedAvg')

    ax.bar(x + width / 2, mean_gta, width, yerr=std_gta,
           color='salmon', alpha=0.9,
           edgecolor='black', capsize=4, label='FedGTA')

    ax.set_ylabel('Average Cosine Similarity', fontsize=14)
    ax.set_title(f'Client-wise Cosine Similarity Distribution - {dataset} ($\\alpha$={alpha})', fontsize=16,
                 fontweight='bold', pad=15)
    ax.set_xticks(x)

    simplified_clients = [f"C{str(i)}" for i in range(len(clients))]
    ax.set_xticklabels(simplified_clients, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.legend(prop={'size': 14}, loc='upper left')
    ax.axhline(0, color='black', linewidth=1)

    # 💡 修改点：文件名加入 alpha 值
    plt.savefig(os.path.join(plot_dir, f"{dataset}_alpha{alpha}_client_mean_error_bars.pdf"), dpi=300,
                bbox_inches='tight')
    plt.close()

    print(f"✅ 功能 3 (误差棒直方图) 绘图完成，已保存至 {plot_dir}")


if __name__ == "__main__":
    print(f"🚀 开始处理数据: {DATASET} | Alpha={ALPHA}")
    plot_client_trajectories(DATASET, ALPHA)
    plot_cosine_drift(DATASET, ALPHA)
    plot_client_mean_error_bar(DATASET, ALPHA)
    print("🎉 全部处理完毕！图片已存入对应的数据集文件夹。")