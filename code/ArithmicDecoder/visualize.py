import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    print("Warning: UMAP not available. Install with: pip install umap-learn")
    UMAP_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import os


class EmbeddingTrajectoryTracker:
    """用于跟踪模型生成过程中的嵌入向量轨迹"""

    def __init__(self, model, tokenizer, config):
        """
        初始化轨迹跟踪器

        参数:
        - model: 要分析的模型
        - tokenizer: 用于编码/解码的分词器
        - config: 配置对象
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()  # 设置为评估模式
        self.trajectories = {}  # 存储所有记录的轨迹

    def record_generation_trajectory(
        self, prompt, temperature=0.8, record_hidden=False
    ):
        """
        记录生成过程中的嵌入向量轨迹

        参数:
        - prompt: 起始提示
        - temperature: 采样温度
        - record_hidden: 是否记录隐藏状态(需要修改模型以支持)

        返回:
        - 轨迹数据字典
        """
        self.model.eval()
        trajectories = {
            "token_embeddings": [],  # 记录token嵌入
            "hidden_states": [] if record_hidden else None,  # 记录隐藏状态(如果需要)
            "tokens": [],  # 记录生成的token
            "token_ids": [],  # 记录token ID
            "logits": [],  # 记录预测logits
            "attention": [],  # 记录注意力权重
            "positions": [],  # 记录位置信息
            "entropies": [],  # 记录预测分布的熵
            "top_probs": [],  # 记录top-k概率
        }

        with torch.no_grad():
            # 编码输入序列
            input_seq = self.tokenizer.encode(prompt)
            answer_start_pos = len(input_seq)
            padded_input = input_seq.copy()
            current_pos = answer_start_pos
            max_length = min(35, self.config.max_seq_len)

            # 记录初始提示的嵌入向量
            for i, token_id in enumerate(input_seq):
                token = self.tokenizer.itos[token_id]
                trajectories["tokens"].append(token)
                trajectories["token_ids"].append(token_id)
                trajectories["positions"].append(i)

            # 生成过程
            while current_pos < max_length - 1:
                # 准备当前输入
                current_input = padded_input + [self.tokenizer.pad_id] * (
                    max_length - len(padded_input)
                )
                input_tensor = torch.tensor(
                    [current_input], dtype=torch.long, device=self.config.device
                )

                # 获取嵌入向量
                if hasattr(self.model, "module"):
                    token_embeddings = self.model.module.token_embed(input_tensor)
                else:
                    token_embeddings = self.model.token_embed(input_tensor)

                # 记录当前位置的嵌入向量
                trajectories["token_embeddings"].append(
                    token_embeddings[0, current_pos].cpu().numpy()
                )

                # 获取模型输出和下一个token
                logits = self.model(input_tensor)
                next_token_logits = logits[0, current_pos]

                # 计算熵
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                log_probs = F.log_softmax(next_token_logits / temperature, dim=-1)
                entropy = -torch.sum(probs * log_probs).item()
                trajectories["entropies"].append(entropy)

                # 记录top-k概率
                top_k = 5
                top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
                top_tokens = [
                    (self.tokenizer.itos[idx.item()], prob.item())
                    for idx, prob in zip(top_indices, top_probs)
                ]
                trajectories["top_probs"].append(top_tokens)

                # 记录logits
                trajectories["logits"].append(next_token_logits.cpu().numpy())

                # 采样下一个token
                next_token = torch.multinomial(probs, 1).item()

                # 更新序列和记录
                padded_input.append(next_token)
                token = self.tokenizer.itos[next_token]
                trajectories["tokens"].append(token)
                trajectories["token_ids"].append(next_token)
                trajectories["positions"].append(current_pos)

                current_pos += 1

                # 如果生成了结束符，停止生成
                if next_token == self.tokenizer.eoa_id:
                    break

        # 保存轨迹数据
        problem_key = prompt
        self.trajectories[problem_key] = trajectories

        # 打印生成结果
        print(f"生成轨迹: {prompt} → {''.join(trajectories['tokens'])}")

        return trajectories

    def save_trajectories(self, filepath):
        """保存记录的轨迹到文件"""
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self.trajectories, f)
        print(f"轨迹数据已保存至 {filepath}")

    def load_trajectories(self, filepath):
        """从文件加载轨迹"""
        import pickle

        with open(filepath, "rb") as f:
            self.trajectories = pickle.load(f)
        print(f"已从 {filepath} 加载轨迹数据")
        return self.trajectories


class EmbeddingVisualizer:
    """用于可视化嵌入轨迹的工具类"""

    def __init__(self, save_dir="visualization_results"):
        """
        初始化可视化器

        参数:
        - save_dir: 保存可视化结果的目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def visualize_trajectory_2d(
        self,
        tracker,
        prompt,
        method="umap",
        n_neighbors=15,
        min_dist=0.1,
        perplexity=30,
        show_only_answer=True,
        with_lines=True,
        figsize=(12, 10),
        save=False,
    ):
        """
        使用降维技术将嵌入轨迹可视化为2D

        参数:
        - tracker: EmbeddingTrajectoryTracker实例
        - prompt: 要可视化的提示
        - method: 降维方法 ('umap', 'tsne', 或 'pca')
        - show_only_answer: 是否只显示答案部分的轨迹
        - with_lines: 是否显示连接线
        """
        # 检查轨迹是否存在
        if prompt not in tracker.trajectories:
            print(f"提示'{prompt}'的轨迹不存在，正在生成...")
            trajectory = tracker.record_generation_trajectory(prompt)
        else:
            trajectory = tracker.trajectories[prompt]

        # 提取嵌入向量
        embeddings = np.array(trajectory["token_embeddings"])
        tokens = trajectory["tokens"]
        positions = trajectory["positions"]

        # 如果只显示答案部分
        if show_only_answer:
            # 找到答案开始的位置
            answer_start = len(prompt)
            answer_indices = [
                i for i, pos in enumerate(positions) if pos >= answer_start
            ]

            if not answer_indices:
                print("没有找到答案部分的嵌入向量")
                return None

            embeddings = embeddings[answer_indices]
            tokens = [tokens[i] for i in answer_indices]

        # 选择降维方法
        if method == "umap":
            if not UMAP_AVAILABLE:
                print("UMAP未安装，使用PCA替代")
                reducer = PCA(n_components=2, random_state=42)
            else:
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=2,
                    random_state=42,
                )
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        else:  # 默认使用PCA
            reducer = PCA(n_components=2)

        # 降维
        embedding_2d = reducer.fit_transform(embeddings)

        # 可视化
        plt.figure(figsize=figsize)

        # 绘制轨迹线
        if with_lines and len(embedding_2d) > 1:
            plt.plot(embedding_2d[:, 0], embedding_2d[:, 1], "b-", alpha=0.3)

        # 绘制点并标注token
        for i, (x, y) in enumerate(embedding_2d):
            plt.scatter(x, y, s=100, c=[plt.cm.rainbow(i / len(embeddings))], alpha=0.8)
            plt.annotate(
                tokens[i],
                (x, y),
                fontsize=12,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
            )

        # 添加颜色条和标题
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap="rainbow"), label="Generation sequence"
        )
        plt.title(f"Embedding Trajectory for: {prompt}")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # 保存图像
        if save:
            filename = (
                f"{self.save_dir}/trajectory_{method}_{prompt.replace('=', '_eq_')}.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"图像已保存至 {filename}")

        return plt

    def create_trajectory_animation(
        self, tracker, prompt, method="umap", interval=700, save=False
    ):
        """
        创建嵌入轨迹的动画

        参数:
        - tracker: EmbeddingTrajectoryTracker实例
        - prompt: 要可视化的提示
        - method: 降维方法
        - interval: 帧间隔(毫秒)
        - save: 是否保存动画
        """
        # 检查轨迹是否存在
        if prompt not in tracker.trajectories:
            print(f"提示'{prompt}'的轨迹不存在，正在生成...")
            trajectory = tracker.record_generation_trajectory(prompt)
        else:
            trajectory = tracker.trajectories[prompt]

        # 提取答案部分的嵌入向量
        answer_start = len(prompt)
        answer_indices = [
            i for i, pos in enumerate(trajectory["positions"]) if pos >= answer_start
        ]

        if not answer_indices:
            print("没有找到答案部分的嵌入向量")
            return None

        embeddings = np.array(trajectory["token_embeddings"])[answer_indices]
        tokens = [trajectory["tokens"][i] for i in answer_indices]

        # 选择降维方法
        if method == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42)
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        else:  # 默认使用PCA
            reducer = PCA(n_components=2)

        # 降维
        embedding_2d = reducer.fit_transform(embeddings)

        # 创建动画
        fig, ax = plt.subplots(figsize=(10, 8))

        def init():
            ax.clear()
            ax.set_xlim(embedding_2d[:, 0].min() - 0.5, embedding_2d[:, 0].max() + 0.5)
            ax.set_ylim(embedding_2d[:, 1].min() - 0.5, embedding_2d[:, 1].max() + 0.5)
            ax.set_title(f"Embedding Trajectory: {prompt}")
            ax.grid(alpha=0.3)
            return []

        def update(frame):
            ax.clear()
            ax.set_xlim(embedding_2d[:, 0].min() - 0.5, embedding_2d[:, 0].max() + 0.5)
            ax.set_ylim(embedding_2d[:, 1].min() - 0.5, embedding_2d[:, 1].max() + 0.5)
            ax.grid(alpha=0.3)

            # 绘制已经生成的轨迹
            if frame > 0:
                ax.plot(
                    embedding_2d[:frame, 0], embedding_2d[:frame, 1], "b-", alpha=0.6
                )

            # 绘制所有点
            for i in range(min(frame + 1, len(embedding_2d))):
                ax.scatter(
                    embedding_2d[i, 0],
                    embedding_2d[i, 1],
                    s=100,
                    c=[plt.cm.rainbow(i / len(embeddings))],
                    alpha=0.8,
                )
                ax.annotate(
                    tokens[i],
                    (embedding_2d[i, 0], embedding_2d[i, 1]),
                    fontsize=12,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                )

            # 显示当前生成的token
            if frame < len(tokens):
                ax.set_title(
                    f"Embedding Trajectory: {prompt} | Current token: {tokens[frame]}"
                )
            else:
                ax.set_title(f"Embedding Trajectory: {prompt} | Complete")

            return []

        ani = FuncAnimation(
            fig,
            update,
            frames=len(embedding_2d) + 10,
            init_func=init,
            blit=True,
            interval=interval,
        )

        # 保存动画
        if save:
            filename = (
                f"{self.save_dir}/animation_{method}_{prompt.replace('=', '_eq_')}.gif"
            )
            ani.save(filename, writer="pillow", fps=1000 // interval, dpi=100)
            print(f"动画已保存至 {filename}")

        return ani

    def create_interactive_3d_trajectory(
        self, tracker, prompts, method="umap", save=False
    ):
        """创建多个问题的3D交互式轨迹可视化"""
        if not PLOTLY_AVAILABLE:
            print("Plotly未安装，无法创建交互式3D可视化")
            return None

        # 收集多个问题的轨迹
        all_embeddings = []
        all_tokens = []
        all_problems = []

        for prompt in prompts:
            if prompt not in tracker.trajectories:
                print(f"提示'{prompt}'的轨迹不存在，正在生成...")
                trajectory = tracker.record_generation_trajectory(prompt)
            else:
                trajectory = tracker.trajectories[prompt]

            # 只提取答案部分
            answer_start = len(prompt)
            answer_indices = [
                i
                for i, pos in enumerate(trajectory["positions"])
                if pos >= answer_start
            ]

            if not answer_indices:
                print(f"警告: 没有找到'{prompt}'的答案部分")
                continue

            embeddings = np.array(trajectory["token_embeddings"])[answer_indices]
            tokens = [trajectory["tokens"][i] for i in answer_indices]

            all_embeddings.extend(embeddings)
            all_tokens.extend(tokens)
            all_problems.extend([prompt] * len(embeddings))

        if not all_embeddings:
            print("没有找到任何有效的嵌入向量")
            return None

        # 选择降维方法
        if method == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=3, random_state=42)
        elif method == "tsne":
            reducer = TSNE(n_components=3, random_state=42)
        else:  # 默认使用PCA
            reducer = PCA(n_components=3)

        # 降维
        embeddings_3d = reducer.fit_transform(np.array(all_embeddings))

        # 创建3D图
        fig = go.Figure()

        # 按问题分组添加轨迹
        for prompt in set(all_problems):
            indices = [i for i, p in enumerate(all_problems) if p == prompt]
            x, y, z = (
                embeddings_3d[indices, 0],
                embeddings_3d[indices, 1],
                embeddings_3d[indices, 2],
            )
            tokens = [all_tokens[i] for i in indices]

            # 添加线
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    name=f"{prompt} (path)",
                    line=dict(width=4, color="rgba(70, 130, 180, 0.8)"),
                    hoverinfo="none",
                )
            )

            # 添加点
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers+text",
                    name=f"{prompt} (tokens)",
                    marker=dict(
                        size=8,
                        color=list(range(len(indices))),
                        colorscale="Viridis",
                        opacity=0.8,
                    ),
                    text=tokens,
                    hovertemplate="Token: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}",
                )
            )

        # 布局
        fig.update_layout(
            title="3D Token Embedding Trajectories",
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3",
            ),
            width=1000,
            height=800,
            margin=dict(l=0, r=0, b=0, t=40),
        )

        # 保存图像
        if save:
            filename = f"{self.save_dir}/3d_trajectory_{method}.html"
            fig.write_html(filename)
            print(f"交互式图表已保存至 {filename}")

        return fig

    def analyze_embedding_clusters(self, tracker, prompts, n_clusters=5, save=False):
        """分析不同问题生成轨迹的聚类模式"""
        # 收集多个问题的嵌入向量
        all_embeddings = []
        all_tokens = []
        all_problems = []

        for prompt in prompts:
            if prompt not in tracker.trajectories:
                print(f"提示'{prompt}'的轨迹不存在，正在生成...")
                trajectory = tracker.record_generation_trajectory(prompt)
            else:
                trajectory = tracker.trajectories[prompt]

            # 只提取答案部分
            answer_start = len(prompt)
            answer_indices = [
                i
                for i, pos in enumerate(trajectory["positions"])
                if pos >= answer_start
            ]

            if not answer_indices:
                print(f"警告: 没有找到'{prompt}'的答案部分")
                continue

            embeddings = np.array(trajectory["token_embeddings"])[answer_indices]
            tokens = [trajectory["tokens"][i] for i in answer_indices]

            all_embeddings.extend(embeddings)
            all_tokens.extend(tokens)
            all_problems.extend([prompt] * len(embeddings))

        if not all_embeddings:
            print("没有找到任何有效的嵌入向量")
            return None, None

        # 聚类分析
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(all_embeddings)

        # UMAP降维可视化
        if UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)

        embedding_2d = reducer.fit_transform(all_embeddings)

        # 可视化聚类结果
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=clusters,
            cmap="viridis",
            s=80,
            alpha=0.8,
        )

        # 为每个点添加标签
        for i, (x, y) in enumerate(embedding_2d):
            plt.annotate(all_tokens[i], (x, y), fontsize=10, alpha=0.7)

        plt.colorbar(scatter, label="Cluster")
        plt.title("Embedding Clusters Analysis")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # 分析每个聚类中的token分布
        cluster_analysis = {}
        for i in range(n_clusters):
            indices = np.where(clusters == i)[0]
            tokens_in_cluster = [all_tokens[idx] for idx in indices]
            problems_in_cluster = [all_problems[idx] for idx in indices]

            # 统计分析
            token_counts = Counter(tokens_in_cluster)
            problem_counts = Counter(problems_in_cluster)

            cluster_analysis[i] = {
                "tokens": token_counts,
                "problems": problem_counts,
                "size": len(indices),
            }

        # 保存图像
        if save:
            filename = f"{self.save_dir}/cluster_analysis_{n_clusters}clusters.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"聚类分析图像已保存至 {filename}")

        # 打印聚类分析结果
        print("\n===== 聚类分析结果 =====")
        for cluster_id, data in cluster_analysis.items():
            print(f"\n集群 {cluster_id+1} (大小: {data['size']}):")
            print("  常见Token:")
            for token, count in data["tokens"].most_common(5):
                print(f"    {token}: {count}次 ({count/data['size']:.1%})")
            print("  问题分布:")
            for problem, count in data["problems"].most_common():
                print(f"    {problem}: {count}次")

        return plt, cluster_analysis

    def compare_trajectories(self, tracker, prompts, save=False):
        """比较不同问题生成轨迹的相似度"""
        # 收集轨迹数据
        mean_embeddings = {}
        for prompt in prompts:
            if prompt not in tracker.trajectories:
                print(f"提示'{prompt}'的轨迹不存在，正在生成...")
                trajectory = tracker.record_generation_trajectory(prompt)
            else:
                trajectory = tracker.trajectories[prompt]

            # 只提取答案部分
            answer_start = len(prompt)
            answer_indices = [
                i
                for i, pos in enumerate(trajectory["positions"])
                if pos >= answer_start
            ]

            if not answer_indices:
                print(f"警告: 没有找到'{prompt}'的答案部分")
                continue

            embeddings = np.array(trajectory["token_embeddings"])[answer_indices]
            mean_embeddings[prompt] = np.mean(embeddings, axis=0)

        # 计算轨迹之间的距离矩阵
        prompts_list = list(mean_embeddings.keys())
        if not prompts_list:
            print("没有找到任何有效的嵌入向量")
            return None, None

        embeddings_array = np.array([mean_embeddings[p] for p in prompts_list])
        distances = squareform(pdist(embeddings_array, "cosine"))

        # 可视化距离矩阵
        plt.figure(figsize=(10, 8))
        im = plt.imshow(distances, cmap="viridis")
        plt.colorbar(im, label="Cosine Distance")
        plt.xticks(range(len(prompts_list)), prompts_list, rotation=90)
        plt.yticks(range(len(prompts_list)), prompts_list)
        plt.title("Trajectory Similarity Comparison")
        plt.tight_layout()

        # 保存图像
        if save:
            filename = f"{self.save_dir}/trajectory_similarity.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"轨迹相似度矩阵已保存至 {filename}")

        return plt, distances


# 辅助函数
def visualize_entropy_distribution(tracker, prompts, save_dir=None):
    """可视化预测熵的分布"""
    all_entropies = []
    all_tokens = []

    for prompt in prompts:
        if prompt not in tracker.trajectories:
            print(f"提示'{prompt}'的轨迹不存在，跳过...")
            continue

        trajectory = tracker.trajectories[prompt]

        # 只分析答案部分
        answer_start = len(prompt)
        answer_indices = [
            i for i, pos in enumerate(trajectory["positions"]) if pos >= answer_start
        ]

        if not answer_indices:
            print(f"警告: 没有找到'{prompt}'的答案部分")
            continue

        entropies = [trajectory["entropies"][i] for i in answer_indices]
        tokens = [trajectory["tokens"][i] for i in answer_indices]

        all_entropies.extend(entropies)
        all_tokens.extend(tokens)

    if not all_entropies:
        print("没有找到任何有效的熵数据")
        return None

    # 创建图表
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(all_entropies, bins=20, alpha=0.7)
    plt.xlabel("Prediction Entropy")
    plt.ylabel("Count")
    plt.title("Distribution of Prediction Entropy")
    plt.grid(alpha=0.3)

    # 添加平均线
    mean_entropy = np.mean(all_entropies)
    plt.axvline(
        x=mean_entropy, color="red", linestyle="--", label=f"Mean: {mean_entropy:.2f}"
    )
    plt.legend()

    # 查找熵值最高和最低的token
    token_entropies = list(zip(all_tokens, all_entropies))
    token_entropies.sort(key=lambda x: x[1])

    # 创建柱状图显示熵值最高和最低的tokens
    top_n = min(10, len(token_entropies) // 2)  # 确保有足够的token
    low_entropy_tokens = token_entropies[:top_n]
    high_entropy_tokens = token_entropies[-top_n:]

    plt.subplot(1, 2, 2)
    tokens = [t for t, _ in low_entropy_tokens] + [t for t, _ in high_entropy_tokens]
    entropies = [e for _, e in low_entropy_tokens] + [e for _, e in high_entropy_tokens]
    colors = ["green"] * top_n + ["red"] * top_n
    y_pos = np.arange(len(tokens))

    plt.barh(y_pos, entropies, color=colors)
    plt.yticks(y_pos, tokens)
    plt.xlabel("Entropy")
    plt.title(f"Tokens with Lowest and Highest Entropy")
    plt.axvline(
        x=mean_entropy, color="black", linestyle="--", label=f"Mean: {mean_entropy:.2f}"
    )
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # 保存图像
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/entropy_distribution.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"熵分布图像已保存至 {filename}")

    return plt


def visualize_token_prediction_confidence(
    tracker, prompts, token_focus=None, top_k=5, save_dir=None
):
    """可视化特定token的预测置信度"""
    token_predictions = {}  # {token: [(prob, problem), ...]}

    for prompt in prompts:
        if prompt not in tracker.trajectories:
            print(f"提示'{prompt}'的轨迹不存在，跳过...")
            continue

        trajectory = tracker.trajectories[prompt]

        # 只分析答案部分
        answer_start = len(prompt)
        answer_indices = [
            i for i, pos in enumerate(trajectory["positions"]) if pos >= answer_start
        ]

        if not answer_indices:
            print(f"警告: 没有找到'{prompt}'的答案部分")
            continue

        for i in answer_indices:
            token = trajectory["tokens"][i]
            if token_focus and token not in token_focus:
                continue

            top_probs = (
                trajectory["top_probs"][i - len(prompt)] if i >= len(prompt) else []
            )

            if token not in token_predictions:
                token_predictions[token] = []

            # 找到正确token的概率
            token_prob = 0.0
            for t, prob in top_probs:
                if t == token:
                    token_prob = prob
                    break

            token_predictions[token].append((token_prob, prompt))

    if not token_predictions:
        print("没有找到任何符合条件的token预测数据")
        return None

    # 为关注的token或所有出现的token创建置信度分布图
    plt.figure(figsize=(12, 8))

    tokens_to_plot = (
        sorted(token_predictions.keys()) if not token_focus else token_focus
    )
    for i, token in enumerate(tokens_to_plot):
        if token not in token_predictions:
            continue

        probs = [p for p, _ in token_predictions[token]]
        plt.subplot(2, 2, i % 4 + 1)
        plt.hist(probs, bins=10, alpha=0.7)
        plt.xlabel("Prediction Probability")
        plt.ylabel("Count")
        plt.title(f'Token "{token}" Prediction Confidence')

        # 添加平均线
        mean_prob = np.mean(probs)
        plt.axvline(
            x=mean_prob, color="red", linestyle="--", label=f"Mean: {mean_prob:.2f}"
        )
        plt.legend()
        plt.grid(alpha=0.3)

        if (i + 1) % 4 == 0 or i == len(tokens_to_plot) - 1:
            plt.tight_layout()

            # 保存图像
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                page_num = (i // 4) + 1
                filename = f"{save_dir}/token_confidence_page{page_num}.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Token置信度图像已保存至 {filename}")

            if i < len(tokens_to_plot) - 1:
                plt.figure(figsize=(12, 8))

    return plt


# 主函数，用于演示可视化功能
def visualize_main(
    model, tokenizer, config, prompts=None, save_dir="visualization_results"
):
    """主函数，演示使用可视化工具"""
    if prompts is None:
        prompts = ["12+34=", "56*78=", "9*87=", "45+67="]

    # 创建轨迹跟踪器和可视化器
    tracker = EmbeddingTrajectoryTracker(model, tokenizer, config)
    visualizer = EmbeddingVisualizer(save_dir=save_dir)

    # 记录轨迹
    for prompt in prompts:
        tracker.record_generation_trajectory(prompt)

    # 保存轨迹数据以便将来使用
    tracker.save_trajectories(f"{save_dir}/trajectories.pkl")

    # 演示各种可视化功能
    print("\n===== 2D轨迹可视化 =====")
    for method in ["umap", "tsne", "pca"]:
        for prompt in prompts[:2]:  # 只演示前两个提示
            visualizer.visualize_trajectory_2d(
                tracker, prompt, method=method, save=True
            )

    print("\n===== 轨迹动画 =====")
    visualizer.create_trajectory_animation(tracker, prompts[0], save=True)

    print("\n===== 3D交互式轨迹 =====")
    visualizer.create_interactive_3d_trajectory(tracker, prompts, save=True)

    print("\n===== 聚类分析 =====")
    visualizer.analyze_embedding_clusters(tracker, prompts, n_clusters=3, save=True)

    print("\n===== 轨迹相似度比较 =====")
    visualizer.compare_trajectories(tracker, prompts, save=True)

    print("\n===== 熵分布可视化 =====")
    visualize_entropy_distribution(tracker, prompts, save_dir=save_dir)

    print("\n===== Token预测置信度可视化 =====")
    visualize_token_prediction_confidence(
        tracker, prompts, token_focus=["1", "2", "3"], save_dir=save_dir
    )

    print("\n可视化完成。所有结果已保存到:", save_dir)
    return tracker, visualizer


if __name__ == "__main__":
    import argparse
    import sys
    import os

    # 添加项目根目录到Python路径以便导入模块
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models import EnhancedTransformer
    from config import Config
    from tokenizer import EnhancedTokenizer

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="嵌入空间轨迹可视化工具")

    # 必需参数组
    required_group = parser.add_argument_group("必需参数")
    required_group.add_argument(
        "--model", type=str, required=True, help="模型文件路径 (.pth)"
    )

    # 可视化模式
    mode_group = parser.add_argument_group("可视化模式")
    mode_group.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=[
            "auto",
            "2d",
            "3d",
            "animation",
            "cluster",
            "compare",
            "entropy",
            "confidence",
            "all",
        ],
        help="可视化模式",
    )

    # 提示和样本
    sample_group = parser.add_argument_group("样本与提示")
    sample_group.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["12+34=", "56*78=", "9*87=", "45+67="],
        help="要分析的算术问题列表",
    )
    sample_group.add_argument(
        "--token-focus",
        type=str,
        nargs="+",
        default=None,
        help="在置信度分析中关注的特定token",
    )

    # 降维和聚类参数
    dim_group = parser.add_argument_group("降维与聚类")
    dim_group.add_argument(
        "--method",
        type=str,
        default="umap",
        choices=["umap", "tsne", "pca"],
        help="降维方法",
    )
    dim_group.add_argument("--clusters", type=int, default=5, help="聚类分析中的簇数量")

    # 生成和输出参数
    output_group = parser.add_argument_group("生成与输出")
    output_group.add_argument(
        "--save-dir",
        type=str,
        default="visualization_results",
        help="可视化结果保存目录",
    )
    output_group.add_argument(
        "--temperature", type=float, default=0.8, help="生成的温度参数"
    )
    output_group.add_argument(
        "--load-trajectories", type=str, default=None, help="从文件加载已保存的轨迹数据"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 初始化配置
    config = Config()

    # 初始化分词器
    tokenizer = EnhancedTokenizer(config)

    # 加载模型
    model = EnhancedTransformer(config, tokenizer)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model = model.to(config.device)

    model.load_state_dict(torch.load(args.model, map_location=config.device))
    model.eval()
    print(f"成功加载模型: {args.model}")

    # 创建轨迹跟踪器和可视化器
    tracker = EmbeddingTrajectoryTracker(model, tokenizer, config)
    visualizer = EmbeddingVisualizer(save_dir=args.save_dir)

    # 如果指定了轨迹文件，尝试加载
    if args.load_trajectories and os.path.exists(args.load_trajectories):
        tracker.load_trajectories(args.load_trajectories)
    else:
        # 否则为每个提示生成新轨迹
        for prompt in args.prompts:
            tracker.record_generation_trajectory(prompt, temperature=args.temperature)

        # 保存轨迹以便将来使用
        os.makedirs(args.save_dir, exist_ok=True)
        tracker.save_trajectories(f"{args.save_dir}/trajectories.pkl")

    # 根据选择的模式执行可视化
    if args.mode in ["auto", "all", "2d"]:
        print("\n===== 2D轨迹可视化 =====")
        for prompt in args.prompts:
            visualizer.visualize_trajectory_2d(
                tracker, prompt, method=args.method, save=True
            )

    if args.mode in ["auto", "all", "animation"]:
        print("\n===== 轨迹动画 =====")
        for prompt in args.prompts[:2]:  # 限制只生成前两个提示的动画
            visualizer.create_trajectory_animation(
                tracker, prompt, method=args.method, save=True
            )

    if args.mode in ["auto", "all", "3d"]:
        print("\n===== 3D交互式轨迹 =====")
        visualizer.create_interactive_3d_trajectory(
            tracker, args.prompts, method=args.method, save=True
        )

    if args.mode in ["auto", "all", "cluster"]:
        print("\n===== 聚类分析 =====")
        visualizer.analyze_embedding_clusters(
            tracker, args.prompts, n_clusters=args.clusters, save=True
        )

    if args.mode in ["auto", "all", "compare"]:
        print("\n===== 轨迹相似度比较 =====")
        visualizer.compare_trajectories(tracker, args.prompts, save=True)

    if args.mode in ["auto", "all", "entropy"]:
        print("\n===== 熵分布可视化 =====")
        visualize_entropy_distribution(tracker, args.prompts, save_dir=args.save_dir)

    if args.mode in ["auto", "all", "confidence"]:
        print("\n===== Token预测置信度可视化 =====")
        visualize_token_prediction_confidence(
            tracker, args.prompts, token_focus=args.token_focus, save_dir=args.save_dir
        )

    print(f"\n可视化完成。所有结果已保存到: {args.save_dir}")
