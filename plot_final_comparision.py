import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
import re
import os
import csv
from matplotlib.ticker import MaxNLocator


def parse_log_file(filename):
    """
    解析日志文件，提取 Epoch, Loss, BLEU 数据。
    假设日志格式类似: "Epoch 1: Loss=5.7007 BLEU=7.26"
    """
    epochs = []
    losses = []
    bleus = []

    # 正则表达式匹配：查找 "Epoch 数字", "Loss=数字", "BLEU=数字"
    # 这里的正则设计得比较宽容，只要行内包含这些关键词和数值即可匹配
    pattern = re.compile(r"Epoch\s+(\d+).*?Loss=([\d\.]+).*?BLEU=([\d\.]+)")

    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return [], [], []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                bleus.append(float(match.group(3)))

    return epochs, losses, bleus


def _read_summary(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_best(rows, prefix):
    candidates = [r for r in rows if r["model"].startswith(prefix)]
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(r.get("best_bleu", 0) or 0))


def _pick_best_test(path):
    if not os.path.exists(path):
        return None
    rows = _read_summary(path)
    if not rows:
        return None
    return max(rows, key=lambda r: float(r.get("bleu", 0) or 0))


def _read_test_bleu(path):
    if not os.path.exists(path):
        return {}
    rows = _read_summary(path)
    scores = {}
    for row in rows:
        model = row.get("model")
        if not model:
            continue
        try:
            scores[model] = float(row.get("bleu", 0) or 0)
        except ValueError:
            scores[model] = 0.0
    return scores


def plot_academic_results():
    # 设置绘图风格
    if sns:
        sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'

    if not os.path.exists("nmt_comparison_summary.csv"):
        print("Missing nmt_comparison_summary.csv. Run analyze_results.py first.")
        return

    rows = _read_summary("nmt_comparison_summary.csv")
    test_bleu = {}
    test_bleu.update(_read_test_bleu(os.path.join("rnn_test_results", "summary.csv")))
    test_bleu.update(_read_test_bleu(os.path.join("transformer_test_results", "summary.csv")))
    rnn_test_best = _pick_best_test(os.path.join("rnn_test_results", "summary.csv"))
    trans_test_best = _pick_best_test(os.path.join("transformer_test_results", "summary.csv"))

    rnn_row = None
    trans_row = None
    if rnn_test_best:
        rnn_row = next((r for r in rows if r["model"] == rnn_test_best["model"]), None)
    if trans_test_best:
        trans_row = next((r for r in rows if r["model"] == trans_test_best["model"]), None)

    if not rnn_row:
        rnn_row = _pick_best(rows, "rnn_")
    if not trans_row:
        trans_row = _pick_best(rows, "transformer_")
    if not trans_row or not rnn_row:
        print("Missing transformer or rnn rows in summary.")
        return

    trans_log = trans_row["log_path"]
    rnn_log = rnn_row["log_path"]

    # 1. 解析 Transformer 日志
    trans_epochs, trans_loss, trans_bleu = parse_log_file(trans_log)
    if not trans_epochs:
        print("Transformer log is empty or invalid.")
        return

    # 2. 解析 RNN 日志
    rnn_epochs, rnn_loss, rnn_bleu = parse_log_file(rnn_log)
    if not rnn_epochs:
        print("RNN log is empty or invalid.")
        return

    max_epoch = max(trans_epochs[-1], rnn_epochs[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), dpi=300)

    # --- 左图：Loss 收敛曲线 (稳定性分析) ---
    ax1.plot(trans_epochs, trans_loss, 'o-', label=f"Transformer: {trans_row['model']}", color='#E63946', linewidth=2, markersize=4)
    ax1.plot(rnn_epochs, rnn_loss, 's--', label=f"RNN: {rnn_row['model']}", color='#457B9D', linewidth=2, markersize=4)
    ax1.set_title('Training Stability (Loss)', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.legend(frameon=True, shadow=True)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

    # --- 右图：BLEU 泛化曲线 (泛化性分析) ---
    ax2.plot(trans_epochs, trans_bleu, 'o-', label='Transformer BLEU', color='#E63946', linewidth=2, markersize=4)
    ax2.plot(rnn_epochs, rnn_bleu, 's--', label='RNN BLEU', color='#457B9D', linewidth=2, markersize=4)

    # 动态寻找最高点进行标注（不再使用硬编码）
    t_max_bleu = max(trans_bleu)
    t_max_idx = trans_bleu.index(t_max_bleu)
    t_max_epoch = trans_epochs[t_max_idx]

    g_max_bleu = max(rnn_bleu)
    g_max_idx = rnn_bleu.index(g_max_bleu)
    g_max_epoch = rnn_epochs[g_max_idx]

    # 标注 Transformer 最高点
    ax2.annotate(f'Trans Best: {t_max_bleu}', xy=(t_max_epoch, t_max_bleu), xytext=(t_max_epoch + 1, t_max_bleu + 5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1), fontsize=10, color='#E63946')

    # 标注 GRU 最高点
    ax2.annotate(f'RNN Best: {g_max_bleu}', xy=(g_max_epoch, g_max_bleu), xytext=(g_max_epoch - 4, g_max_bleu + 7),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1), fontsize=10, color='#457B9D')

    # 高亮过拟合区域 (这里假设从 Epoch 12 开始，或者你可以根据逻辑动态设定)
    if max_epoch >= 12:
        ax2.axvspan(12, max_epoch, color='gray', alpha=0.12, label='Overfitting Zone')

    ax2.set_title('Generalization Performance (BLEU)', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('BLEU Score', fontsize=12)
    ax2.set_ylim(0, max(max(trans_bleu), max(rnn_bleu)) + 5)
    ax2.legend(loc='lower right', frameon=True)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

    # Test BLEU overlay (from test summaries)
    rnn_test_bleu = test_bleu.get(rnn_row["model"])
    trans_test_bleu = test_bleu.get(trans_row["model"])
    if rnn_test_bleu is not None and trans_test_bleu is not None:
        ax2.text(
            0.02,
            0.02,
            f"Test BLEU - RNN: {rnn_test_bleu:.2f} | Transformer: {trans_test_bleu:.2f}",
            transform=ax2.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    plt.tight_layout()
    plt.savefig('nmt_final_comparison_log.png')
    plt.show()


if __name__ == "__main__":
    plot_academic_results()
