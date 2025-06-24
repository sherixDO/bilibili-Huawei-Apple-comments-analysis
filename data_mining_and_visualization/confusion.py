import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import pandas as pd

# --- 设置 ---
# 指定中文字体文件路径，确保 simhei.ttf 在同一目录下
try:
    font = FontProperties(fname='simhei.ttf', size=14)
except FileNotFoundError:
    print("错误：未在本文件夹下找到 simhei.ttf 字体文件。")
    print("请下载 SimHei 字体并将其命名为 simhei.ttf 放置在代码相同目录下。")
    # 如果找不到字体，使用系统默认字体，中文可能显示为方块
    font = FontProperties(size=14)

# 类别标签
labels = ['apple', 'huawei']
chinese_labels = ['苹果', '华为']

# --- 1. 混淆矩阵可视化 ---

# 根据您的数据计算得出的混淆矩阵
# 格式: [[TP_apple, FN_apple], [FP_apple, TN_apple]]
# FP_apple (假阳性) 等于 FN_huawei (假阴性)
# TN_apple (真阴性) 等于 TP_huawei (真阳性)
conf_matrix = np.array([
    [2775, 452],  # 实际为 apple: 2775个被正确预测, 452个被错误预测为 huawei
    [386, 2832]   # 实际为 huawei: 386个被错误预测为 apple, 2832个被正确预测
])

# 创建一个图形和坐标轴
fig, ax = plt.subplots(figsize=(8, 6))

# 使用 seaborn 生成热力图
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=chinese_labels, yticklabels=chinese_labels,
            annot_kws={"size": 16})

# 设置标题和标签，并应用中文字体
ax.set_title('混淆矩阵 (Confusion Matrix)', fontproperties=font, fontsize=20, pad=20)
ax.set_xlabel('预测类别 (Predicted Label)', fontproperties=font, fontsize=16, labelpad=15)
ax.set_ylabel('实际类别 (True Label)', fontproperties=font, fontsize=16, labelpad=15)

# 设置刻度标签字体
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)

# 调整布局并显示图像
plt.tight_layout()
plt.savefig('./output_images/10_confusion_matrix.png', dpi=300)
plt.show()



# --- 2. 热力图 ---

# 从您的报告中提取数据
metrics_data = {
    'Precision': [0.87, 0.86],
    'Recall': [0.86, 0.88],
    'f1-score': [0.87, 0.87]
}

# 创建 DataFrame
metrics_df = pd.DataFrame(metrics_data, index=labels)
# 创建一个图形和坐标轴
fig, ax = plt.subplots(figsize=(8, 6))
# 使用 seaborn 生成热力图
sns.heatmap(metrics_df, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax,
            xticklabels=chinese_labels, yticklabels=metrics_df.index,
            annot_kws={"size": 16})
# 设置标题和标签，并应用中文字体
ax.set_title('分类指标热力图 (Classification Metrics Heatmap)', fontproperties=font, fontsize=20, pad=20)
ax.set_xlabel('手机品牌 (Mobile Brand)', fontproperties=font, fontsize=16, labelpad=15)
ax.set_ylabel('指标 (Metrics)', fontproperties=font, fontsize=16, labelpad=15)
# 设置刻度标签字体
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)
# 调整布局并显示图像
plt.tight_layout()
plt.savefig('./output_images/11_classification_metrics_heatmap.png', dpi=300)
plt.show()


