# 华为与苹果手机评论数据挖掘与分析

这个项目主要针对B站上关于华为和苹果手机的评论进行数据采集、情感分析和可视化，通过对比两大品牌在B站评论区的用户反馈来分析用户偏好和情感倾向。

## 项目功能

- 爬取B站视频评论（支持多视频批量爬取）
- 评论情感分析（正面、负面、中性）
- 数据可视化（情感分布、评论关键词、高赞评论分析）
- 品牌对比分析（苹果 vs 华为）

## 环境设置

强烈建议使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理，这是一个基于Rust的快速Python包安装器和解析器。

### 安装 uv

如果您尚未安装uv，可以通过以下命令安装：

```bash
# 使用curl安装
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 安装项目依赖

```bash
# 使用pyproject.toml
uv sync
```

## 项目结构

```
Data_mining_report/
├── README.md                     # 项目说明文档
├── comments.py                   # B站评论爬取脚本
├── visualization.py              # 数据可视化脚本
├── confusion.py                  # 混淆矩阵和评估指标可视化
├── simhei.ttf                    # 中文字体文件（用于可视化）
├── pyproject.toml                # 项目依赖配置
├── uv.lock                       # uv依赖锁定文件
├── apple_analyzed_results.csv    # 苹果评论分析结果
├── huawei_analyzed_results.csv   # 华为评论分析结果
├── bilibili_comments_apple.csv   # 苹果评论原始数据
├── bilibili_comments_huawei.csv  # 华为评论原始数据
├── bilibili_comments_apple.db    # 苹果评论数据库
├── bilibili_comments_huawei.db   # 华为评论数据库
└── output_images/                # 输出图像目录
    ├── 1_sentiment_distribution_bar.png
    ├── 2_sentiment_distribution_pie.png
    ├── 3_likes_vs_sentiment_boxplot.png
    ├── 4_wordcloud_huawei_negative_content.png
    ├── 5_wordcloud_apple_negative_content.png
    ├── 6_wordcloud_huawei_all_content.png
    ├── 7_wordcloud_apple_all_content.png
    ├── 8_wordcloud_huawei_top_likes_content.png
    ├── 9_wordcloud_apple_top_likes_content.png
    ├── 10_confusion_matrix.png
    └── 11_classification_metrics_heatmap.png
```

## 使用说明

### 1. 评论数据爬取

在 `comments.py` 中设置要爬取的B站视频BV号和您的SESSDATA：

```python
# 设置要爬取的视频BVID列表
BVID_LIST = [
    "BV1JoCpYLECF",
    "BV1j6iYYHEYG"
]

# 设置您的SESSDATA (从浏览器Cookie获取)
SESSDATA = "你的SESSDATA值"
```

运行爬虫脚本：

```bash
uv run comments.py
```

### 2. 数据可视化

确保已经准备好以下文件：
- huawei_analyzed_results.csv
- apple_analyzed_results.csv
- simhei.ttf (中文字体文件)

运行可视化脚本：

```bash
uv run visualization.py
```

脚本会在 `output_images` 目录下生成各种可视化图表。

### 3. 混淆矩阵可视化

运行混淆矩阵和分类指标可视化脚本：

```bash
uv run confusion.py
```

## 关键技术

- **数据获取**：使用Python requests库获取B站评论数据
- **数据存储**：同时支持CSV和SQLite数据库
- **文本分析**：使用jieba分词进行中文文本处理
- **可视化**：使用matplotlib、seaborn和wordcloud进行数据可视化
- **反爬虫策略**：随机延时、模拟浏览器Headers

## 注意事项

1. 请确保 `simhei.ttf` 字体文件位于项目根目录，以便正确显示中文
2. 爬取B站数据时需要有效的SESSDATA，可从浏览器Cookie中获取
3. 爬取过程中请遵守B站API使用规范，避免频繁请求
4. 代码中已设置随机延时以避免被反爬虫机制识别

## 依赖项

主要依赖库：
- pandas
- matplotlib
- seaborn
- wordcloud
- jieba
- requests
- sqlite3

详细依赖请参考 `pyproject.toml` 文件。

## 许可证

MIT
