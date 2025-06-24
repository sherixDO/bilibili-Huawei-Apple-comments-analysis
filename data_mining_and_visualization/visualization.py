import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from wordcloud import WordCloud
import jieba
import os


# --- 1. 环境准备与数据加载（已更新字体处理和图片保存逻辑） ---

# --- 智能字体设置 ---
def set_chinese_font():
    """
    设置中文字体为同目录下的 simhei.ttf。
    返回字体路径，供 WordCloud 使用。
    """
    font_path = 'simhei.ttf'
    if not os.path.exists(font_path):
        print(f"❌ 字体文件 '{font_path}' 不存在于当前目录。")
        print("请确保 simhei.ttf 文件与此脚本在同一文件夹下。")
        print("图表中的中文可能无法正常显示。")
        plt.rcParams['axes.unicode_minus'] = False
        return None

    try:
        # 将字体文件添加到 matplotlib 的字体管理器中，使其可以按名称找到
        fm.fontManager.addfont(font_path)
        
        # 设置 matplotlib 使用该字体
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"✅ 成功将字体 '{font_name}' ({font_path}) 添加到 Matplotlib 并设为默认。")
        return font_path
    except Exception as e:
        print(f"⚠️ 加载字体 {font_path} 失败: {e}")
        print("图表中的中文可能无法正常显示。")
        return None


# 执行字体设置，并获取WordCloud所需的字体路径
CHINESE_FONT_PATH = set_chinese_font()

# 检查字体是否成功设置
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)
print(f"所有图片将保存在 '{output_dir}' 文件夹中。")

# 加载数据
huawei_df = pd.read_csv('huawei_analyzed_results.csv')
apple_df = pd.read_csv('apple_analyzed_results.csv')
huawei_df['brand'] = '华为'
apple_df['brand'] = '苹果'
combined_df = pd.concat([huawei_df, apple_df], ignore_index=True)

# --- 2. 总体情感分布对比 ---
print("\n--- 正在生成情感分布对比图... ---")
sentiment_order = ['正面', '中性', '负面']
plt.figure(figsize=(12, 6))
sns.countplot(data=combined_df, x='sentiment', hue='brand', order=sentiment_order, palette='viridis')
plt.title('华为 vs 苹果 Bilibili评论情感分布对比', fontsize=16)
plt.xlabel('情感类别', fontsize=12)
plt.ylabel('评论数量', fontsize=12)
plt.legend(title='手机品牌')
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 保存图片
plt.savefig(os.path.join(output_dir, '1_sentiment_distribution_bar.png'), dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
huawei_sentiment_counts = huawei_df['sentiment'].value_counts()
axes[0].pie(huawei_sentiment_counts, labels=huawei_sentiment_counts.index, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette('viridis', n_colors=len(huawei_sentiment_counts)))
axes[0].set_title('华为评论情感分布', fontsize=16)
apple_sentiment_counts = apple_df['sentiment'].value_counts()
axes[1].pie(apple_sentiment_counts, labels=apple_sentiment_counts.index, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette('viridis', n_colors=len(apple_sentiment_counts)))
axes[1].set_title('苹果评论情感分布', fontsize=16)
plt.suptitle('华为与苹果评论情感分布饼图', fontsize=20)
# 保存图片
plt.savefig(os.path.join(output_dir, '2_sentiment_distribution_pie.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- 3. 高赞评论分析 ---
# (这部分是文本输出，不需要保存图片)
print("\n--- 正在分析高赞评论... ---")


def display_top_comments(df, brand_name, sentiment, n=3):
    print(f"\n--- {brand_name} | {sentiment}情绪 | 点赞数 TOP {n} 评论 ---")
    top_comments = df[(df['brand'] == brand_name) & (df['sentiment'] == sentiment)].nlargest(n, 'like_count')
    if top_comments.empty:
        print("无该类别的评论。")
        return
    for index, row in top_comments.iterrows():
        print(f"👍 点赞数: {row['like_count']}")
        print(f"💬 评论内容: {row['content']}")
        print(f"🤖 分析原因: {row['reason']}\n")


display_top_comments(combined_df, '华为', '正面')
display_top_comments(combined_df, '华为', '负面')
display_top_comments(combined_df, '苹果', '正面')
display_top_comments(combined_df, '苹果', '负面')

# --- 4. 点赞数与情感倾向关系 ---
print("\n--- 正在生成点赞数与情感关系箱形图... ---")
plt.figure(figsize=(14, 8))
sns.boxplot(data=combined_df, x='sentiment', y='like_count', hue='brand', order=sentiment_order, palette='Set2')
plt.yscale('log')
plt.title('不同情感评论的点赞数分布 (对数刻度)', fontsize=16)
plt.xlabel('情感类别', fontsize=12)
plt.ylabel('点赞数 (Log Scale)', fontsize=12)
plt.legend(title='手机品牌')
plt.grid(True, which="both", ls="--", alpha=0.6)
# 保存图片
plt.savefig(os.path.join(output_dir, '3_likes_vs_sentiment_boxplot.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- 5. 评论关键词分析（词云） ---
print("\n--- 正在生成关键词词云... ---")
stopwords = set(
    ['的', '了', '是', '我', '你', '他', '她', '它', '们', '也', '在', '都', '就', '和', '等', '啊', '吗', '吧', '这个',
     '那个', '非常', '很', '有', '没有', '会', '可以', '要', '说', '对', '不', '但', '如果', '因为', '所以', '还是',
     '回复', '评论', '点赞', '支持', '喜欢', '使用', '体验', '感觉', '手机', '品牌', '飓风', '什么', '好', '差', '赞', '差评',
     '就是', '感觉', '真的', '真的很', '真的不错', '真的好', '真的差', 'doge', '哈哈', '哈哈哈', '哈哈哈哈', '哈哈哈哈',
     '觉得', '但是', '金箍', '知道', '可能', '怎么', '为什么', '怎么会', '怎么可能', '怎么会这样', '怎么会这样呢', 'tim',
     '现在', '一下', '不是', '已经', '自己', '吃瓜', '时候', '一个', '视频', '评论区', '评论', '直接', '一直', '只能'
     '这么', '不会', '确实', '不能', '这么', '其他', '其它'])


def generate_wordcloud(text, title, filename):
    if CHINESE_FONT_PATH is None:
        print(f"无法生成词云 '{title}'，因为未找到中文字体。")
        return

    word_list = jieba.lcut(text)
    words = [word for word in word_list if word not in stopwords and len(word) > 1]
    text_processed = " ".join(words)

    if not text_processed:
        print(f"'{title}' 没有足够的内容生成词云。")
        return

    wc = WordCloud(
        font_path=CHINESE_FONT_PATH,  # 使用我们找到的字体路径
        background_color='white',
        width=800,
        height=400,
        max_words=100,
        colormap='viridis'
    ).generate(text_processed)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    # 保存图片
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()


# 华为负面评论内容词云
huawei_negative_content = " ".join(
    combined_df[(combined_df['brand'] == '华为') & (combined_df['sentiment'] == '负面')]['content'])
generate_wordcloud(huawei_negative_content, '华为负面评论内容 关键词词云', '4_wordcloud_huawei_negative_content.png')

# 苹果负面评论内容词云
apple_negative_content = " ".join(
    combined_df[(combined_df['brand'] == '苹果') & (combined_df['sentiment'] == '负面')]['content'])
generate_wordcloud(apple_negative_content, '苹果负面评论内容 关键词词云', '5_wordcloud_apple_negative_content.png')

# 华为评论总计词云
huawei_all_content = " ".join(combined_df[combined_df['brand'] == '华为']['content'])
generate_wordcloud(huawei_all_content, '华为评论内容 关键词词云', '6_wordcloud_huawei_all_content.png')

# 苹果评论总计词云
apple_all_content = " ".join(combined_df[combined_df['brand'] == '苹果']['content'])
generate_wordcloud(apple_all_content, '苹果评论内容 关键词词云', '7_wordcloud_apple_all_content.png')

# 点赞数排名前100的华为评论词云
huawei_top_likes_content = " ".join(
    combined_df[combined_df['brand'] == '华为'].nlargest(100, 'like_count')['content'])
generate_wordcloud(huawei_top_likes_content, '华为点赞数排名前100评论 关键词词云', '8_wordcloud_huawei_top_likes_content.png')

# 点赞数排名前100的苹果评论词云
apple_top_likes_content = " ".join(
    combined_df[combined_df['brand'] == '苹果'].nlargest(100, 'like_count')['content'])
generate_wordcloud(apple_top_likes_content, '苹果点赞数排名前100评论 关键词词云', '9_wordcloud_apple_top_likes_content.png')

print("\n🎉 所有分析和图片生成已完成！")