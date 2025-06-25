import re
import pandas as pd
from sqlalchemy import create_engine
import jieba


def load_stopwords(filepath):
    """从文件加载停用词，返回一个集合"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = {line.strip() for line in f if line.strip()}
        print(f"成功加载 {len(stopwords)} 个停用词。")
        return stopwords
    except FileNotFoundError:
        print(f"警告: 停用词文件 '{filepath}' 未找到，将不进行停用词移除。")
        return set()


def clean_and_segment_text(text, stopwords_set=None):
    """
    一个通用的文本清洗和分词函数：
    1. 移除表情符号和图形符号。
    2. 移除方括号内容。
    3. 清洗非必要字符。
    4. 过滤纯数字评论。
    5. 使用jieba分词并移除停用词。
    """
    if not isinstance(text, str):
        return ""

    try:
        emoji_pattern = re.compile(u'[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u2B55\U0001F900-\U0001F9FF\U0001F1E0-\U0001F1FF]')
    except re.error:
        emoji_pattern = re.compile(u'[\uD83C[\uDF00-\uDFFF]|\uD83D[\uDC00-\uDE4F\uDE80-\uDEFF]|[\u2600-\u2B55]|\uD83E[\uDD00-\uDDFF]|\uD83C[\uDDE0-\uDDFF]]')
    text = emoji_pattern.sub(r' ', text)

    text = re.sub(r'[\[［].*?[]］]', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()

    if re.fullmatch(r'[\d\s]*', text) or not text:
        return ""

    word_list = jieba.cut(text, cut_all=False)

    if stopwords_set:
        filtered_words = [word for word in word_list if word.strip() and word not in stopwords_set]
    else:
        filtered_words = [word for word in word_list if word.strip()]

    return " ".join(filtered_words)


def clean_dataframe(df, stopwords_set=None):
    """
    清洗DataFrame的函数：
    1. 移除指定的列。
    2. 对'comment'列进行文本清洗、分词和停用词移除。
    3. 删除'comment'清洗后为空的行。
    4. **新增**：按点赞数（like_count）由高到低排序。
    """
    df_cleaned = df.copy()
    columns_to_remove = [
        'bvid', 'comment_id', 'parent_id', 'user_name',
        'user_mid', 'publish_time', 'is_up_liked'
    ]
    existing_columns_to_remove = [col for col in columns_to_remove if col in df_cleaned.columns]

    if existing_columns_to_remove:
        df_cleaned.drop(columns=existing_columns_to_remove, inplace=True)
        print(f"已移除列: {', '.join(existing_columns_to_remove)}")
    else:
        print("未找到需要移除的指定列。")

    if 'comment' in df_cleaned.columns:
        original_rows = len(df_cleaned)

        df_cleaned['comment'] = df_cleaned['comment'].apply(
            lambda x: clean_and_segment_text(x, stopwords_set=stopwords_set)
        )
        print("已对 'comment' 列进行文本清洗、分词和停用词移除。")

        df_cleaned = df_cleaned[df_cleaned['comment'].str.strip() != ''].copy()
        new_rows = len(df_cleaned)
        print(f"移除了 {original_rows - new_rows} 个因清洗后内容为空或无意义的行。")

        # **关键优化**：按 'like_count' 列排序
        if 'like_count' in df_cleaned.columns:
            print("按 'like_count' 列（点赞数）由高到低排序。")
            # 确保 'like_count' 是数值类型，以防排序出错
            df_cleaned['like_count'] = pd.to_numeric(df_cleaned['like_count'], errors='coerce').fillna(0).astype(int)
            df_cleaned.sort_values(by='like_count', ascending=False, inplace=True)
        else:
            print("警告: 未找到 'like_count' 列，跳过按点赞数排序。")

        # 重置索引，使之在排序后保持连续
        df_cleaned.reset_index(drop=True, inplace=True)
    else:
        print("警告: DataFrame中未找到 'comment' 列，跳过文本处理。")

    return df_cleaned


def process_csv_file(input_path, output_path, stopwords_set=None):
    """读取、清洗并保存CSV文件"""
    print(f"\n--- 开始处理CSV文件: {input_path} ---")
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
        cleaned_df = clean_dataframe(df, stopwords_set=stopwords_set)
        cleaned_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"处理完成，清洗后的CSV文件已保存至: '{output_path}'")
    except FileNotFoundError:
        print(f"错误: 文件 '{input_path}' 未找到。")
    except Exception as e:
        print(f"处理CSV文件时发生错误: {e}")
    print("--- CSV文件处理结束 ---")





if __name__ == '__main__':
    jieba.setLogLevel(jieba.logging.INFO)

    # --- 1. 加载停用词 ---
    stopwords_path = '/Users/macbook/Downloads/cn_stopwords.txt'
    stopwords = load_stopwords(stopwords_path)

    # --- 2. 处理CSV文件 ---
    csv_input_file = '/Users/macbook/Downloads/bilibili_comments_huawei.csv'
    csv_output_file = 'huawei-1.csv'
    process_csv_file(csv_input_file, csv_output_file, stopwords_set=stopwords)

    print("\n" + "=" * 50 + "\n")

    # --- 3. 处理数据库文件 (示例) ---
    # db_file_path = 'your_database.db'
    # source_table = 'your_table_name'
    # destination_table = 'cleaned_sorted_' + source_table
    # process_db_file(db_file_path, source_table, destination_table, stopwords_set=stopwords)