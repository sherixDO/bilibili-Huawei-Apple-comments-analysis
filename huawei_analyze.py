# huawei_analyze_robust.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import re
import json
import logging
import sys
import os
import argparse
from typing import List, Dict, Tuple



# =======================================================================
# 步骤 0: 配置日志
# =======================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =======================================================================
# 【优化版】更直接、更强调格式的Prompt模板
# =======================================================================
NEW_PROMPT_TEMPLATE = """你是一个专门用于情感分析的API。你的唯一工作是接收一段用户评论，然后返回一个严格的JSON对象。绝对不允许输出任何JSON以外的文字。

JSON对象必须包含以下两个键:
1. `sentiment`: 字符串类型，值必须是 "正面"、"负面" 或 "中性" 之一。
2. `reason`: 字符串类型，简明扼要地解释你做出判断的核心理由。

---
评论: 『今年最伟大的创新就是这个新颜色，绝了。』
```json
{{
  "reason": "评论表面上在夸赞'新颜色'，但暗示除了颜色之外毫无创新。这是对产品缺乏实质性升级的典型讽刺，真实意图是负面的。",
  "sentiment": "负面"
}}```
---
评论: 『不坑穷人是最大的优点，没了...』
```json
{{
  "reason": "这是一个网络梗，意思是产品定价太高，穷人买不起，所以'坑'不到。这是对高定价的强烈讽刺，真实意图是负面的。",
  "sentiment": "负面"
}}```
---
现在，处理以下输入。只返回JSON代码块。

输入: ```{text_to_analyze}```
输出:
```json
"""



def initialize_model(model_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """加载并初始化分词器和模型"""
    logging.info(f"正在从 '{model_path}' 加载模型，请稍候...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
        logging.info("模型加载完成！")
        return tokenizer, model
    except Exception as e:
        logging.error(f"模型加载失败: {e}", exc_info=True)
        sys.exit(1)



def get_analysis_batch(texts: List[str], tokenizer, model) -> List[Dict[str, str]]:
    """
    【最终修正、完整且健壮的版本】
    使用LLM批量分析文本，包含完整的模型调用和响应解码流程，
    并用正则表达式强行提取第一个JSON对象进行解析。
    """
    if not texts:
        return []

    # 确保你使用的是最强的Prompt模板
    prompts = [NEW_PROMPT_TEMPLATE.format(text_to_analyze=str(t)[:256]) for t in texts]

    try:
        ### 关键修复：确保模型生成和解码的逻辑完整 ###
        with torch.no_grad():
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(
                model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        # 解码得到 responses 列表，这里是之前缺失的部分
        responses = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        analysis_results = []
        for i, res in enumerate(responses):
            result = {'sentiment': '中性', 'reason': '模型未能正确解析输出。'}
            try:
                # 使用正则表达式，只提取第一个出现的、结构上像JSON的块
                json_match = re.search(r'{\s*".*?"\s*:.*?}', res, re.DOTALL)

                if not json_match:
                    raise json.JSONDecodeError("正则表达式未在模型输出中找到任何JSON对象。", res, 0)

                json_str = json_match.group(0)

                data = json.loads(json_str)

                sentiment = data.get("sentiment")
                if sentiment not in ["正面", "负面", "中性"]:
                    logging.warning(f"情感值 '{sentiment}' 无效，已修正为'中性'。原始文本: '{texts[i][:50]}...'")
                    sentiment = "中性"

                result = {'sentiment': sentiment, 'reason': data.get("reason", "无理由。")}

            except (json.JSONDecodeError, AttributeError) as e:
                logging.warning(
                    f"JSON解析失败。原始文本: '{texts[i][:50]}...'. 错误: {e}. 模型原始输出: '{res[:150]}...'")
                result['reason'] = f"JSON解析失败，原始输出: {res[:100]}..."

            analysis_results.append(result)
        return analysis_results

    except Exception as e:
        # 这个 except 块现在只会在模型生成等更底层的步骤出错时被触发
        logging.error(f"处理批次时发生严重错误: {e}", exc_info=True)
        return [{'sentiment': '中性', 'reason': f'批处理错误: {str(e)}'}] * len(texts)

def generate_summary_report(df_analyzed: pd.DataFrame, brand_name: str, high_like_threshold: int, like_col: str,
                            content_col: str):
    """根据分析完成的DataFrame生成并打印汇总报告"""
    print("\n\n" + "=" * 25 + " 分析结果汇总 " + "=" * 25)

    # 1. 高赞评论详情
    if like_col in df_analyzed.columns:
        high_like_comments = df_analyzed[df_analyzed[like_col] > high_like_threshold]
        if not high_like_comments.empty:
            print(f"\n--- {brand_name} 高赞评论情感分析详情 (点赞 > {high_like_threshold}) ---")
            for _, row in high_like_comments.sort_values(by=like_col, ascending=False).head(20).iterrows():
                content = str(row[content_col]).strip().replace('\n', ' ')
                print(f"点赞: {int(row[like_col])} | 情感: [{row['sentiment']}]")
                print(f"  评论: {content[:100]}{'...' if len(content) > 100 else ''}")
                print(f"  模型理由: {row['reason']}")
                print("-" * 20)
        else:
            print(f"\n--- 未找到点赞数超过 {high_like_threshold} 的高赞评论 ---")

    # 2. 总体情感统计
    print(f"\n--- {brand_name} 总体情感分析统计 ---")
    total_analyzed = len(df_analyzed)
    if total_analyzed > 0:
        sentiment_counts = df_analyzed['sentiment'].value_counts()
        results = {
            '正面': sentiment_counts.get('正面', 0),
            '负面': sentiment_counts.get('负面', 0),
            '中性': sentiment_counts.get('中性', 0)
        }
        print(f"分析总数: {total_analyzed} 条评论")
        print(f"  [+] 正面评价: {results['正面']} 条 ({results['正面'] / total_analyzed:.2%})")
        print(f"  [-] 负面评价: {results['负面']} 条 ({results['负面'] / total_analyzed:.2%})")
        print(f"  [~] 中性评价: {results['中性']} 条 ({results['中性'] / total_analyzed:.2%})")
    else:
        print("未处理任何评论。")
    print("\n" + "=" * 60)


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用大语言模型分析社交媒体评论情感。")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/DeepSeek-R1-Qwen2.5-Math-14B",
                        help="大语言模型路径。")
    parser.add_argument("--input-csv", type=str, default="huawei.csv", help="待分析的输入CSV文件。")
    parser.add_argument("--output-csv", type=str, default="huawei_analyzed_results.csv",
                        help="保存分析结果的输出CSV文件。")
    parser.add_argument("--brand-name", type=str, default="华为", help="分析报告中显示的品牌名。")
    parser.add_argument("--like-col", type=str, default="like_count", help="CSV中表示点赞数的列名。")
    parser.add_argument("--content-col", type=str, default="content", help="CSV中表示评论内容的列名。")
    parser.add_argument("--batch-size", type=int, default=32, help="进行LLM推理的批大小。")
    parser.add_argument("--like-threshold", type=int, default=100, help="定义高赞评论的点赞数阈值。")
    parser.add_argument("--test", action='store_true', help="开启测试模式，仅处理前200条数据。")
    parser.add_argument("--resume", action='store_true', help="从上次中断的地方继续分析，跳过已在输出文件中存在的评论。")
    return parser.parse_args()


# =======================================================================
def main():
    """主执行函数"""
    args = parse_arguments()

    # --- 1. 【健壮性优化】确定输出路径并加载数据 ---
    # 定义基础输出目录
    output_dir = "analysis_results"

    # 组合最终的CSV文件路径
    # 如果用户在命令行提供了包含目录的路径，则以用户为准
    if os.path.dirname(args.output_csv):
        final_csv_path = args.output_csv
        # 更新输出目录为用户指定的目录
        output_dir = os.path.dirname(final_csv_path)
    else:
        # 否则，使用默认的 "analysis_results" 目录
        final_csv_path = os.path.join(output_dir, args.output_csv)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"所有输出将被保存到: '{output_dir}/'")

    # 加载数据，以最终输出文件为准进行断点续传
    if args.resume and os.path.exists(final_csv_path):
        logging.info(f"检测到 '--resume' 标志，将从 {final_csv_path} 继续...")
        try:
            df = pd.read_csv(final_csv_path)
        except Exception as e:
            logging.error(f"读取输出文件 {final_csv_path} 失败: {e}", exc_info=True)
            sys.exit(1)
    else:
        logging.info(f"将从头开始分析 {args.input_csv}...")
        try:
            df = pd.read_csv(args.input_csv)
        except FileNotFoundError:
            logging.error(f"错误：输入文件未找到！路径: {args.input_csv}")
            sys.exit(1)

    # --- 2. 预处理数据列 ---
    if args.content_col not in df.columns:
        logging.error(f"错误：未找到内容列 '{args.content_col}'。")
        sys.exit(1)
    if 'sentiment' not in df.columns:
        df['sentiment'] = None
    if 'reason' not in df.columns:
        df['reason'] = None
    if args.like_col not in df.columns:
        logging.warning(f"警告：未找到点赞列 '{args.like_col}'，将无法进行高赞分析。")
        df[args.like_col] = 0
    else:
        df[args.like_col] = pd.to_numeric(df[args.like_col], errors='coerce').fillna(0)

    # --- 3. 确定待分析数据 ---
    df_to_process = df[df['sentiment'].isnull()].copy()
    if args.test:
        logging.info(f"---已开启测试模式，将最多分析 {min(200, len(df_to_process))} 条数据---")
        df_to_process = df_to_process.head(200)

    # --- 4. 执行分析 ---
    if not df_to_process.empty:
        tokenizer, model = initialize_model(args.model_path)
        logging.info(f"--- 开始分析 {len(df_to_process)} 条新评论 (批大小: {args.batch_size}) ---")

        # 定义定期保存的频率 (每处理10个批次保存一次)
        save_interval = 10
        total_batches = (len(df_to_process) + args.batch_size - 1) // args.batch_size

        for i in tqdm(range(0, len(df_to_process), args.batch_size), desc=f"分析 {args.brand_name} 数据中"):
            batch_df = df_to_process.iloc[i:i + args.batch_size]
            valid_texts, original_indices = [], []

            for index, row in batch_df.iterrows():
                text = row[args.content_col]
                if isinstance(text, str) and text.strip():
                    valid_texts.append(text)
                    original_indices.append(index)
                else:
                    df.loc[index, 'sentiment'] = '中性'
                    df.loc[index, 'reason'] = '评论内容为空。'

            if valid_texts:
                analysis_batch = get_analysis_batch(valid_texts, tokenizer, model)
                for j, analysis in enumerate(analysis_batch):
                    idx = original_indices[j]
                    df.loc[idx, 'sentiment'] = analysis['sentiment']
                    df.loc[idx, 'reason'] = analysis['reason']

                # 【实时反馈】打印当前批次的第一个样本结果
                if analysis_batch:
                    first_text = valid_texts[0]
                    first_analysis = analysis_batch[0]
                    print("\n" + "-" * 15 + f" 批次 {i // args.batch_size + 1}/{total_batches} 样本预览 " + "-" * 15)
                    print(f"  评论: {first_text[:80].replace(chr(10), ' ')}...")
                    print(f"  情感: [{first_analysis['sentiment']}]")
                    print(f"  理由: {first_analysis['reason']}")
                    print("-" * 60)

            # 【定期保存】每隔 save_interval 个批次，或在最后一个批次结束后，保存一次
            current_batch_num = i // args.batch_size
            if (current_batch_num + 1) % save_interval == 0 or (current_batch_num + 1) == total_batches:
                try:
                    df.to_csv(final_csv_path, index=False, encoding='utf-8-sig')
                    logging.info(f"进度 {current_batch_num + 1}/{total_batches} - 中间结果已保存到 {final_csv_path}")
                except Exception as e:
                    logging.error(f"中间结果保存失败: {e}")
    else:
        logging.info("没有新的评论需要分析。")

    # --- 5. 最终保存并报告 ---
    final_df_to_report = df[df['sentiment'].notna()]
    if not final_df_to_report.empty:
        try:
            # 最后再完整地保存一次，确保万无一失
            df.to_csv(final_csv_path, index=False, encoding='utf-8-sig')
            logging.info(f"全部分析结果已成功保存到: {final_csv_path}")
        except Exception as e:
            logging.error(f"保存结果到CSV失败: {e}")

        # 将报告保存到文件
        report_path = os.path.splitext(final_csv_path)[0] + "_summary_report.txt"
        logging.info(f"正在生成最终分析报告...")
        original_stdout = sys.stdout
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                sys.stdout = f
                generate_summary_report(final_df_to_report, args.brand_name, args.like_threshold, args.like_col,
                                        args.content_col)
            logging.info(f"分析报告已成功保存到: {report_path}")
        except Exception as e:
            logging.error(f"保存报告到文件失败: {e}")
        finally:
            sys.stdout = original_stdout

        # 在控制台也打印一次最终报告
        generate_summary_report(final_df_to_report, args.brand_name, args.like_threshold, args.like_col,
                                args.content_col)
    else:
        logging.warning("最终没有可报告或保存的分析结果。")

    logging.info("所有任务完成！")


if __name__ == "__main__":
    main()
