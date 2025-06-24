import requests
import time
import csv
import sqlite3
import os
import random

# --- 配置区 ---
# 1. 在这里填入你想爬取的视频BVID列表
BVID_LIST = [
    #"BV11CdHYEEAB",
    #"BV1uKS3YVEPd",
    #"BV1NadLYYEny",
    #"BV1fF2dYVEJQ",
    #"BV1yXtjeSEDZ",
    #"BV1k2C3YpEfJ",
    #"BV15pkjYHEmP",
    "BV1JoCpYLECF",
    "BV1j6iYYHEYG"
]

# 2. 在这里填入你从浏览器获取的SESSDATA
# SESSDATA = "你的SESSDATA值"
SESSDATA = ""  # 替换成你的SESSDATA

# 3. 输出文件名配置
CSV_FILENAME = 'bilibili_comments_huawei.csv'
DB_FILENAME = 'bilibili_comments_huawei.db'

# 4. 反爬虫延时（秒），每次请求后随机等待的时间范围
SLEEP_RANGE = (1, 3)
# --- 配置区结束 ---

# API URL
API_URL = "https://api.bilibili.com/x/v2/reply"

# 请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
    'Cookie': f'SESSDATA={SESSDATA}'
}

# 禁用requests的SSL警告
requests.packages.urllib3.disable_warnings()


def setup_database():
    """初始化SQLite数据库和表"""
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS comments
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       bvid
                       TEXT
                       NOT
                       NULL,
                       comment_id
                       INTEGER
                       UNIQUE
                       NOT
                       NULL,
                       parent_id
                       INTEGER
                       NOT
                       NULL,
                       user_name
                       TEXT,
                       user_mid
                       INTEGER,
                       publish_time
                       INTEGER,
                       like_count
                       INTEGER,
                       content
                       TEXT,
                       is_up_liked
                       BOOLEAN
                   )
                   ''')
    conn.commit()
    return conn


def setup_csv():
    """初始化CSV文件并写入表头"""
    file_exists = os.path.isfile(CSV_FILENAME)
    # 以追加模式打开，避免覆盖
    csv_file = open(CSV_FILENAME, 'a', newline='', encoding='utf-8-sig')
    writer = csv.writer(csv_file)
    if not file_exists or os.path.getsize(CSV_FILENAME) == 0:
        writer.writerow(
            ['bvid', 'comment_id', 'parent_id', 'user_name', 'user_mid', 'publish_time', 'like_count', 'content',
             'is_up_liked'])
    return csv_file, writer


def save_to_db(cursor, comment_data):
    """将一条评论数据存入数据库"""
    try:
        cursor.execute('''
                       INSERT INTO comments (bvid, comment_id, parent_id, user_name, user_mid, publish_time, like_count,
                                             content, is_up_liked)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', (
                           comment_data['bvid'],
                           comment_data['comment_id'],
                           comment_data['parent_id'],
                           comment_data['user_name'],
                           comment_data['user_mid'],
                           comment_data['publish_time'],
                           comment_data['like_count'],
                           comment_data['content'],
                           comment_data['is_up_liked']
                       ))
    except sqlite3.IntegrityError:
        # 如果comment_id已存在，则忽略
        pass
    except Exception as e:
        print(f"  [DB Error] 存入数据库失败: {e}")


def save_to_csv(writer, comment_data):
    """将一条评论数据写入CSV"""
    try:
        writer.writerow([
            comment_data['bvid'],
            comment_data['comment_id'],
            comment_data['parent_id'],
            comment_data['user_name'],
            comment_data['user_mid'],
            comment_data['publish_time'],
            comment_data['like_count'],
            comment_data['content'],
            comment_data['is_up_liked']
        ])
    except Exception as e:
        print(f"  [CSV Error] 写入CSV失败: {e}")


def process_replies(replies, bvid, parent_id, db_cursor, csv_writer):
    """递归处理评论和子评论"""
    if not replies:
        return 0

    count = 0
    for reply in replies:
        # UP主觉得很赞的标志在 up_action.like 中
        is_up_liked = reply.get('up_action', {}).get('like', False)

        comment_data = {
            'bvid': bvid,
            'comment_id': reply['rpid'],
            'parent_id': parent_id,
            'user_name': reply['member']['uname'],
            'user_mid': reply['member']['mid'],
            'publish_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(reply['ctime'])),
            'like_count': reply['like'],
            'content': reply['content']['message'],
            'is_up_liked': is_up_liked
        }

        save_to_db(db_cursor, comment_data)
        save_to_csv(csv_writer, comment_data)
        count += 1

        # 递归处理楼中楼回复
        if 'replies' in reply and reply['replies']:
            count += process_replies(reply['replies'], bvid, reply['rpid'], db_cursor, csv_writer)

    return count


def get_comments_for_video(bvid, db_conn, csv_writer):
    """获取单个视频的所有评论"""
    print(f"\n--- 开始采集视频 {bvid} 的评论 ---")
    page_num = 1
    total_comments_processed = 0

    db_cursor = db_conn.cursor()

    while True:
        params = {
            'type': 1,  # 1 代表视频评论
            'oid': bvid,
            'sort': 0,  # 0 按时间，1 按点赞，2 按回复
            'pn': page_num,
            'ps': 20  # 每页数量
        }

        try:
            print(f"[Page {page_num}] 正在请求API...")
            wait = random.uniform(1, 3) * 5
            print("[Info] 等待时间: {:.2f}秒".format(wait))
            time.sleep(wait)
            response = requests.get(API_URL, params=params, headers=HEADERS, verify=False, timeout=10)
            response.raise_for_status()  # 如果请求失败则抛出异常

            data = response.json()

            if data['code'] != 0:
                print(f"[Error] API返回错误: {data['message']} (Code: {data['code']})")
                if data['code'] == 12002:  # 评论区已关闭
                    print("评论区已关闭，停止采集该视频。")
                break

            # 检查是否有评论
            replies = data.get('data', {}).get('replies')
            if not replies:
                print("[Info] 本页没有评论，采集完成。")
                break

            # 处理主评论和它们的子评论
            comments_on_page = process_replies(replies, bvid, 0, db_cursor, csv_writer)
            total_comments_processed += comments_on_page
            print(f"[Page {page_num}] 处理了 {comments_on_page} 条评论。")

            # 提交数据库事务
            db_conn.commit()

            page_num += 1
            # 随机延时
            sleep_time = random.uniform(SLEEP_RANGE[0], SLEEP_RANGE[1])
            time.sleep(sleep_time)

        except requests.exceptions.RequestException as e:
            print(f"[Network Error] 请求失败: {e}，将在10秒后重试...")
            time.sleep(10)
        except Exception as e:
            print(f"[Fatal Error] 发生未知错误: {e}")
            break

    print(f"--- 视频 {bvid} 采集结束，共处理 {total_comments_processed} 条评论 ---")


if __name__ == '__main__':
    # 初始化输出文件
    db_connection = setup_database()
    csv_file, csv_writer = setup_csv()

    print("=== Bilibili评论采集器启动 ===")

    try:
        for bvid in BVID_LIST:
            get_comments_for_video(bvid, db_connection, csv_writer)
    finally:
        # 确保文件和数据库连接被关闭
        db_connection.close()
        csv_file.close()
        print("\n=== 所有任务完成，程序退出。 ===")
        print(f"数据已保存至 {CSV_FILENAME} 和 {DB_FILENAME}")
