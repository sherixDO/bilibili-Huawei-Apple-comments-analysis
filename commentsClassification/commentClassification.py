# -*- coding: utf-8 -*-
# 评论品牌识别模型训练 + 推理功能

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
import numpy as np
from sentence_transformers import SentenceTransformer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
df = pd.concat([pd.read_csv('/root/apple.csv').assign(label=0), pd.read_csv('/root/huawei.csv').assign(label=1)])

# 文本嵌入用于SMOTE
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = embedder.encode(df['content'].tolist(), convert_to_numpy=True)
smote = SMOTE(random_state=42)
embeddings_res, labels_res = smote.fit_resample(embeddings, df['label'])

# 构造新的DataFrame
df_resampled = pd.DataFrame({
    'text': [df['content'].iloc[i] if i < len(df) else '' for i in range(len(labels_res))],
    'label': labels_res
})

# 计算类权重
class_weights = torch.tensor([len(df_resampled[df_resampled['label'] == 1]) / len(df_resampled),
                              len(df_resampled[df_resampled['label'] == 0]) / len(df_resampled)]).to(device)
print("Class weights:", class_weights)

# 数据拆分
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_resampled['text'].tolist(), df_resampled['label'].tolist(), test_size=0.2, random_state=42
)

# 构建数据集
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


class CommentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = CommentDataset(train_texts, train_labels)
test_dataset = CommentDataset(test_texts, test_labels)

# 模型初始化
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model.to(device)

# 超参数手动搜索
param_grid = [
    {'batch_size': 16, 'max_length': 64, 'patience': 3},
    {'batch_size': 32, 'max_length': 128, 'patience': 5},
    {'batch_size': 64, 'max_length': 256, 'patience': 7}
]

best_score = float('-inf')
best_params = None

for params in param_grid:
    print(f"Testing params: {params}")
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler()


    class AdjustedDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=params['max_length'])
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)


    train_dataset_adjusted = AdjustedDataset(train_texts, train_labels)
    train_loader = DataLoader(train_dataset_adjusted, batch_size=params['batch_size'], shuffle=True)

    best_loss = float('inf')
    trigger_times = 0

    model.train()
    for epoch in range(30):
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        epoch_loss = 0
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                outputs = model(**batch)
                loss = criterion(outputs.logits, batch['labels'])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= params['patience']:
                break

    score = -best_loss
    if score > best_score:
        best_score = score
        best_params = params
    print(f"Params {params} - Best Loss: {best_loss}")

print("Best parameters:", best_params)

# 使用最佳参数重新训练
train_dataset = CommentDataset(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
scaler = GradScaler()

best_loss = float('inf')
patience, trigger_times = best_params['patience'], 0

model.train()
for epoch in range(30):
    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    epoch_loss = 0
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast():
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        trigger_times = 0
        model.save_pretrained("/root/autodl-tmp/brand_comment_classifier_v2")
        tokenizer.save_pretrained("/root/autodl-tmp/brand_comment_classifier_v2")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# 模型评估
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print(classification_report(true_labels, predictions, target_names=['apple', 'huawei'], zero_division=0))


# 预测函数
def predict_brand(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=best_params['max_length'])
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return "apple" if pred == 0 else "huawei"


# 示例预测
example1 = "这款手机真的拍照效果很棒，系统运行流畅"
example2 = "苹果手机系统真的和流畅"
example3 = "手机高价低配，性价比真的很低"
print("评论：", example1)
print("预测品牌：", predict_brand(example1, tokenizer, model))
print("评论：", example2)
print("预测品牌：", predict_brand(example2, tokenizer, model))
print("评论：", example3)
print("预测品牌：", predict_brand(example3, tokenizer, model))