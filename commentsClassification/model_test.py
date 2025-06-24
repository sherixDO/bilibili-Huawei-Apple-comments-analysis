import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
model_path = "/root/autodl-tmp/brand_comment_classifier_v2"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# 预测函数
def predict_brand(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return "apple" if pred == 0 else "huawei"

# 交互式预测
if __name__ == "__main__":
    print("请输入评论（输入'quit'退出）：")
    while True:
        comment = input("评论：")
        if comment.lower() == 'quit':
            break
        prediction = predict_brand(comment)
        print(f"预测品牌：{prediction}\n")