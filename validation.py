# validation.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import logging
import os
from datetime import datetime
from config import LABEL_DICT
# 配置
MAX_LENGTH = 128
BATCH_SIZE = 64
MODEL_NAME = 'bert-base-uncased'
CHECKPOINT_PATH = './checkpoints/checkpoint_train_epoch_8.pt'  # 指定要加载的checkpoint路径

# 创建日志文件夹
log_dir = 'logs/validation'
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_filename)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)


# 定义数据集类
class TrafficDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, label_dict):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_dict = label_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        payload = str(self.data.iloc[index]['payload'])
        label_str = self.data.iloc[index]['label']
        label = self.label_dict[label_str]
        
        encoding = self.tokenizer(
            payload,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 加载验证数据
val_data = pd.read_csv('./datasets/test_small.csv')
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 创建验证数据集和数据加载器
val_dataset = TrafficDataset(val_data, tokenizer, max_length=MAX_LENGTH, label_dict=LABEL_DICT)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 加载模型
num_labels = len(LABEL_DICT)
# 在加载模型时保持DataParallel模式
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 检查GPU数量，如果有多个GPU，将模型放在DataParallel中
if torch.cuda.device_count() >= 1:
    model = torch.nn.DataParallel(model)

model = model.to(device)

# 加载checkpoint
checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
logger.info(f"加载的模型checkpoint来自 {CHECKPOINT_PATH}")

# 获取每个GPU的显存使用情况
def log_gpu_memory():
    memory_info = []
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / (1024 ** 2)  # 转换为MB
        memory_info.append(f"GPU {i}: {mem:.2f} MB")
    logger.info(" | ".join(memory_info))

# 定义验证函数
def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            logger.info(f"Step [{batch_idx+1}/{len(data_loader)}], Batch Loss: {loss.item():.4f}")

    # 计算验证集的性能指标
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(data_loader)

    return avg_loss, acc, precision, recall, f1

# 在验证集上评估模型
val_loss, val_acc, val_precision, val_recall, val_f1 = eval_model(model, val_loader, device)

# 输出验证结果
logger.info(f"Validation Loss: {val_loss:.4f}")
logger.info(f"Validation Accuracy: {val_acc:.4f}")
logger.info(f"Validation Precision: {val_precision:.4f}")
logger.info(f"Validation Recall: {val_recall:.4f}")
logger.info(f"Validation F1 Score: {val_f1:.4f}")

# print(f"Validation Loss: {val_loss:.4f}")
# print(f"Validation Accuracy: {val_acc:.4f}")
# print(f"Validation Precision: {val_precision:.4f}")
# print(f"Validation Recall: {val_recall:.4f}")
# print(f"Validation F1 Score: {val_f1:.4f}")
