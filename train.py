# train.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
import logging
import os
from datetime import datetime
from config import LABEL_DICT
import gc

import warnings
warnings.filterwarnings('ignore')

# 创建日志文件夹
log_dir = 'logs/train'
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(
    log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_filename)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

# 配置
MAX_LENGTH = 128
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 2e-5
MODEL_NAME = 'bert-base-uncased'
CHECKPOINT_DIR = 'checkpoints/'

logger.info('''
    Parameters:
    MAX_LENGTH = {max_length}
    BATCH_SIZE = {batch_size}
    EPOCHS = {epochs}
    LEARNING_RATE = {learning_rate}
    MODEL_NAME = {model_name}
    CHECKPOINT_DIR = {checkpoint_dir}
    '''.format(max_length=MAX_LENGTH, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE, model_name=MODEL_NAME, checkpoint_dir=CHECKPOINT_DIR))

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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


# 加载训练数据
logger.info('load data for training...')
train_data = pd.read_csv('./datasets/train_small.csv')
logger.info('load pretrained model...')
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

logger.info('setting up dataloader...')
num_labels = len(LABEL_DICT)
train_dataset = TrafficDataset(
    train_data, tokenizer, max_length=MAX_LENGTH, label_dict=LABEL_DICT)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 将模型加载到多个GPU上
logger.info('load model to gpus...')
device_ids = [0, 1]  # 使用GPU 0 和 GPU 1
device_ids = [0]  # 使用GPU 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels)
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=device_ids)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 使用checkpoints恢复训练
logger.info('load checkpoint if have')
# checkpoint = torch.load('./checkpoints/checkpoint_train_epoch_2.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = int(checkpoint['epoch'])
# logger.info(f'checkpoint start epoch: {start_epoch}')


# 定义训练函数
def train_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.mean().item()

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        # 日志记录
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx +
                    1}/{len(data_loader)}], Loss: {loss.mean().item():.4f}")
    return total_loss / len(data_loader)

# 保存checkpoint函数


def save_checkpoint(model, optimizer, epoch):
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, f"checkpoint_train_epoch_{epoch+1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    logging.info(f"Checkpoint saved at {checkpoint_path}")


# 开始训练
logger.info('training begin')
for epoch in range(EPOCHS):
    # if start_epoch != 0:
    #    if epoch != start_epoch:
    #        continue
    logger.info(f'Starting Epoch {epoch + 1}/{EPOCHS}')
    train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
    logger.info(
        f"Epoch [{epoch + 1}/{EPOCHS}] completed with Average Loss: {train_loss:.4f}")

    # 每轮迭代后保存checkpoint
    save_checkpoint(model, optimizer, epoch)
    logger.info(f'Checkpoint for epoch {epoch + 1} saved')
    # 释放显存
    torch.cuda.empty_cache()
    gc.collect()
