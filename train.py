"""
train.py

完整训练脚本。假设项目目录包含：
- dataset.py (RadiologyReportDataset)
- reportgen_model.py (上面的模型实现)
- dataset_split/train.json / val.json / test.json

运行:
    python train.py
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from dataset import RadiologyReportDataset
from reportgen_model import ReportGenModel
from tqdm import tqdm
import random
import numpy as np

# 固定随机种子（便于复现）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_loaders(tokenizer, batch_size=8, max_len=100, num_workers=4):
    train_dataset = RadiologyReportDataset("./dataset_split/train.json", tokenizer, max_len=max_len)
    val_dataset   = RadiologyReportDataset("./dataset_split/val.json", tokenizer, max_len=max_len)
    test_dataset  = RadiologyReportDataset("./dataset_split/test.json", tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc="Train", leave=False)
    for batch in loop:
        images, input_ids, attention_mask, token_type_ids, labels = [x.to(device) for x in batch]

        optimizer.zero_grad()
        loss, logits = model(images, input_ids, attention_mask, token_type_ids, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    loop = tqdm(dataloader, desc="Val", leave=False)
    for batch in loop:
        images, input_ids, attention_mask, token_type_ids, labels = [x.to(device) for x in batch]
        loss, logits = model(images, input_ids, attention_mask, token_type_ids, labels)
        total_loss += loss.item()
        loop.set_postfix(val_loss=loss.item())
    return total_loss / len(dataloader)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 超参（可修改）
    batch_size = 8         # 若显存小，调小
    max_len = 100
    num_workers = 4        # Windows 上若报错可改为 0
    epochs = 5
    learning_rate = 1e-4
    warmup_steps = 100

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # dataloaders
    train_loader, val_loader, test_loader = get_loaders(tokenizer, batch_size=batch_size, max_len=max_len, num_workers=num_workers)

    # model
    model = ReportGenModel(bert_model_name='bert-base-uncased', d_model=768, memory_size=100, ma_heads=4)
    model.to(device)

    # optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # checkpoint dir
    os.makedirs("checkpoints", exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, epochs+1):
        print(f"==== Epoch {epoch}/{epochs} ====")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

        # 保存模型
        ckpt_path = f"checkpoints/reportgen_epoch{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print("Saved checkpoint:", ckpt_path)

        # 保存最优模型
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoints/best_reportgen.pth")
            print("Saved best model.")

    # 测试集（简单 loss 评估）
    test_loss = validate(model, test_loader, device)
    print("Final Test Loss:", test_loss)

if __name__ == "__main__":
    main()
