import json
import random
import os

# 输入输出路径
input_file = "./records.json"
output_dir = "./dataset_split"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取记录
with open(input_file, "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"总共有 {len(records)} 条 图像-报告 对")

# 打乱数据
random.shuffle(records)

# 划分比例
n = len(records)
train_end = int(0.7 * n)
val_end = int(0.8 * n)

train_set = records[:train_end]
val_set = records[train_end:val_end]
test_set = records[val_end:]

# 保存结果
with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
    json.dump(train_set, f, ensure_ascii=False, indent=2)
with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as f:
    json.dump(val_set, f, ensure_ascii=False, indent=2)
with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
    json.dump(test_set, f, ensure_ascii=False, indent=2)

print(f"划分完成: 训练集 {len(train_set)} | 验证集 {len(val_set)} | 测试集 {len(test_set)}")
