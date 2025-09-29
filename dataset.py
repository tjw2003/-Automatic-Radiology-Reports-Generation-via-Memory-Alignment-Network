import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer

class RadiologyReportDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_len=100, transform=None):
        # 1. 读取 JSON 文件
        with open(json_file, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        # 2. 保存分词器
        self.tokenizer = tokenizer

        # 3. 设置文本最大长度
        self.max_len = max_len

        # 4. 图像预处理流程
        # 如果用户没传 transform，就用默认的
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),   # 调整大小为 224x224
            transforms.ToTensor(),          # 转换成 Tensor，范围 [0,1]
            transforms.Normalize(           # 按 ImageNet 的均值方差归一化
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def __len__(self):
        """返回数据集的大小"""
        return len(self.records)

    def __getitem__(self, idx):
        """
        根据索引 idx 返回一条数据
        :param idx: 第 idx 条数据
        :return: (图像张量, input_ids, attention_mask, token_type_ids, labels)
        """
        record = self.records[idx]

        # --------- 图像部分 ---------
        # 打开 PNG 图片，并转为 RGB (3 通道)
        image = Image.open(record["image_path"]).convert("RGB")
        # 进行预处理（resize + toTensor + normalize）
        image = self.transform(image)

        # --------- 文本部分 ---------
        # 用 tokenizer 把报告文本转成 token id
        enc = self.tokenizer(
            record["report_text"],           # 输入文本
            padding="max_length",            # 不够 max_len 的地方用 [PAD] 补齐
            truncation=True,                 # 超过 max_len 的部分截断
            max_length=self.max_len,         # 最大长度
            return_tensors="pt"              # 返回 PyTorch Tensor 格式
        )

        # 取出编码后的结果 (shape: [1, max_len])，去掉 batch 维度
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # 某些 tokenizer 可能没有 token_type_ids，就用全 0 替代
        token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

        # --------- 构造 labels ---------
        # 训练时我们需要 labels (目标输出)
        # 把 input_ids 拷贝一份
        labels = input_ids.clone()
        # 把 padding 的地方标记为 -100，loss 会忽略这些位置
        labels[labels == self.tokenizer.pad_token_id] = -100

        return image, input_ids, attention_mask, token_type_ids, labels
