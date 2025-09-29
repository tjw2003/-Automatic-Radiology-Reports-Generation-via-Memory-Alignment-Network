from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import RadiologyReportDataset

def main():
    # 初始化 tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 加载训练集
    train_dataset = RadiologyReportDataset("./dataset_split/train.json", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    # 加载验证集
    val_dataset = RadiologyReportDataset("./dataset_split/val.json", tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 加载测试集
    test_dataset = RadiologyReportDataset("./dataset_split/test.json", tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 打印训练集的一个 batch
    images, input_ids, attention_mask, token_type_ids, labels = next(iter(train_loader))
    print("Train batch:")
    print("images:", images.shape)
    print("input_ids:", input_ids.shape)
    print("attention_mask:", attention_mask.shape)
    print("labels:", labels.shape)

    # 打印验证集的一个 batch
    images, input_ids, attention_mask, token_type_ids, labels = next(iter(val_loader))
    print("\nVal batch:")
    print("images:", images.shape)
    print("input_ids:", input_ids.shape)

    # 打印测试集的一个 batch
    images, input_ids, attention_mask, token_type_ids, labels = next(iter(test_loader))
    print("\nTest batch:")
    print("images:", images.shape)
    print("input_ids:", input_ids.shape)

if __name__ == "__main__":
    main()
