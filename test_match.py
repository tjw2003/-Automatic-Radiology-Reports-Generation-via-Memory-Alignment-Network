import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

xml_dir = "./ecgen-radiology"    
image_dir = "./NLMCXR_png"       
output_file = "./records.json"

records = []

for fname in tqdm(os.listdir(xml_dir)):
    if not fname.endswith(".xml"):
        continue
    xml_file = os.path.join(xml_dir, fname)
    try:
        tree = ET.parse(xml_file)
    except:
        continue
    root = tree.getroot()

    # 报告内容
    findings, impression = [], []
    abstract = root.find("MedlineCitation/Article/Abstract")
    if abstract is not None:
        for abs_text in abstract.findall("AbstractText"):
            label = abs_text.attrib.get("Label", "").upper()
            text = abs_text.text or ""
            if "FINDINGS" in label:
                findings.append(text.strip())
            elif "IMPRESSION" in label:
                impression.append(text.strip())
    report_text = " ".join(findings + impression).strip()
    if not report_text:
        continue

    # 图像路径
    for img in root.findall("parentImage"):
        img_id = img.attrib.get("id", "")
        img_file = os.path.join(image_dir, f"{img_id}.png")  # 如果是.jpg就改这里
        if os.path.exists(img_file):
            records.append({
                "image_path": img_file,
                "report_text": report_text
            })

print(f"共匹配到 {len(records)} 条 图像-报告 对")

# 保存
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"已保存到 {output_file}")
