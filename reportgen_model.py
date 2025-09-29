"""
reportgen_model.py

实现论文中的模型骨架：
- DenseNetBackbone 提取图像 patch 特征
- MemoryAlignmentModule (MA)
- ReportGenModel: 把视觉 token + 文本 token 合并后送入 BERT encoder，
  对文本位置的 hidden state 做线性投影到词表做预测。

说明：
- 这是一个尽量靠近论文结构的实现模板，用于训练/调试。实际复现可能要在细节（超参、位置编码、batch策略）上进一步调整。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel, BertTokenizer

# ----------------------------
# DenseNet Backbone
# ----------------------------
class DenseNetBackbone(nn.Module):
    """
    使用 torchvision 的 DenseNet121 提取卷积特征图并 flatten 成 patch 序列。
    输出 (B, S, C)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        densenet = models.densenet121(pretrained=pretrained)
        # torchvision 的 densenet.features 输出 (B, C, Hf, Wf)
        self.features = densenet.features

    def forward(self, x):
        """
        x: [B, 3, H, W]
        返回: [B, S, C] 其中 S = Hf * Wf
        """
        feat = self.features(x)  # (B, C, Hf, Wf)
        B, C, Hf, Wf = feat.shape
        feat = feat.view(B, C, Hf * Wf).permute(0, 2, 1)  # (B, S, C)
        return feat  # (B, S, C)

# ----------------------------
# Memory Alignment Module (MA)
# ----------------------------
class MemoryAlignmentModule(nn.Module):
    """
    多头注意力在可学习的 memory matrix M 上查询
    输入: visual_emb (B, S, d), pos_emb (B, S, d)
    输出: R (B, S, d)
    """
    def __init__(self, d_model=768, num_heads=4, memory_size=100, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.memory_size = memory_size

        # memory matrix M
        self.M = nn.Parameter(torch.randn(memory_size, d_model) * 0.02)

        # linear projections
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_emb, pos_emb):
        """
        visual_emb: (B, S, d)
        pos_emb: (B, S, d)
        return: (B, S, d)
        """
        B, S, D = visual_emb.shape
        device = visual_emb.device

        # query = visual + pos
        q = (visual_emb + pos_emb)  # (B,S,D)
        q = self.Wq(q)  # (B,S,D)

        # key/value from memory M
        M = self.M.unsqueeze(0).expand(B, -1, -1)  # (B, eta, D)
        k = self.Wk(M)  # (B, eta, D)
        v = self.Wv(M)  # (B, eta, D)

        # reshape for multi-head
        qh = q.view(B, S, self.num_heads, self.head_dim).transpose(1,2)  # (B,H,S,hd)
        kh = k.view(B, self.memory_size, self.num_heads, self.head_dim).permute(0,2,1,3)  # (B,H,eta,hd)
        vh = v.view(B, self.memory_size, self.num_heads, self.head_dim).permute(0,2,1,3)  # (B,H,eta,hd)

        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(qh, kh.transpose(-2,-1)) / scale  # (B,H,S,eta)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        response = torch.matmul(attn_weights, vh)  # (B,H,S,hd)
        response = response.transpose(1,2).contiguous().view(B, S, D)  # (B,S,D)
        out = self.out(response)
        return out  # (B,S,D)

# ----------------------------
# ReportGenModel
# ----------------------------
class ReportGenModel(nn.Module):
    """
    主模型：DenseNet -> proj -> MA -> 合并 BERT 输入 -> BERT encoder -> generator
    forward 支持训练（labels 给定）和仅返回 logits（推理阶段）
    """
    def __init__(self,
                 bert_model_name='bert-base-uncased',
                 d_model=768,
                 memory_size=100,
                 ma_heads=4,
                 visual_max_patch=49):
        super().__init__()
        self.d_model = d_model
        # 视觉骨干
        self.backbone = DenseNetBackbone(pretrained=True)

        # lazy projection: 将 backbone 的 C 映射到 d_model（延迟初始化）
        self._proj = None

        # BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_config = self.bert.config

        # Memory Alignment
        self.ma = MemoryAlignmentModule(d_model=d_model, num_heads=ma_heads, memory_size=memory_size)

        # 视觉位置 embedding（可学习），预设最大 patch 数（比如 7x7=49）
        self.visual_max_patch = visual_max_patch
        self.visual_pos_emb = nn.Parameter(torch.randn(self.visual_max_patch, d_model) * 0.02)

        # generator: hidden -> vocab
        self.generator = nn.Linear(d_model, self.bert_config.vocab_size)

    def _ensure_proj(self, C):
        """延迟创建将 C -> d_model 的线性映射（因为 backbone channel 可能不确定）"""
        if self._proj is None:
            self._proj = nn.Linear(C, self.d_model)

    def forward_encoder(self, images, input_ids, attention_mask, token_type_ids):
        """
        把 images 与文本 input 一起送入 BERT encoder。
        返回 encoder 的所有隐藏状态（B, S+L, d）和 S（视觉 token 数）。
        """
        B = images.size(0)
        # backbone -> (B, S, C)
        feat = self.backbone(images)
        B, S, C = feat.shape
        self._ensure_proj(C)
        visual_emb = self._proj(feat)  # (B, S, d)

        # pos embedding for visual patches (slice or repeat)
        if S <= self.visual_max_patch:
            pos = self.visual_pos_emb[:S, :].unsqueeze(0).expand(B, -1, -1)  # (B,S,d)
        else:
            # 如果 patch 数更多，可以插值或切取；这里简单切取前 visual_max_patch 并截断 visual_emb
            pos = self.visual_pos_emb.unsqueeze(0).expand(B, -1, -1)
            visual_emb = visual_emb[:, :self.visual_max_patch, :]
            S = self.visual_max_patch

        # MA embedding
        ma_emb = self.ma(visual_emb, pos)  # (B,S,d)

        # BERT embedding layers
        bert_emb_layer = self.bert.embeddings

        # text token embeddings (B, L, d)
        text_emb = bert_emb_layer.word_embeddings(input_ids)  # (B,L,d)

        # visual token embedding: use visual_emb (B,S,d)
        vis_token_emb = visual_emb  # (B,S,d)

        # concat tokens : [vis tokens] + [text tokens]
        tokens_emb = torch.cat([vis_token_emb, text_emb], dim=1)  # (B, S+L, d)

        # position embeddings for combined sequence
        seq_len = tokens_emb.size(1)
        pos_ids = torch.arange(seq_len, device=tokens_emb.device).unsqueeze(0).expand(B, -1)
        pos_emb = bert_emb_layer.position_embeddings(pos_ids)  # (B, S+L, d)

        # token_type embeddings: first S are image (0), rest use token_type_ids provided (text)
        token_type_image = torch.zeros((B, S), dtype=torch.long, device=input_ids.device)
        # 确保 token_type_ids 是 2 维（去除可能存在的冗余维度）
        token_type_ids = token_type_ids.squeeze(1)  # 新增这一行，处理维度不匹配问题
        combined_token_type = torch.cat([token_type_image, token_type_ids], dim=1)
        token_type_emb = bert_emb_layer.token_type_embeddings(combined_token_type)  # (B, S+L, d)

        # ma embedding only added to visual positions
        ma_pad = torch.zeros((B, seq_len, self.d_model), device=tokens_emb.device)
        ma_pad[:, :S, :] = ma_emb

        # final inputs_embeds for BERT
        inputs_embeds = tokens_emb + pos_emb + token_type_emb + ma_pad  # (B, S+L, d)

        # attention mask: visual tokens present (1) + text attention_mask
        visual_mask = torch.ones((B, S), dtype=torch.long, device=input_ids.device)
        combined_attn = torch.cat([visual_mask, attention_mask], dim=1)  # (B, S+L)

        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=combined_attn)
        # last_hidden_state: (B, S+L, d)
        return outputs.last_hidden_state, S

    def forward(self, images, input_ids, attention_mask, token_type_ids, labels=None):
        """
        主 forward：
        - 如果 labels 给出，按训练返回 (loss, logits)
        - 否则返回 logits（推理）
        """
        encoder_outputs, S = self.forward_encoder(images, input_ids, attention_mask, token_type_ids)
        # split text portion (the last L tokens)
        L = input_ids.size(1)
        text_encoded = encoder_outputs[:, S:, :]  # (B, L, d)
        logits = self.generator(text_encoded)    # (B, L, V)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        return logits

    @torch.no_grad()
    def generate_greedy(self, images, tokenizer, max_len=100, device='cuda'):
        """
        简单贪心生成（示例，不用于高效生产）
        说明：当前实现每步会重新计算 encoder（低效）。生产建议实现缓存或使用 decoder 模型。
        """
        self.eval()
        B = images.size(0)
        # 初始化生成序列为 [CLS]
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        generated = torch.full((B,1), cls_id, dtype=torch.long, device=images.device)

        for step in range(max_len):
            attention_mask = torch.ones_like(generated, device=generated.device)
            token_type_ids = torch.zeros_like(generated, device=generated.device)
            logits = self.forward(images, generated, attention_mask, token_type_ids)  # (B, L, V) if no labels
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == sep_id).all():
                break
        return generated
