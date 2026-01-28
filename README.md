# 实验五：多模态情感分类（CLIP / VisualBERT / BLIP）
本项目实现并对比了三类多模态模型用于情感分类任务：**CLIP 融合模型**、**VisualBERT** 与 **BLIP**。包含数据划分、训练、验证、消融实验，以及对测试集 `test_without_label.txt` 的推理与提交文件生成。

## 一、环境配置
### 1.1 Python & 依赖
本项目主要实验环境为 **Apple Silicon MacBook Pro（M 系列芯片，MPS 后端）**，并已在该环境下完成全部训练与推理实验。项目同时兼容 Linux / Windows（CPU 或 CUDA），但由于底层计算后端差异，数值结果可能存在轻微波动。  

- Python >= 3.10
- PyTorch == 2.1.2
- torchvision == 0.16.2
- transformers == 4.36.2

所有依赖及其版本已统一写入 `requirements.txt`，以保证实验流程的可复现性。

```bash
conda create -n aihw5 python=3.10 -y
conda activate aihw5
pip install -r requirements.txt
```

### 1.2 目录结构
```text
ai_hw5/
├── code/
│   ├── train.py
│   ├── infer.py
│   ├── dataset.py
│   ├── model.py
│   ├── train_visualbert.py
│   ├── infer_visualbert.py
│   ├── visualbert_model.py
│   ├── train_blip.py
│   ├── infer_blip.py
│   ├── blip_model.py
│   └── split_train_val.py
├── project5/
│   ├── train.txt
│   ├── test_without_label.txt
│   └── data/                # 图片根目录
├── runs/                     # 训练与推理输出
└── README.md
```

## 二、数据准备
### 2.1 数据格式
`train.txt / val_split.txt / test_without_label.txt` 均支持以下格式之一（第 2 列为情感标签）：

```text
guid,tag,text
```
或
```text
guid\ttag\ttext
```

- 训练 / 验证集：`tag ∈ {negative, neutral, positive}`
- 测试集：`tag = null`

### 2.2 划分训练 / 验证集
```bash
python split_train_val.py \
  --input ../project5/train.txt \
  --train_out ../project5/train_split.txt \
  --val_out ../project5/val_split.txt \
  --val_ratio 0.2 \
  --seed 42 \
  --shuffle
```

## 三、模型训练
### 3.1 CLIP 融合模型（Multimodal / Image-only / Text-only）
#### 多模态（Multimodal）
```bash
python train.py \
  --train_path ../project5/train_split.txt \
  --val_path ../project5/val_split.txt \
  --image_root ../project5/data \
  --epochs 20 --bs 16 --lr 2e-5 \
  --head_lr_mul 10 --warmup 0.1 --wd 0.01 --dropout 0.2 \
  --proj_dim 256 --max_len 77 --text_clean basic --img_aug strong \
  --fusion cross_attn --sampler --use_class_weight --class_weight_power 0.5 \
  --label_smoothing 0.05 --logit_adjust --adjust_tau 0.8 \
  --ema --ema_decay 0.999 --use_clip \
  --clip_name laion/CLIP-ViT-B-32-laion2B-s34B-b79K \
  --patience 4
```

#### 消融实验（Text-only / Image-only / Multimodal）

```bash
python train.py \
  --train_path ../project5/train_split.txt \
  --val_path ../project5/val_split.txt \
  --image_root ../project5/data \
  --epochs 15 --bs 16 --lr 2e-5 \
  --head_lr_mul 10 --warmup 0.1 --wd 0.01 --dropout 0.2 \
  --proj_dim 256 --max_len 128 --text_clean basic \
  --sampler --use_class_weight --class_weight_power 0.5 \
  --label_smoothing 0.05 --ema --ema_decay 0.999 \
  --fusion gated --run_dir runs_bert_text_only \
  --patience 4 --ablation
```

**说明：使用 `--ablation` 会顺序训练 `text_only → image_only → multimodal`，如仅需 text_only，可在其结束后手动 `Ctrl+C`。**

### 3.2 VisualBERT

```bash
python train_visualbert.py \
  --train_path ../project5/train_split.txt \
  --val_path ../project5/val_split.txt \
  --image_root ../project5/data \
  --epochs 20 --bs 16 --lr 1e-5 \
  --head_lr_mul 12 --img_lr_mul 0.5 \
  --warmup 0.06 --wd 0.01 --dropout 0.1 \
  --use_class_weight --class_weight_power 0.5
```

### 3.3 BLIP

```bash
python train_blip.py \
  --train_path ../project5/train_split.txt \
  --val_path ../project5/val_split.txt \
  --image_root ../project5/data \
  --epochs 15 --bs 16 --lr 1e-5 \
  --head_lr_mul 5 --vision_lr_mul 0.5 \
  --warmup 0.06 --wd 0.01 --dropout 0.1 \
  --use_class_weight --class_weight_power 0.5 \
  --use_focal --focal_gamma 2.0 --ema
```

## 四、推理与提交文件生成
### 4.1 CLIP 融合模型
```bash
python infer.py \
  --ckpt runs/best_multimodal.pt \
  --data_path ../project5/test_without_label.txt \
  --image_root ../project5/data \
  --bs 16 \
  --out_csv runs/test_preds_clip.csv \
  --out_submit_txt runs/test_without_label_clip.txt
```

### 4.2 VisualBERT
```bash
python infer_visualbert.py \
  --ckpt runs_visualbert/best.pt \
  --data_path ../project5/test_without_label.txt \
  --image_root ../project5/data \
  --bs 16 \
  --out_csv runs_visualbert/test_preds.csv \
  --out_submit_txt runs_visualbert/test_without_label_visualbert.txt
```

### 4.3 BLIP
```bash
python infer_blip.py \
  --ckpt runs_blip/best.pt \
  --data_path ../project5/test_without_label.txt \
  --image_root ../project5/data \
  --bs 16 \
  --out_csv runs_blip/test_preds.csv \
  --out_submit_txt runs_blip/test_without_label_blip.txt
```

## 五、实验设计与消融说明
- **Text-only**：仅使用文本特征（BERT 或 CLIP Text）
- **Image-only**：仅使用图像特征（CLIP Image Encoder）
- **Multimodal**：图文融合（concat / gated / cross-attn）

在验证集上对三种输入模式进行对比，报告 `Accuracy` 与 `Macro-F1`。

## 六、结果与分析
| 模型 | 输入模态 | Val Acc | Val Macro-F1 |
|---|---|---:|---:|
| CLIP | Text-only (BERT) | TBD | TBD |
| CLIP | Image-only | ~0.62 | ~0.57 |
| CLIP | Multimodal | ~0.67 | ~0.60 |
| VisualBERT | Multimodal | TBD | TBD |
| BLIP | Multimodal | TBD | TBD |


## 七、参考
- CLIP: Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*
- VisualBERT: Li et al., *VisualBERT: A Simple and Performant Baseline for Vision and Language*
- BLIP: Li et al., *BLIP: Bootstrapping Language-Image Pre-training*

## 八、备注
- 本项目所有实验均可复现，随机种子与参数均在脚本中显式给出。
- 若在 Mac M 系列设备上运行，建议使用 `mps` 并关闭 DataLoader 多进程。