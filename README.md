# 实验五：多模态情感分类（CLIP / VisualBERT / BLIP）
本项目实现并对比了三类多模态模型用于情感分类任务：**CLIP 融合模型**、**VisualBERT** 与 **BLIP**。包含数据划分、训练、验证、消融实验，以及对测试集 `test_without_label.txt` 的推理与提交文件生成。

## 一、环境配置
### 1.1 Python & 依赖
本项目主要在 **Apple Silicon (MPS)** 环境完成训练与推理；同时兼容 Linux/Windows（CPU/CUDA），但不同后端可能带来轻微数值差异。  

- Python >= 3.10
- PyTorch == 2.5.1
- torchvision == 0.20.1
- transformers == 4.57.6

所有依赖已统一写入 `requirements.txt`，以保证实验流程的可复现性。

```bash
conda create -n aihw5 python=3.10 -y
conda activate aihw5
pip install -r requirements.txt
```

### 1.2 目录结构
```text
ai_hw5/
├── code/
│   ├── train.py                 # CLIP 多模态模型训练（支持消融）
│   ├── train_blip.py            # BLIP 模型训练脚本
│   ├── train_visualbert.py      # VisualBERT 模型训练脚本
│   ├── infer.py                 # CLIP 模型推理与结果生成
│   ├── infer_blip.py            # BLIP 模型推理
│   ├── infer_visualbert.py      # VisualBERT 模型推理
│   ├── model.py                 # CLIP 多模态模型结构定义
│   ├── blip_model.py            # BLIP 模型封装
│   ├── visualbert_model.py      # VisualBERT 模型封装
│   ├── dataset.py               # 数据集加载与预处理（基于 guid 回盘）
│   ├── plot_history_one.py      # 训练过程曲线绘制脚本
│   └── plot_cm_one.py           # 混淆矩阵绘制脚本
│
├── project5/
│   └── data/
│       ├── train.txt
│       ├── train_split.txt
│       ├── val_split.txt
│       └── test_without_label.txt
│
├── README.md                    # 项目说明与复现指南
└── requirements.txt             # 环境依赖
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
### 3.1 CLIP 融合模型（含消融实验）
CLIP 模型采用统一的训练脚本 `train.py`，并通过 `--ablation` 参数顺序完成**Text-only → Image-only → Multimodal** 三种设置的训练，用于消融实验对比。  
```bash
python train.py --train_path ../project5/train_split.txt --val_path ../project5/val_split.txt --image_root ../project5/data --epochs 15 --bs 16 --lr 2e-5 --head_lr_mul 6 --warmup 0.06 --wd 0.01 --dropout 0.1 --proj_dim 256 --max_len 128 --image_size 224 --num_workers 0 --text_clean basic --img_aug strong --fusion gated --text_model bert-base-multilingual-cased --ema --ema_decay 0.999 --patience 6 --run_dir runs_ablation_after_textfix --ablation
```

输出：
- `runs_ablation_after_textfix/best_text_only.pt`
- `runs_ablation_after_textfix/best_image_only.pt`
- `runs_ablation_after_textfix/best_multimodal.pt`
- `runs_ablation_after_textfix/history_*.json`
- `runs_ablation_after_textfix/ablation_results.json`

说明：  
- 使用 --ablation 时，脚本会在一次运行中依次训练：text_only → image_only → multimodal
- 三种设置共享相同的数据划分与训练轮数，保证对比公平
- 各阶段最优模型与训练日志均保存在 runs_ablation_after_textfix/

### 3.2 VisualBERT
VisualBERT 模型采用阶段式训练策略：在训练初期冻结视觉编码器，待文本与跨模态注意力结构初步收敛后再解冻视觉分支，以提升训练稳定性。  
```bash
python train_visualbert.py --train_path ../project5/train_split.txt --val_path ../project5/val_split.txt --image_root ../project5/data --epochs 15 --bs 16 --lr 1e-5 --head_lr_mul 12 --img_lr_mul 0.5 --warmup 0.06 --wd 0.01 --dropout 0.1 --text_clean basic --img_aug strong --use_class_weight --class_weight_power 0.5 --label_smoothing 0.0 --freeze_image_epochs 3 --use_focal --focal_gamma 2.0 --ema --ema_decay 0.999 --run_dir runs_visualbert_tuned_v1
```

输出：  
- `runs_visualbert_tuned_v1/best_multimodal.pt`
- `runs_visualbert_tuned_v1/history_multimodal.json`

### 3.3 BLIP
BLIP 模型基于 `Salesforce/blip-itm-base-coco` 预训练权重。  
考虑到 Apple MPS 后端的稳定性，训练过程中保持视觉编码器冻结，仅微调高层跨模态模块与分类头。  
```bash
python train_blip.py --train_path ../project5/train_split.txt --val_path ../project5/val_split.txt --image_root ../project5/data --epochs 15 --bs 16 --lr 1e-5 --head_lr_mul 5 --vision_lr_mul 0.5 --warmup 0.06 --wd 0.01 --dropout 0.1 --pool cls --max_len 128 --image_size 224 --text_clean basic --img_aug strong --use_class_weight --class_weight_power 0.5 --use_focal --focal_gamma 2.0 --ema --ema_decay 0.999 --freeze_vision_epochs 99 --patience 4 --num_workers 0 --run_dir runs_blip_itm_retrieval_focal
```

输出：  
- `runs_blip_itm_retrieval_focal/best_blip.pt`
- `runs_blip_itm_retrieval_focal/history_blip.json`

## 四、实验设计与消融说明
- **Text-only**：仅使用文本特征（CLIP Text）
- **Image-only**：仅使用图像特征（CLIP Image Encoder）
- **Multimodal**：图文融合（concat / gated / cross-attn）

在验证集上对三种输入模式进行对比，报告 `Accuracy` 与 `Macro-F1`。

## 五、结果可视化与模型分析
5.1 训练过程曲线绘制（Loss / Acc / Macro-F1）  
本项目为每个模型在训练过程中保存了完整的 history_*.json 文件，可用于绘制训练与验证曲线。  
训练曲线主要用于分析模型的收敛速度与训练稳定性，不作为最终性能评判标准。  
绘制单个模型训练曲线：  
```bash
python plot_history_one.py \
  --history runs_xxx/history_xxx.json \
  --out runs_xxx/train_curve_xxx.png
```

示例：  
```bash
# CLIP multimodal
python plot_history_one.py \
  --history runs_ablation_after_textfix/history_multimodal.json \
  --out runs_ablation_after_textfix/curve_clip_multimodal.png

# BLIP
python plot_history_one.py \
  --history runs_blip_itm_retrieval_focal/history_blip.json \
  --out runs_blip_itm_retrieval_focal/curve_blip.png

# VisualBERT
python plot_history_one.py \
  --history runs_visualbert_tuned_v1/history_multimodal.json \
  --out runs_visualbert_tuned_v1/curve_visualbert.png
```

5.2 混淆矩阵绘制（Confusion Matrix）  
混淆矩阵用于分析模型在不同情感类别上的预测偏差，重点观察 neutral 类与正负类的混淆情况，是实验报告中重要的定性分析依据。  
基于验证集预测结果绘制混淆矩阵：  
```bash
python plot_cm_one.py \
  --pred_csv runs_xxx/val_pred_xxx.csv \
  --out runs_xxx/cm_xxx.png
```

示例：  
```bash
# CLIP multimodal
python plot_cm_one.py \
  --pred_csv runs_ablation_after_textfix/val_pred_multimodal.csv \
  --out runs_ablation_after_textfix/cm_clip_multimodal.png

# BLIP
python plot_cm_one.py \
  --pred_csv runs_blip_after_textfix/val_pred_blip.csv \
  --out runs_blip_after_textfix/cm_blip.png

# VisualBERT
python plot_cm_one.py \
  --pred_csv runs_visualbert_tuned_v1/val_pred_visualbert.csv \
  --out runs_visualbert_tuned_v1/cm_visualbert.png
```

## 六、验证集推理（用于实验分析）
验证集推理用于：  
- 计算最终 Accuracy / Macro-F1
- 导出预测结果
- 生成 bad cases 供错误分析

6.1 CLIP（Multimodal）
```bash
python infer.py \
  --ckpt runs_ablation_after_textfix/best_multimodal.pt \
  --data_path ../project5/val_split.txt \
  --image_root ../project5/data \
  --bs 16 \
  --out_csv runs_ablation_after_textfix/val_pred_multimodal.csv \
  --out_bad_csv runs_ablation_after_textfix/val_bad_cases.csv
```

6.2 BLIP
```bash
python infer_blip.py \
  --ckpt runs_blip_itm_retrieval_focal/best_blip.pt \
  --data_path ../project5/val_split.txt \
  --image_root ../project5/data \
  --bs 32 \
  --out_csv runs_blip_after_textfix/val_pred_blip.csv \
  --out_bad_csv runs_blip_after_textfix/val_bad_cases_blip.csv
```

6.3 VisualBERT
```bash
python infer_visualbert.py \
  --ckpt runs_visualbert_tuned_v1/best_multimodal.pt \
  --data_path ../project5/val_split.txt \
  --image_root ../project5/data \
  --bs 16 \
  --out_csv runs_visualbert_tuned_v1/val_pred_visualbert.csv \
  --out_bad_csv runs_visualbert_tuned_v1/val_bad_visualbert.csv
```

## 七、测试集推理与最终提交文件生成
最终提交基于验证集 Macro-F1 最优的模型（本实验中为 BLIP），在测试集 test_without_label.txt 上进行推理。    
```bash
python infer_blip.py --ckpt runs_blip_itm_retrieval_focal/best_blip.pt --data_path ../project5/test_without_label.txt --image_root ../project5/data --bs 32 --out_csv runs_blip_itm_retrieval_focal/test_pred_blip.csv --out_submit_txt runs_blip_itm_retrieval_focal/test_submit_blip.txt
```

最后重命名test_submit_blip.txt为test_without_labei.txt进行提交  

## 八、结果与分析
| 模型 | 输入模态 | Val Acc | Val Macro-F1 |
|---|---|---:|---:|
| CLIP | Text-only  | 0.7125 | 0.5801 |
| CLIP | Image-only | 0.6550 | 0.4996 |
| CLIP | Multimodal | 0.7312 | 0.6049 |
| VisualBERT | Multimodal | 0.7212 | 0.6275 |
| BLIP | Multimodal | 0.7375 | 0.6465 |

**结果分析：**  
从表中可以观察到，不同模型与输入模态在验证集上的性能存在明显差异。  
首先，在 CLIP 模型内部对比中，多模态输入显著优于单一模态。Text-only 的 Macro-F1 为 0.5801，Image-only 进一步下降至 0.4996，而 Multimodal 提升至 0.6049，说明图像信息能够为情感分类提供有效补充，且 CLIP 的跨模态对齐机制在该任务中发挥了积极作用。  
在多模态模型之间进行横向比较时，BLIP 模型取得了最佳整体性能，其验证集 Accuracy 与 Macro-F1 分别达到 0.7375 和 0.6465。相比之下，VisualBERT 的 Macro-F1 为 0.6275，虽略低于 BLIP，但仍明显优于 CLIP 多模态结果。这表明基于跨模态预训练并显式建模图文交互的模型，在情感分类任务中具有更强的判别能力。  
进一步来看，Macro-F1 指标的提升幅度普遍大于 Accuracy，说明多模态模型在少数类（尤其是 neutral 类）上的识别能力得到了改善。其中 BLIP 在类别不均衡场景下表现最为稳健，验证了引入 Focal Loss 与 EMA 训练策略的有效性。  
总体而言，实验结果表明：多模态融合是提升情感分类性能的关键因素，而结构设计更复杂、跨模态交互更充分的模型在该任务中更具优势。  

## 九、参考
- CLIP: Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*
- VisualBERT: Li et al., *VisualBERT: A Simple and Performant Baseline for Vision and Language*
- BLIP: Li et al., *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*

## 十、备注
- 本项目所有实验均可复现，随机种子与参数均在脚本中显式给出。
- 若在 Mac M 系列设备上运行，建议使用 `mps` 并关闭 DataLoader 多进程。
- 所有推理均基于 验证集最优 checkpoint（best_*.pt）
- 训练曲线仅用于辅助分析，不作为模型选择依据
- 混淆矩阵与 bad cases 用于实验报告中的误差分析部分