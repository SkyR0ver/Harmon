#!/usr/bin/env python3
"""
将指定模型的输出和数据集标注整理为 pycocoevalcap 评估所需的格式

使用方法:
    python prepare_evaluation_data.py <model_name>
"""

import json
import sys
from pathlib import Path

# ==================== 配置参数 ====================
# 数据集标注目录
GT_TEXT_DIR = "../data/cc3m_validation/texts"

# 模型输出根目录
MODEL_OUTPUT_ROOT = "../output"

# 评估输出根目录
EVAL_OUTPUT_ROOT = "../eval_res"
# ==================================================

def load_ground_truth_texts(texts_dir):
    """加载数据集标注文本"""
    gt_data = {}
    for txt_file in Path(texts_dir).glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
                if caption:
                    gt_data[txt_file.stem] = [caption]
        except Exception:
            continue
    return gt_data

def load_predictions(pred_dir):
    """加载模型预测文本"""
    pred_data = {}
    for txt_file in Path(pred_dir).glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                caption = f.read().strip().split('\n')[0].strip()
                if caption:
                    pred_data[txt_file.stem] = caption
        except Exception:
            continue
    return pred_data

def find_common_ids(gt_data, pred_data):
    """找到同时存在于标注和预测中的图像ID"""
    common_ids = set(gt_data.keys()).intersection(set(pred_data.keys()))
    print(f"找到 {len(common_ids)} 个有效样本")
    return common_ids

def prepare_evaluation_data(gt_data, pred_data, common_ids):
    """准备评估数据格式"""
    return ({img_id: gt_data[img_id] for img_id in common_ids},
            {img_id: pred_data[img_id] for img_id in common_ids})

def main():
    if len(sys.argv) != 2:
        print("用法: python prepare_evaluation_data.py <model_name>")
        print("示例:")
        print("  python prepare_evaluation_data.py raw")
        print("  python prepare_evaluation_data.py finetune") 
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    base_dir = Path(__file__).parent
    texts_dir = base_dir / GT_TEXT_DIR
    model_output_dir = base_dir / MODEL_OUTPUT_ROOT / model_name
    eval_data_dir = base_dir / EVAL_OUTPUT_ROOT / model_name
    eval_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"准备 {model_name} 模型评估数据...")
    
    # 加载数据
    gt_data = load_ground_truth_texts(texts_dir)
    if not gt_data:
        print("错误: 未找到标注数据")
        return
    
    if not model_output_dir.exists():
        print(f"错误: 输出目录不存在: {model_output_dir}")
        return
        
    pred_data = load_predictions(model_output_dir)
    if not pred_data:
        print("错误: 未找到预测数据")
        return
    
    common_ids = find_common_ids(gt_data, pred_data)
    if not common_ids:
        print("错误: 未找到共同 ID")
        return
    
    # 保存数据
    filtered_gt, filtered_pred = prepare_evaluation_data(gt_data, pred_data, common_ids)
    
    gt_file = eval_data_dir / f"ground_truth_{model_name}.json"
    pred_file = eval_data_dir / f"predictions_{model_name}.json"
    
    with open(gt_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_gt, f, ensure_ascii=False, indent=2)
    
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_pred, f, ensure_ascii=False, indent=2)
    
    print(f"数据已保存到 {eval_data_dir}")

if __name__ == "__main__":
    main()
