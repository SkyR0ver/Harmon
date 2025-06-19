#!/bin/bash

# =====================================================
# 图像标注评估脚本
# 使用 pycocoevalcap 计算 BLEU、METEOR、ROUGE_L、CIDEr 指标
#
# 使用方法:
#   bash evaluate.sh <model_name>
# =====================================================

# ==================== 配置参数 ====================
# 评估输出根目录
EVAL_OUTPUT_ROOT="../eval_res"

# pycocoevalcap 评估脚本目录
PYCOCOEVALCAP_DIR="../metric/pycocoevalcap/example"
# ==================================================

# 检查参数
if [ $# -ne 1 ]; then
    echo "用法: bash evaluate.sh <model_name>"
    echo "示例:"
    echo "  bash evaluate.sh raw"
    echo "  bash evaluate.sh finetune"
    exit 1
fi

MODEL_NAME="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$SCRIPT_DIR/$EVAL_OUTPUT_ROOT/$MODEL_NAME"
PYCOCOEVALCAP_PATH="$SCRIPT_DIR/$PYCOCOEVALCAP_DIR"

# 检查目录和文件
if [ ! -d "$PYCOCOEVALCAP_PATH" ]; then
    echo "错误: 评估工具目录不存在: $PYCOCOEVALCAP_PATH"
    exit 1
fi

GROUND_TRUTH_FILE="$EVAL_DIR/ground_truth_${MODEL_NAME}.json"
PREDICTIONS_FILE="$EVAL_DIR/predictions_${MODEL_NAME}.json"
RESULTS_FILE="$EVAL_DIR/results_${MODEL_NAME}.json"

if [ ! -f "$GROUND_TRUTH_FILE" ] || [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "错误: 评估数据文件不存在"
    echo "请先运行 'python prepare_evaluation_data.py $MODEL_NAME'"
    exit 1
fi

cd "$PYCOCOEVALCAP_PATH"
python eval_captions.py "$GROUND_TRUTH_FILE" "$PREDICTIONS_FILE" "$RESULTS_FILE"

if [ $? -ne 0 ]; then
    echo "评估失败"
    exit 1
fi
