#!/bin/bash

# =====================================================
# 批量处理图片生成文本描述
# 输入：指定目录中的 .jpg 图片文件
# 输出：对应的 .txt 文本描述文件
# 
# 使用方法：
#   ./validate_unified.sh <gpu_id> [max_files]
# 
# 参数说明：
#   gpu_id:     使用的 GPU 编号
#   max_files:  最大处理文件数量（可选，默认处理所有文件）
# 
# 示例：
#   ./validate_unified.sh 0     # 使用 GPU 0，处理所有文件
#   ./validate_unified.sh 1 100 # 使用 GPU 1，最多处理 100 个文件
# =====================================================

# ==================== 配置参数 ====================
# 模型配置文件路径
MODEL_CONFIG="../configs/models/qwen2_5_0_5b_kl16_mar_b.py"

# 模型权重目录路径
# CHECKPOINT="../checkpoints/harmon_0.5b.pth"
CHECKPOINT="../work_dirs/qwen2_5_0_5b_kl16_mar_b_train/iter_2800.pth"

# 输出目录
# OUTPUT_DIR="../output/raw"
OUTPUT_DIR="../output/finetune_iter2800"

# 输入图片目录
INPUT_IMAGE_DIR="../data/cc3m_validation/images"

# 图片处理配置
IMAGE_SIZE=512
PROMPT="Describe the image in detail."

# 脚本路径
SCRIPT_PATH="../scripts/image2text.py"
# ==================================================

# 检查参数
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <gpu_id> [max_files]"
    echo "Example: $0 0        # 处理所有文件"
    echo "Example: $0 0 100    # 最多处理100个文件"
    exit 1
fi

GPU_ID=$1
MAX_FILES=${2:-0}  # 默认为0表示处理所有文件

if [ ! -e "$CHECKPOINT" ]; then
    echo "错误：检查点路径不存在: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$MODEL_CONFIG" ]; then
    echo "错误：模型配置文件不存在: $MODEL_CONFIG"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误：脚本文件不存在: $SCRIPT_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 获取图片文件列表
image_files=($(ls "$INPUT_IMAGE_DIR"/*.jpg 2>/dev/null | sort))
total_available=${#image_files[@]}

if [ $total_available -eq 0 ]; then
    echo "错误：在 $INPUT_IMAGE_DIR 中没有找到 .jpg 文件"
    exit 1
fi

# 计算实际要处理的文件数
if [ $MAX_FILES -gt 0 ] && [ $MAX_FILES -lt $total_available ]; then
    total_to_process=$MAX_FILES
else
    total_to_process=$total_available
fi

echo "开始处理 $total_to_process 个文件 (GPU: $GPU_ID)"

# 计数器
processed=0
skipped=0
failed=0
total=0

# 处理每个图片文件
for image_file in "${image_files[@]}"; do
    basename_no_ext=$(basename "$image_file" .jpg)
    output_file="$OUTPUT_DIR/${basename_no_ext}.txt"
    
    total=$((total + 1))
    
    # 检查是否达到最大处理数量限制
    if [ $MAX_FILES -gt 0 ] && [ $total -gt $MAX_FILES ]; then
        break
    fi
    
    echo "进度: $total/$total_to_process"

    # 检查输出文件是否已存在
    if [ -f "$output_file" ]; then
        skipped=$((skipped + 1))
        continue
    fi
    
    # 运行图片转文本
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    PYTHONPATH=.. \
    python "$SCRIPT_PATH" \
        "$MODEL_CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --image_size $IMAGE_SIZE \
        --image "$image_file" \
        --prompt "$PROMPT" \
        --output "$output_file" >/dev/null 2>&1
    
    # 检查是否成功生成
    if [ $? -eq 0 ] && [ -f "$output_file" ]; then
        processed=$((processed + 1))
    else
        failed=$((failed + 1))
    fi
done

echo "完成: 处理 $processed，跳过 $skipped，失败 $failed"
