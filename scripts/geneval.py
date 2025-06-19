# coding=utf-8
# Copyright 2024 Harmon Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import numpy as np
import torch
from src.builder import BUILDER
from PIL import Image
from mmengine.config import Config
import argparse
from einops import rearrange
from tqdm import tqdm, trange
import json

from xtuner.model.utils import guess_load_checkpoint

def set_seed(seed=0):
    """设置随机种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path.', default='configs/models/qwen2_5_1_5b_kl16_mar_h.py')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=12)  # 默认批量生成12张图
    parser.add_argument("--mode", type=str, default="t2i")
    parser.add_argument("--guidance_scale", "--cfg", type=float, default=3.0)
    parser.add_argument("--generation_timesteps", "--num_iter", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--cfg_schedule', type=str, default='constant')
    parser.add_argument('--cfg_prompt', type=str, default='Generate an image.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--validation_prompts_file', type=str, default='../Show-o/evaluation_metadata.jsonl')
    parser.add_argument('--l', type=int, default=0, help='Start index for processing')
    parser.add_argument('--r', type=int, default=None, help='End index for processing')
    parser.add_argument('--use_template', action='store_true', help='Use prompt template')
    parser.add_argument('--exp', type=str, default='exp4')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--remove_prefix', action='store_true', help='Remove prefix from prompts')

    
    
    args = parser.parse_args()
    if args.outdir is None:
        args.outdir = args.exp + f"_{args.step}"
    if args.checkpoint is None:
        args.checkpoint = f"work_dirs/{args.exp}/{args.exp}_{args.step}"

    # 确保输出目录存在
    os.makedirs(args.outdir, exist_ok=True)

    # 加载配置
    config = Config.fromfile(args.config)
    
    # 加载模型
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)
    
    if args.checkpoint is not None:
        print(f"Load checkpoint: {args.checkpoint}", flush=True)
        if os.path.isdir(args.checkpoint):
            checkpoint = guess_load_checkpoint(args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, weights_only=False)
    info = model.load_state_dict(checkpoint, strict=False)
    # 加载评估数据
    try:
        with open(args.validation_prompts_file) as fp:
            metadatas = [json.loads(line) for line in fp]
    except Exception as e:
        print(f"Error loading validation prompts file: {e}")
        metadatas = [{"prompt": "a dog on the left and a cat on the right."}]

    # 设置处理范围
    start_idx = args.l
    end_idx = args.r if args.r is not None else len(metadatas)
    
    # 处理每个提示
    for index, metadata in tqdm(enumerate(metadatas[start_idx:end_idx]), 
                                total=end_idx-start_idx,
                                desc="Processing prompts"):
        
        set_seed(args.seed)
        actual_index = index + start_idx
        
        # 创建输出目录
        outpath = os.path.join(args.outdir, f"{actual_index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        
        prompt = metadata.get("prompt", None)
        if args.remove_prefix:
            # remove "a photo of"
            prompt = prompt.replace("a photo of", "").strip()
            
        print(f"Prompt ({actual_index: >3}/{len(metadatas)}): '{prompt}'")
        
        # 创建样本目录并保存元数据
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
        
        set_seed(args.seed)
        
        # 检查现有图片数量，确定是否需要生成新图片
        existing_images = [f for f in os.listdir(sample_path) if f.endswith('.png')]
        if len(existing_images) >= 12:
            print(f"已存在12个或更多样本，跳过生成...")
            continue
            
        # 准备文本条件
        if not args.use_template:
            full_prompt = f"Generate an image: {prompt}"
        else:
            full_prompt = f"Describe the image: {prompt}"
        print(f"Processing prompt: {full_prompt}", flush=True)
        class_info = model.prepare_text_conditions(full_prompt, args.cfg_prompt)
        
        input_ids = class_info['input_ids']
        attention_mask = class_info['attention_mask']
        
        assert len(input_ids) == 2  # 最后一个是无条件提示
        
        # 复制到批处理大小 (12)
        batch_size = args.batch_size
        
        if args.guidance_scale != 1.0:
            input_ids = torch.cat([
                input_ids[0:1].expand(batch_size, -1),
                input_ids[1:2].expand(batch_size, -1),
            ])
            attention_mask = torch.cat([
                attention_mask[0:1].expand(batch_size, -1),
                attention_mask[1:2].expand(batch_size, -1),
            ])
        else:
            input_ids = input_ids[0:1].expand(batch_size, -1)
            attention_mask = attention_mask[0:1].expand(batch_size, -1)
        
        # 设置图像大小
        img_h = img_w = args.image_size // 16
        
        # 生成样本
        print(f"正在一次性生成 {batch_size} 张图片...")
        with torch.no_grad():
            samples = model.sample(input_ids=input_ids, 
                                  attention_mask=attention_mask,
                                  num_iter=args.generation_timesteps, 
                                  cfg=args.guidance_scale, 
                                  cfg_schedule=args.cfg_schedule,
                                  temperature=args.temperature, 
                                  progress=True, 
                                  image_shape=(img_h, img_w))
        # samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=1, n=1)
        # 后处理和保存每张图片
        for i in range(batch_size):
            sample = samples[i]
            # 将图像数据转换为正确格式
            sample = torch.clamp(127.5 * sample + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
            # import pdb; pdb.set_trace()
            sample = sample.transpose(1, 2, 0)  # 从[C,H,W]转换为[H,W,C]
            
            # 保存单张图片
            out_path = os.path.join(sample_path, f"{i:05}.png")
            Image.fromarray(sample).save(out_path)
            print(f"图像保存到: {out_path}")
    
    print("Done!")