#!/usr/bin/env python3
"""
评估结果可视化脚本

该脚本用于读取不同模型的评估结果并使用matplotlib绘制评估分数的变化图表。
评估结果位于 eval_res/{model_name}/results_{model_name}.json 中。

功能特性：
1. 绘制 Bleu 系列分数共享的折线图
2. 绘制 METEOR、ROUGE_L、CIDEr 各自独占的折线图  
3. 横坐标为迭代轮数（按真实比例）
4. 对比 raw、finetune、finetune_iterxxx 系列模型

使用方法：
    python plot_eval_res.py                    # 使用默认设置
    python plot_eval_res.py --show            # 显示图片而不保存
    python plot_eval_res.py --output_dir plots # 指定输出目录
"""

import os
import json
import matplotlib.pyplot as plt
from typing import Dict
import argparse


def load_evaluation_results(eval_data_dir: str) -> Dict[str, Dict[str, float]]:
    """
    加载所有模型的评估结果
    
    Args:
        eval_data_dir: 评估数据目录路径
        
    Returns:
        包含所有模型评估结果的字典
    """
    results = {}
    
    # 遍历eval_res目录下的所有子目录
    for model_name in os.listdir(eval_data_dir):
        model_dir = os.path.join(eval_data_dir, model_name)
        
        if not os.path.isdir(model_dir):
            continue
            
        # 查找results文件
        result_files = [f for f in os.listdir(model_dir) if f.startswith('results') and f.endswith('.json')]
        
        if result_files:
            result_file = os.path.join(model_dir, result_files[0])
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    results[model_name] = json.load(f)
                print(f"已加载模型 {model_name} 的评估结果")
            except Exception as e:
                print(f"加载模型 {model_name} 的评估结果时出错: {e}")
                
    return results


def parse_model_iteration(model_name: str) -> int:
    """
    解析模型名称中的迭代次数
    
    Args:
        model_name: 模型名称
        
    Returns:
        迭代次数，raw为0，finetune为5000，finetune_iterxxx为xxx
    """
    if model_name == 'raw':
        return 0
    elif model_name == 'finetune':
        return 5000
    elif 'finetune_iter' in model_name:
        try:
            iter_str = model_name.replace('finetune_iter', '')
            return int(iter_str)
        except ValueError:
            return -1  # 无法解析的模型
    else:
        return -1  # 未知模型


def plot_bleu_scores(results: Dict[str, Dict[str, float]], save_path: str = None):
    """
    绘制 Bleu 系列分数的折线图
    
    Args:
        results: 包含所有模型评估结果的字典
        save_path: 图片保存路径，如果为None则显示图片
    """
    # 提取并排序数据
    data_points = []
    for model_name, model_results in results.items():
        iteration = parse_model_iteration(model_name)
        if iteration >= 0:  # 只处理可识别的模型
            data_points.append((iteration, model_results))
    
    # 按迭代次数排序
    data_points.sort(key=lambda x: x[0])
    
    if not data_points:
        print("没有找到可用的模型数据")
        return
    
    # 提取数据
    iterations = [point[0] for point in data_points]
    bleu_metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
    
    # 设置图表
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'v']
    
    for i, metric in enumerate(bleu_metrics):
        scores = [point[1].get(metric, 0) for point in data_points]
        plt.plot(iterations, scores, marker=markers[i], color=colors[i], 
                linewidth=2, markersize=8, label=metric)
    
    plt.title('Bleu Scores vs Training Iterations', fontsize=16, fontweight='bold')
    plt.xlabel('Training Iterations', fontsize=14)
    plt.ylabel('Bleu Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度，确保显示所有迭代点
    plt.xticks(iterations)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bleu分数图已保存到: {save_path}")
    else:
        plt.show()


def plot_individual_metric(results: Dict[str, Dict[str, float]], metric_name: str, save_path: str = None):
    """
    绘制单个评估指标的折线图
    
    Args:
        results: 包含所有模型评估结果的字典
        metric_name: 要绘制的指标名称
        save_path: 图片保存路径，如果为None则显示图片
    """
    # 提取并排序数据
    data_points = []
    for model_name, model_results in results.items():
        iteration = parse_model_iteration(model_name)
        if iteration >= 0 and metric_name in model_results:  # 只处理可识别的模型且包含该指标
            data_points.append((iteration, model_results[metric_name]))
    
    # 按迭代次数排序
    data_points.sort(key=lambda x: x[0])
    
    if not data_points:
        print(f"没有找到指标 {metric_name} 的数据")
        return
    
    # 提取数据
    iterations = [point[0] for point in data_points]
    scores = [point[1] for point in data_points]
    
    # 设置图表
    plt.figure(figsize=(12, 8))
    
    # 选择颜色
    if metric_name == 'METEOR':
        color = '#ff7f0e'
    elif metric_name == 'ROUGE_L':
        color = '#2ca02c'
    elif metric_name == 'CIDEr':
        color = '#d62728'
    else:
        color = '#1f77b4'
    
    plt.plot(iterations, scores, marker='o', color=color, 
            linewidth=3, markersize=10, label=metric_name)
    
    plt.title(f'{metric_name} Score vs Training Iterations', fontsize=16, fontweight='bold')
    plt.xlabel('Training Iterations', fontsize=14)
    plt.ylabel(f'{metric_name} Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度，确保显示所有迭代点
    plt.xticks(iterations)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{metric_name}分数图已保存到: {save_path}")
    else:
        plt.show()


def sort_models_by_training_progress(results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    按照训练进度（batch数）对模型进行排序
    
    Args:
        results: 包含所有模型评估结果的字典
        
    Returns:
        按训练进度排序后的模型结果字典
    """
    def get_training_order(model_name: str) -> int:
        """获取模型的训练顺序编号"""
        if model_name == 'raw':
            return 0
        elif 'iter1000' in model_name:
            return 1
        elif 'iter2000' in model_name:
            return 2
        elif 'iter3000' in model_name:
            return 3
        elif 'iter4000' in model_name:
            return 4
        elif model_name == 'finetune':
            return 5  # finetune是最终微调模型，放在最后
        else:
            return 999  # 未知模型放到最后
    
    # 按训练进度排序
    sorted_items = sorted(results.items(), key=lambda x: get_training_order(x[0]))
    return dict(sorted_items)


def main():
    parser = argparse.ArgumentParser(description='可视化不同模型的评估分数变化')
    parser.add_argument('--eval_data_dir', type=str, default='../eval_res',
                       help='评估数据目录路径 (默认: ../eval_res)')
    parser.add_argument('--output_dir', type=str, default='../plots',
                       help='图片输出目录 (默认: plots)')
    parser.add_argument('--show', action='store_true',
                       help='显示图片而不是保存')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    if not args.show:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载评估结果
    print("正在加载评估结果...")
    results = load_evaluation_results(args.eval_data_dir)
    
    if not results:
        print("没有找到任何评估结果")
        return
    
    print(f"找到 {len(results)} 个模型的评估结果:")
    # 按迭代次数排序显示模型列表
    sorted_models = sorted(results.keys(), key=lambda x: parse_model_iteration(x))
    for model_name in sorted_models:
        iteration = parse_model_iteration(model_name)
        if iteration >= 0:
            print(f"  - {model_name} (迭代: {iteration})")
    
    # 设置中文字体支持
    try:
        # 尝试设置中文字体，如果没有则使用英文
        import matplotlib
        matplotlib.rcParams['font.family'] = ['sans-serif']
        matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
    except Exception:
        pass
    
    # 绘制 Bleu 系列分数图
    print("\n正在生成 Bleu 系列分数图...")
    bleu_path = None if args.show else os.path.join(args.output_dir, 'bleu_scores.png')
    plot_bleu_scores(results, bleu_path)
    
    # 绘制其他单独指标图
    other_metrics = ['METEOR', 'ROUGE_L', 'CIDEr']
    for metric in other_metrics:
        # 检查是否有该指标的数据
        has_metric = any(metric in model_results for model_results in results.values())
        if has_metric:
            print(f"正在生成 {metric} 分数图...")
            metric_path = None if args.show else os.path.join(args.output_dir, f'{metric.lower()}_scores.png')
            plot_individual_metric(results, metric, metric_path)
        else:
            print(f"警告: 没有找到 {metric} 指标的数据")
    
    print("\n可视化完成!")


if __name__ == "__main__":
    main()
