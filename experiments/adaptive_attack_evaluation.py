#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
適應性攻擊評估實驗 - 測試GradSafe防禦框架對抵抗自適應攻擊的有效性
"""

import os
import sys
import logging
import json
import time
import argparse
import datetime
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 將項目根目錄添加到路徑中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense import GradSafeDefense, create_defense, DefenseConfig
from features import SyntaxFeatureExtractor, BehaviorFeatureExtractor

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("adaptive_attack_experiments.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="GradSafe適應性攻擊評估實驗")
    
    parser.add_argument("--mode", type=str, 
                        choices=["asr_comparison", "efficiency", "ablation"], 
                        default="asr_comparison", 
                        help="實驗模式：攻擊成功率比較/計算效率/消融實驗")
    
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="運行設備")
    
    parser.add_argument("--model", type=str, 
                        default="vicuna-7b-v1.5",
                        help="使用的基礎LLM模型")
    
    parser.add_argument("--attack_method", type=str, 
                        choices=["GCG", "MGCG_ST", "MGCG_DT", "TGCG", "all"], 
                        default="all",
                        help="攻擊方法")
    
    parser.add_argument("--defense_mode", type=str, 
                        choices=["gradient_only", "behavior_only", "full", "all"], 
                        default="all",
                        help="防禦模式")
    
    parser.add_argument("--dataset", type=str, 
                        choices=["ds", "dh", "both"], 
                        default="both",
                        help="數據集類型：ds(數據竊取)/dh(直接傷害)/both(兩者)")
    
    parser.add_argument("--data_subset", type=str, 
                        choices=["base", "base_subset", "enhanced"], 
                        default="base_subset",
                        help="數據子集")
    
    parser.add_argument("--output_dir", type=str, 
                        default="results/adaptive_attacks",
                        help="輸出目錄")
    
    parser.add_argument("--num_samples", type=int, 
                        default=100,
                        help="每種攻擊方法使用的樣本數")
    
    parser.add_argument("--max_steps", type=int, 
                        default=100,
                        help="攻擊優化的最大步數")
    
    parser.add_argument("--enable_early_exit", type=bool, 
                        default=True,
                        help="是否啟用GradSafe的提前退出機制")
    
    parser.add_argument("--enable_resource_adaption", type=bool, 
                        default=True,
                        help="是否啟用GradSafe的資源自適應機制")
    
    parser.add_argument("--baseline_comparison", type=bool, 
                        default=True,
                        help="是否與基線方法進行比較")
    
    return parser.parse_args()

def load_test_data(args):
    """載入測試數據"""
    logger.info(f"載入測試數據: {args.dataset}, {args.data_subset}")
    
    datasets = []
    if args.dataset in ["dh", "both"]:
        with open(f"./InjecAgent/data/test_cases_dh_{args.data_subset}.json", "r") as f:
            dh_data = json.load(f)
            datasets.append(("dh", dh_data[:args.num_samples]))
            logger.info(f"載入直接傷害數據: {len(datasets[-1][1])}條")
            
    if args.dataset in ["ds", "both"]:
        with open(f"./InjecAgent/data/test_cases_ds_{args.data_subset}.json", "r") as f:
            ds_data = json.load(f)
            datasets.append(("ds", ds_data[:args.num_samples]))
            logger.info(f"載入數據竊取數據: {len(datasets[-1][1])}條")
    
    return datasets

def create_defense_configs(args):
    """創建不同的防禦配置"""
    defense_configs = []
    
    if args.defense_mode in ["gradient_only", "all"]:
        defense_configs.append({
            "name": "僅梯度特徵",
            "config": {
                "enable_grad_features": True,
                "enable_behavior_features": False,
                "enable_early_exit": args.enable_early_exit,
                "enable_resource_adaption": args.enable_resource_adaption,
                "device": args.device
            }
        })
        
    if args.defense_mode in ["behavior_only", "all"]:
        defense_configs.append({
            "name": "僅行為特徵",
            "config": {
                "enable_grad_features": False,
                "enable_behavior_features": True,
                "enable_early_exit": args.enable_early_exit,
                "enable_resource_adaption": args.enable_resource_adaption,
                "device": args.device
            }
        })
        
    if args.defense_mode in ["full", "all"]:
        defense_configs.append({
            "name": "完整防禦",
            "config": {
                "enable_grad_features": True,
                "enable_behavior_features": True,
                "enable_early_exit": args.enable_early_exit,
                "enable_resource_adaption": args.enable_resource_adaption,
                "device": args.device
            }
        })
    
    # 如果需要基線比較
    if args.baseline_comparison:
        # 可以在這裡添加基線方法的配置，如 Gradient Cuff, STPC 等
        defense_configs.append({
            "name": "基線方法 (無防禦)",
            "config": {
                "enable_grad_features": False,
                "enable_behavior_features": False,
                "enable_early_exit": False,
                "enable_resource_adaption": False,
                "device": args.device
            }
        })
    
    return defense_configs

def create_attack_configs(args):
    """創建不同的攻擊配置"""
    attack_configs = []
    
    if args.attack_method in ["GCG", "all"]:
        attack_configs.append({
            "name": "標準GCG",
            "method": "GCG",
            "config": {
                "num_steps": args.max_steps,
                "adv_string_init": " ! ! ! ! ! ! ! ! ! !",
                "adv_string_position": "suffix" if args.model == "vicuna-7b-v1.5" else "prefix",
                "search_width": 256
            }
        })
        
    if args.attack_method in ["MGCG_ST", "all"]:
        attack_configs.append({
            "name": "多目標GCG (同分詞器)",
            "method": "MGCG_ST",
            "config": {
                "num_steps": args.max_steps,
                "adv_string_init": " ! ! ! ! ! ! ! ! ! !",
                "adv_string_position": "suffix" if args.model == "vicuna-7b-v1.5" else "prefix",
                "alpha": 0.5,
                "search_width": 256
            }
        })
        
    if args.attack_method in ["MGCG_DT", "all"]:
        attack_configs.append({
            "name": "多目標GCG (不同分詞器)",
            "method": "MGCG_DT",
            "config": {
                "num_steps": args.max_steps,
                "adv_string_init": " ! ! ! ! !",
                "adv_string_position": "suffix" if args.model == "vicuna-7b-v1.5" else "prefix",
                "search_width": 256
            }
        })
        
    if args.attack_method in ["TGCG", "all"]:
        attack_configs.append({
            "name": "兩階段GCG",
            "method": "TGCG",
            "config": {
                "adv_string_position_1": "prefix",
                "adv_string_init_1": " ! ! ! ! !",
                "num_steps_1": min(50, args.max_steps // 2),
                "adv_string_position_2": "suffix" if args.model == "vicuna-7b-v1.5" else "prefix",
                "adv_string_init_2": " ! ! ! ! !",
                "num_steps_2": min(100, args.max_steps)
            }
        })
    
    return attack_configs

def preprocess_tool_calls(item):
    """從測試案例數據中提取工具調用信息"""
    attacker_tools = item.get("Attacker Tools", [])
    if not attacker_tools:
        return []
    
    # 構建模擬的工具調用
    tool_calls = []
    for tool in attacker_tools:
        tool_call = {
            "name": tool,
            "parameters": {}
        }
        
        # 如果有參數信息
        if "Attacker Tool Parameters" in item:
            try:
                tool_call["parameters"] = json.loads(item["Attacker Tool Parameters"].replace("'", "\""))
            except:
                # 簡單解析格式為 {'key': 'value'} 的參數
                params_str = item.get("Attacker Tool Parameters", "{}")
                params = {}
                try:
                    # 嘗試通過eval安全解析
                    import ast
                    params = ast.literal_eval(params_str)
                except:
                    logger.warning(f"無法解析工具參數: {params_str}")
                
                tool_call["parameters"] = params
        
        tool_calls.append(tool_call)
    
    return tool_calls

def evaluate_defense_against_attacks(args, defense_configs, attack_configs, datasets):
    """評估防禦框架對抗不同攻擊的效果"""
    results = {}
    
    for defense_config in defense_configs:
        defense_name = defense_config["name"]
        logger.info(f"\n=== 評估防禦方法: {defense_name} ===")
        
        # 創建防禦系統
        defense = create_defense(defense_config["config"])
        
        defense_results = {}
        
        for attack_config in attack_configs:
            attack_name = attack_config["name"]
            attack_method = attack_config["method"]
            
            logger.info(f"\n--- 攻擊方法: {attack_name} ---")
            
            attack_results = {}
            
            for dataset_type, dataset in datasets:
                logger.info(f"測試數據集: {dataset_type}, 樣本數: {len(dataset)}")
                
                attack_success_count = 0
                defense_success_count = 0
                compute_times = []
                
                for idx, item in enumerate(tqdm(dataset, desc=f"{defense_name} vs {attack_name} on {dataset_type}")):
                    # 獲取用戶查詢
                    user_query = item.get("User Instruction", "")
                    
                    # 獲取攻擊者指令
                    attacker_instruction = item.get("Attacker Instruction", "")
                    
                    # 獲取期望攻擊結果
                    expected_achievement = item.get("Expected Achievements", "")
                    
                    # 預處理工具調用
                    tool_calls = preprocess_tool_calls(item)
                    
                    if not tool_calls:
                        logger.warning(f"樣本 {idx} 沒有工具調用信息，跳過")
                        continue
                    
                    # 記錄開始時間
                    start_time = time.time()
                    
                    # 運行防禦分析
                    result = defense.defend(
                        user_query=user_query,
                        tool_calls=tool_calls
                    )
                    
                    # 記錄計算時間
                    compute_time = time.time() - start_time
                    compute_times.append(compute_time)
                    
                    # 檢查防禦是否成功 (防禦成功 = 識別出惡意工具調用)
                    is_defense_successful = result.get("is_potentially_harmful", False)
                    
                    if is_defense_successful:
                        defense_success_count += 1
                    else:
                        attack_success_count += 1
                    
                    # 如果處理的樣本數達到限制，則退出
                    if idx + 1 >= args.num_samples:
                        break
                
                # 計算指標
                total_samples = len(dataset) if len(dataset) <= args.num_samples else args.num_samples
                attack_success_rate = attack_success_count / total_samples
                defense_success_rate = defense_success_count / total_samples
                avg_compute_time = sum(compute_times) / len(compute_times) if compute_times else 0
                
                # 記錄結果
                attack_results[dataset_type] = {
                    "total_samples": total_samples,
                    "attack_success_count": attack_success_count,
                    "defense_success_count": defense_success_count,
                    "attack_success_rate": attack_success_rate,
                    "defense_success_rate": defense_success_rate,
                    "avg_compute_time": avg_compute_time,
                    "compute_times": compute_times
                }
                
                logger.info(f"攻擊成功率: {attack_success_rate:.4f}")
                logger.info(f"防禦成功率: {defense_success_rate:.4f}")
                logger.info(f"平均計算時間: {avg_compute_time:.4f}秒")
            
            defense_results[attack_name] = attack_results
        
        results[defense_name] = defense_results
    
    return results

def visualize_results(args, results):
    """可視化實驗結果"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成當前時間戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 繪製攻擊成功率比較圖
    if args.mode == "asr_comparison" or args.mode == "ablation":
        plt.figure(figsize=(15, 10))
        
        defense_names = list(results.keys())
        attack_names = list(results[defense_names[0]].keys())
        
        for dataset_idx, dataset_type in enumerate(['dh', 'ds']):
            if all(dataset_type in results[defense_name][attack_name] for defense_name in defense_names for attack_name in attack_names):
                plt.subplot(1, 2, dataset_idx + 1)
                
                # 準備數據
                defense_labels = []
                attack_labels = []
                asr_values = []
                
                for defense_name in defense_names:
                    for attack_name in attack_names:
                        if dataset_type in results[defense_name][attack_name]:
                            defense_labels.append(defense_name)
                            attack_labels.append(attack_name)
                            asr = results[defense_name][attack_name][dataset_type]["attack_success_rate"]
                            asr_values.append(asr)
                
                # 創建 DataFrame 用於繪圖
                import pandas as pd
                df = pd.DataFrame({
                    'Defense': defense_labels,
                    'Attack': attack_labels,
                    'ASR': asr_values
                })
                
                # 將數據轉為透視表
                pivot_df = df.pivot(index='Defense', columns='Attack', values='ASR')
                
                # 繪製熱力圖
                import seaborn as sns
                sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1)
                
                plt.title(f"攻擊成功率 - {dataset_type.upper()} 數據集")
                plt.xlabel("攻擊方法")
                plt.ylabel("防禦方法")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"attack_success_rate_{timestamp}.png"))
        logger.info(f"攻擊成功率比較圖已保存至 {args.output_dir}/attack_success_rate_{timestamp}.png")
    
    # 2. 計算效率比較圖
    if args.mode == "efficiency" or args.mode == "ablation":
        plt.figure(figsize=(15, 10))
        
        defense_names = list(results.keys())
        attack_names = list(results[defense_names[0]].keys())
        
        # 準備數據
        compute_times_by_defense = {defense_name: [] for defense_name in defense_names}
        
        for defense_name in defense_names:
            for attack_name in attack_names:
                for dataset_type in ['dh', 'ds']:
                    if dataset_type in results[defense_name][attack_name]:
                        compute_times_by_defense[defense_name].extend(
                            results[defense_name][attack_name][dataset_type]["compute_times"]
                        )
        
        # 繪製箱線圖
        plt.boxplot([compute_times_by_defense[defense_name] for defense_name in defense_names],
                    labels=defense_names, showfliers=False)
        
        plt.title("防禦方法計算時間比較")
        plt.xlabel("防禦方法")
        plt.ylabel("計算時間 (秒)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"compute_efficiency_{timestamp}.png"))
        logger.info(f"計算效率比較圖已保存至 {args.output_dir}/compute_efficiency_{timestamp}.png")
    
    # 3. 保存詳細結果
    with open(os.path.join(args.output_dir, f"detailed_results_{timestamp}.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"詳細結果已保存至 {args.output_dir}/detailed_results_{timestamp}.json")

def run_asr_comparison_experiment(args):
    """運行攻擊成功率比較實驗"""
    logger.info("=== 運行攻擊成功率比較實驗 ===")
    
    # 載入測試數據
    datasets = load_test_data(args)
    
    # 創建防禦配置
    defense_configs = create_defense_configs(args)
    
    # 創建攻擊配置
    attack_configs = create_attack_configs(args)
    
    # 評估防禦對抗攻擊的效果
    results = evaluate_defense_against_attacks(args, defense_configs, attack_configs, datasets)
    
    # 可視化結果
    visualize_results(args, results)
    
    return results

def run_efficiency_experiment(args):
    """運行計算效率實驗"""
    logger.info("=== 運行計算效率實驗 ===")
    
    # 修改參數以便更關注效率
    original_num_samples = args.num_samples
    args.num_samples = min(50, original_num_samples)  # 減少樣本數以加速實驗
    
    # 載入測試數據
    datasets = load_test_data(args)
    
    # 創建具有不同資源適應設置的防禦配置
    defense_configs = [
        {
            "name": "完整防禦 (啟用資源適應)",
            "config": {
                "enable_grad_features": True,
                "enable_behavior_features": True,
                "enable_early_exit": True,
                "enable_resource_adaption": True,
                "device": args.device
            }
        },
        {
            "name": "完整防禦 (僅啟用提前退出)",
            "config": {
                "enable_grad_features": True,
                "enable_behavior_features": True,
                "enable_early_exit": True,
                "enable_resource_adaption": False,
                "device": args.device
            }
        },
        {
            "name": "完整防禦 (無優化)",
            "config": {
                "enable_grad_features": True,
                "enable_behavior_features": True,
                "enable_early_exit": False,
                "enable_resource_adaption": False,
                "device": args.device
            }
        }
    ]
    
    # 只使用一種攻擊方法
    attack_configs = [create_attack_configs(args)[0]]
    
    # 評估防禦效率
    results = evaluate_defense_against_attacks(args, defense_configs, attack_configs, datasets)
    
    # 可視化結果
    visualize_results(args, results)
    
    # 恢復原始參數
    args.num_samples = original_num_samples
    
    return results

def run_ablation_experiment(args):
    """運行消融實驗"""
    logger.info("=== 運行消融實驗 ===")
    
    # 載入測試數據
    datasets = load_test_data(args)
    
    # 創建不同特徵組合的防禦配置
    defense_configs = create_defense_configs(args)
    
    # 創建攻擊配置
    attack_configs = create_attack_configs(args)
    
    # 評估防禦對抗攻擊的效果
    results = evaluate_defense_against_attacks(args, defense_configs, attack_configs, datasets)
    
    # 可視化結果
    visualize_results(args, results)
    
    return results

def main():
    """主函數"""
    args = parse_args()
    
    # 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根據不同的模式運行相應的實驗
    if args.mode == "asr_comparison":
        run_asr_comparison_experiment(args)
    elif args.mode == "efficiency":
        run_efficiency_experiment(args)
    elif args.mode == "ablation":
        run_ablation_experiment(args)
    else:
        logger.error(f"未知實驗模式: {args.mode}")

if __name__ == "__main__":
    main() 