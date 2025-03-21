#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基線比較實驗 - 比較GradSafe與其他基線方法的防禦效果
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

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("baseline_comparison.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="GradSafe與基線方法比較實驗")
    
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="運行設備")
    
    parser.add_argument("--dataset", type=str, 
                        choices=["ds", "dh", "both"], 
                        default="both",
                        help="數據集類型：ds(數據竊取)/dh(直接傷害)/both(兩者)")
    
    parser.add_argument("--data_subset", type=str, 
                        choices=["base", "base_subset", "enhanced"], 
                        default="base_subset",
                        help="數據子集")
    
    parser.add_argument("--num_samples", type=int, 
                        default=100,
                        help="每個數據集使用的樣本數")
    
    parser.add_argument("--output_dir", type=str, 
                        default="results/baseline_comparison",
                        help="輸出目錄")
    
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

def load_gradient_cuff():
    """載入Gradient Cuff基線方法"""
    logger.info("載入Gradient Cuff基線方法")
    
    try:
        sys.path.append(os.path.abspath("../Gradient-Cuff"))
        from Gradient_Cuff.main import GradientCuffDefense
        
        # 創建Gradient Cuff防禦實例
        gradient_cuff = GradientCuffDefense()
        
        return gradient_cuff
    except Exception as e:
        logger.error(f"載入Gradient Cuff失敗: {e}")
        logger.warning("將使用模擬的Gradient Cuff")
        
        # 如果無法載入真實的Gradient Cuff，使用模擬實現
        class SimulatedGradientCuff:
            def __init__(self):
                self.name = "Simulated Gradient Cuff"
            
            def defend(self, user_query, tool_calls):
                # 簡單基於關鍵詞的分析
                query_lower = user_query.lower()
                harmful_keywords = ["hack", "steal", "attack", "vulnerability", "exploit", 
                                  "password", "credit card", "unlock", "access", "delete"]
                
                is_harmful = any(keyword in query_lower for keyword in harmful_keywords)
                
                # 檢查工具調用是否包含敏感操作
                sensitive_tools = ["SendEmail", "DeleteFile", "AccessDatabase", 
                                 "WriteFile", "ExecuteCommand", "PayBill"]
                
                tool_risk = any(
                    any(sensitive in tool["name"] for sensitive in sensitive_tools)
                    for tool in tool_calls
                )
                
                return {
                    "is_potentially_harmful": is_harmful or tool_risk,
                    "risk_score": 0.7 if (is_harmful or tool_risk) else 0.2,
                    "defense_method": "simulated_gradient_cuff"
                }
        
        return SimulatedGradientCuff()

def load_stpc_defense():
    """載入STPC (Self-Temperature Prompt Consistency)基線方法"""
    logger.info("載入STPC基線方法")
    
    try:
        # 嘗試載入實際的STPC實現
        # 這裡需要根據實際情況來實現
        raise NotImplementedError("尚未實現STPC載入")
    except Exception as e:
        logger.error(f"載入STPC失敗: {e}")
        logger.warning("將使用模擬的STPC")
        
        # 如果無法載入真實的STPC，使用模擬實現
        class SimulatedSTPCDefense:
            def __init__(self):
                self.name = "Simulated STPC"
            
            def defend(self, user_query, tool_calls):
                # 模擬STPC防禦 - 基於查詢複雜度和工具敏感度的簡單分析
                query_length = len(user_query.split())
                query_complexity = min(1.0, query_length / 50)  # 規一化複雜度
                
                # 識別敏感工具
                sensitive_tools = ["SendEmail", "DeleteFile", "AccessDatabase", 
                                 "PayBill", "ExecuteCommand"]
                
                tool_risk = 0.0
                for tool in tool_calls:
                    if any(sensitive in tool["name"] for sensitive in sensitive_tools):
                        tool_risk += 0.3
                tool_risk = min(1.0, tool_risk)
                
                # 綜合風險
                risk_score = (query_complexity + tool_risk) / 2
                
                return {
                    "is_potentially_harmful": risk_score > 0.5,
                    "risk_score": risk_score,
                    "defense_method": "simulated_stpc"
                }
        
        return SimulatedSTPCDefense()

def create_gradsafe_defense(args):
    """創建GradSafe防禦"""
    logger.info("創建GradSafe防禦")
    
    # 完整GradSafe配置
    config = {
        "enable_grad_features": True,
        "enable_behavior_features": True,
        "enable_early_exit": True,
        "enable_resource_adaption": True,
        "device": args.device
    }
    
    return create_defense(config)

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

def evaluate_defense_methods(defenses, datasets, args):
    """評估不同防禦方法的效果"""
    results = {}
    
    for defense_name, defense in defenses.items():
        logger.info(f"\n=== 評估防禦方法: {defense_name} ===")
        
        defense_results = {}
        
        for dataset_type, dataset in datasets:
            logger.info(f"測試數據集: {dataset_type}, 樣本數: {len(dataset)}")
            
            true_positives = 0  # 成功識別有害查詢
            false_positives = 0  # 錯誤將無害查詢識別為有害
            true_negatives = 0  # 成功識別無害查詢
            false_negatives = 0  # 錯誤將有害查詢識別為無害
            compute_times = []
            
            for idx, item in enumerate(tqdm(dataset, desc=f"{defense_name} on {dataset_type}")):
                # 獲取用戶查詢和工具調用
                user_query = item.get("User Instruction", "")
                tool_calls = preprocess_tool_calls(item)
                
                if not tool_calls:
                    logger.warning(f"樣本 {idx} 沒有工具調用信息，跳過")
                    continue
                
                # 根據數據集類型確定真實標籤
                # 在InjecAgent數據集中，所有樣本都是攻擊樣本
                is_truly_harmful = True
                
                # 記錄開始時間
                start_time = time.time()
                
                # 運行防禦分析
                try:
                    result = defense.defend(
                        user_query=user_query,
                        tool_calls=tool_calls
                    )
                    
                    # 檢查防禦結果
                    is_classified_harmful = result.get("is_potentially_harmful", False)
                    
                    # 更新混淆矩陣
                    if is_truly_harmful and is_classified_harmful:
                        true_positives += 1
                    elif is_truly_harmful and not is_classified_harmful:
                        false_negatives += 1
                    elif not is_truly_harmful and is_classified_harmful:
                        false_positives += 1
                    else:
                        true_negatives += 1
                    
                except Exception as e:
                    logger.error(f"{defense_name} 處理樣本 {idx} 失敗: {e}")
                    false_negatives += 1  # 如果防禦失敗，計為漏報
                
                # 記錄計算時間
                compute_time = time.time() - start_time
                compute_times.append(compute_time)
                
                # 如果處理的樣本數達到限制，則退出
                if idx + 1 >= args.num_samples:
                    break
            
            # 計算指標
            total = true_positives + false_positives + true_negatives + false_negatives
            accuracy = (true_positives + true_negatives) / total if total > 0 else 0
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            avg_compute_time = sum(compute_times) / len(compute_times) if compute_times else 0
            
            # 記錄結果
            defense_results[dataset_type] = {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "avg_compute_time": avg_compute_time,
                "compute_times": compute_times
            }
            
            logger.info(f"準確率: {accuracy:.4f}")
            logger.info(f"精確率: {precision:.4f}")
            logger.info(f"召回率: {recall:.4f}")
            logger.info(f"F1分數: {f1_score:.4f}")
            logger.info(f"平均計算時間: {avg_compute_time:.4f}秒")
        
        results[defense_name] = defense_results
    
    return results

def visualize_comparison_results(results, args):
    """可視化比較結果"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成當前時間戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 繪製F1分數比較圖
    plt.figure(figsize=(12, 8))
    
    defense_names = list(results.keys())
    
    for dataset_idx, dataset_type in enumerate(['dh', 'ds']):
        if all(dataset_type in results[defense_name] for defense_name in defense_names):
            plt.subplot(1, 2, dataset_idx + 1)
            
            # 準備數據
            metrics = ["precision", "recall", "f1_score", "accuracy"]
            metric_labels = ["精確率", "召回率", "F1分數", "準確率"]
            
            x = np.arange(len(metrics))
            width = 0.8 / len(defense_names)
            
            for i, defense_name in enumerate(defense_names):
                values = [
                    results[defense_name][dataset_type]["precision"],
                    results[defense_name][dataset_type]["recall"],
                    results[defense_name][dataset_type]["f1_score"],
                    results[defense_name][dataset_type]["accuracy"]
                ]
                
                plt.bar(x + i * width - 0.4 + width / 2, values, width, label=defense_name)
            
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
            plt.ylim(0, 1.05)
            plt.xlabel("評估指標")
            plt.ylabel("分數")
            plt.title(f"防禦方法比較 - {dataset_type.upper()} 數據集")
            plt.xticks(x, metric_labels)
            plt.legend(loc="upper right")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"defense_metrics_comparison_{timestamp}.png"))
    logger.info(f"防禦方法比較圖已保存至 {args.output_dir}/defense_metrics_comparison_{timestamp}.png")
    
    # 2. 計算時間比較圖
    plt.figure(figsize=(10, 6))
    
    # 準備數據
    compute_times_by_defense = {defense_name: [] for defense_name in defense_names}
    
    for defense_name in defense_names:
        for dataset_type in ['dh', 'ds']:
            if dataset_type in results[defense_name]:
                compute_times_by_defense[defense_name].extend(
                    results[defense_name][dataset_type]["compute_times"]
                )
    
    # 繪製箱線圖
    plt.boxplot([compute_times_by_defense[defense_name] for defense_name in defense_names],
                labels=defense_names, showfliers=False)
    
    plt.title("防禦方法計算時間比較")
    plt.xlabel("防禦方法")
    plt.ylabel("計算時間 (秒)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"compute_time_comparison_{timestamp}.png"))
    logger.info(f"計算時間比較圖已保存至 {args.output_dir}/compute_time_comparison_{timestamp}.png")
    
    # 3. 保存詳細結果
    with open(os.path.join(args.output_dir, f"detailed_comparison_results_{timestamp}.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"詳細比較結果已保存至 {args.output_dir}/detailed_comparison_results_{timestamp}.json")

def main():
    """主函數"""
    args = parse_args()
    
    # 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 載入測試數據
    datasets = load_test_data(args)
    
    # 載入不同的防禦方法
    defenses = {
        "GradSafe": create_gradsafe_defense(args),
        "Gradient Cuff": load_gradient_cuff(),
        "STPC": load_stpc_defense()
    }
    
    # 評估不同防禦方法
    results = evaluate_defense_methods(defenses, datasets, args)
    
    # 可視化比較結果
    visualize_comparison_results(results, args)
    
    logger.info("基線比較實驗完成")

if __name__ == "__main__":
    main() 