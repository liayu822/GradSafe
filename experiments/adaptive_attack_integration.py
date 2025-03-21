#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
適應性攻擊整合工具 - 將AdaptiveAttackAgent中的攻擊方法整合到GradSafe測試框架中
"""

import os
import sys
import logging
import json
import torch
import argparse
from typing import Dict, List, Any, Optional, Tuple
import importlib.util
from tqdm import tqdm

# 將項目根目錄添加到路徑中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("attack_integration.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="適應性攻擊整合工具")
    
    parser.add_argument("--model_path", type=str, 
                        default="vicuna-7b-v1.5",
                        help="模型路徑")
    
    parser.add_argument("--attack_method", type=str, 
                        choices=["GCG", "MGCG_ST", "MGCG_DT"], 
                        default="GCG",
                        help="攻擊方法")
    
    parser.add_argument("--dataset", type=str, 
                        choices=["dh", "ds"], 
                        default="dh",
                        help="數據集類型")
    
    parser.add_argument("--data_setting", type=str, 
                        choices=["base", "base_subset", "enhanced"], 
                        default="base_subset",
                        help="數據設置")
    
    parser.add_argument("--num_steps", type=int, 
                        default=100,
                        help="攻擊優化步數")
    
    parser.add_argument("--adv_string_position", type=str, 
                        choices=["prefix", "suffix"], 
                        default="suffix",
                        help="對抗字符串位置")
    
    parser.add_argument("--adv_string_init", type=str, 
                        default=" ! ! ! ! ! ! ! ! ! !",
                        help="對抗字符串初始值")
    
    parser.add_argument("--search_width", type=int, 
                        default=256,
                        help="搜索寬度")
    
    parser.add_argument("--output_dir", type=str, 
                        default="results/adaptive_attacks",
                        help="輸出目錄")
    
    parser.add_argument("--num_samples", type=int, 
                        default=10,
                        help="處理的樣本數")
    
    return parser.parse_args()

def load_model_and_tokenizer(model_path: str) -> Tuple[Any, Any]:
    """載入模型和分詞器"""
    logger.info(f"載入模型: {model_path}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"載入模型失敗: {e}")
        raise

def load_attack_method(attack_method: str):
    """載入攻擊方法"""
    logger.info(f"載入攻擊方法: {attack_method}")
    
    try:
        # 使用AdaptiveAttackAgent中的方法
        sys.path.append(os.path.abspath("./AdaptiveAttackAgent"))
        
        # 動態導入攻擊方法
        if attack_method == "GCG":
            from AdaptiveAttackAgent.attacks.gcg import GCG
            return GCG
        elif attack_method == "MGCG_ST":
            from AdaptiveAttackAgent.attacks.multi_gcg_same_tokenizer import MGCG_ST
            return MGCG_ST
        elif attack_method == "MGCG_DT":
            from AdaptiveAttackAgent.attacks.multi_gcg_different_tokenizer import MGCG_DT
            return MGCG_DT
        else:
            raise ValueError(f"未知攻擊方法: {attack_method}")
    except Exception as e:
        logger.error(f"載入攻擊方法失敗: {e}")
        raise

def load_test_data(dataset: str, data_setting: str, num_samples: int) -> List[Dict]:
    """載入測試數據"""
    logger.info(f"載入數據: {dataset}, {data_setting}")
    
    try:
        data_path = f"./InjecAgent/data/test_cases_{dataset}_{data_setting}.json"
        
        with open(data_path, "r") as f:
            data = json.load(f)
        
        # 限制樣本數
        data = data[:num_samples]
        
        logger.info(f"載入 {len(data)} 條測試數據")
        return data
    except Exception as e:
        logger.error(f"載入數據失敗: {e}")
        raise

def generate_adaptive_attacks(
    attack_class,
    model,
    tokenizer,
    test_data: List[Dict],
    attack_config: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """生成適應性攻擊數據"""
    logger.info("開始生成適應性攻擊...")
    
    # 創建攻擊實例
    attacker = attack_class(
        model=model,
        tokenizer=tokenizer,
        num_steps=attack_config.get("num_steps", 100),
        adv_string_init=attack_config.get("adv_string_init", " ! ! ! ! !"),
        search_width=attack_config.get("search_width", 256),
    )
    
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"{attack_config['attack_method']}_{attack_config['dataset']}_results.json")
    
    # 處理每個測試樣本
    results = []
    for idx, item in enumerate(tqdm(test_data, desc="生成適應性攻擊")):
        # 準備數據字典
        data_dict = {
            "user_query": item.get("User Instruction", ""),
            "attacker_instruction": item.get("Attacker Instruction", ""),
            "attacker_tools": item.get("Attacker Tools", []),
            "adv_string_position": attack_config.get("adv_string_position", "suffix"),
            "index": idx
        }
        
        # 生成適應性攻擊
        try:
            attack_result = attacker.train_adv_string(data_dict, result_file)
            
            # 合併原始數據和攻擊結果
            combined_result = {**item, **attack_result, "attack_config": attack_config}
            results.append(combined_result)
            
            # 每10個樣本保存一次中間結果
            if (idx + 1) % 10 == 0:
                with open(result_file, "w") as f:
                    json.dump(results, f, indent=2)
        
        except Exception as e:
            logger.error(f"處理樣本 {idx} 失敗: {e}")
            continue
    
    # 保存最終結果
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"適應性攻擊生成完成，結果保存至: {result_file}")
    return {"results": results, "result_file": result_file}

def integrate_with_gradsafe_experiments(attack_results: Dict[str, Any], args) -> None:
    """整合到GradSafe實驗框架"""
    logger.info("整合攻擊結果到GradSafe測試框架...")
    
    try:
        # 為GradSafe實驗準備數據
        adaptive_attack_dir = os.path.join(args.output_dir, "adaptive_data")
        os.makedirs(adaptive_attack_dir, exist_ok=True)
        
        # 創建適應性攻擊數據集
        adaptive_data_file = os.path.join(
            adaptive_attack_dir, 
            f"adaptive_{args.attack_method}_{args.dataset}_{args.data_setting}.json"
        )
        
        # 處理攻擊結果為GradSafe可用格式
        processed_data = []
        
        for idx, item in enumerate(attack_results["results"]):
            # 提取對抗字符串
            adv_string = item.get("adv_string", "")
            
            # 修改原始查詢，添加對抗字符串
            if args.adv_string_position == "prefix":
                modified_query = adv_string + " " + item.get("User Instruction", "")
            else:  # suffix
                modified_query = item.get("User Instruction", "") + " " + adv_string
            
            # 創建新的項目
            new_item = {
                "User Instruction": modified_query,
                "Attacker Instruction": item.get("Attacker Instruction", ""),
                "Attacker Tools": item.get("Attacker Tools", []),
                "Original Query": item.get("User Instruction", ""),
                "Adversarial String": adv_string,
                "Attack Method": args.attack_method,
                "Attack Position": args.adv_string_position,
                "Expected Achievements": item.get("Expected Achievements", ""),
                "Attack Type": item.get("Attack Type", "")
            }
            
            # 如果有工具參數
            if "Attacker Tool Parameters" in item:
                new_item["Attacker Tool Parameters"] = item["Attacker Tool Parameters"]
            
            processed_data.append(new_item)
        
        # 保存處理後的數據
        with open(adaptive_data_file, "w") as f:
            json.dump(processed_data, f, indent=2)
        
        logger.info(f"已將攻擊結果整合到GradSafe測試框架，數據保存至: {adaptive_data_file}")
        
        # 生成運行適應性攻擊評估實驗的命令
        cmd = (
            f"python experiments/adaptive_attack_evaluation.py "
            f"--attack_method={args.attack_method} "
            f"--dataset={args.dataset} "
            f"--data_subset={args.data_setting} "
            f"--model={args.model_path.split('/')[-1]} "
            f"--output_dir={args.output_dir}/evaluation_results"
        )
        
        logger.info(f"可運行以下命令進行評估:\n{cmd}")
        
    except Exception as e:
        logger.error(f"整合到GradSafe測試框架失敗: {e}")
        raise

def main():
    """主函數"""
    args = parse_args()
    
    try:
        # 1. 載入模型和分詞器
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        
        # 2. 載入攻擊方法
        attack_class = load_attack_method(args.attack_method)
        
        # 3. 載入測試數據
        test_data = load_test_data(args.dataset, args.data_setting, args.num_samples)
        
        # 4. 準備攻擊配置
        attack_config = {
            "attack_method": args.attack_method,
            "dataset": args.dataset,
            "data_setting": args.data_setting,
            "num_steps": args.num_steps,
            "adv_string_position": args.adv_string_position,
            "adv_string_init": args.adv_string_init,
            "search_width": args.search_width
        }
        
        # 5. 生成適應性攻擊
        attack_results = generate_adaptive_attacks(
            attack_class=attack_class,
            model=model,
            tokenizer=tokenizer,
            test_data=test_data,
            attack_config=attack_config,
            output_dir=args.output_dir
        )
        
        # 6. 整合到GradSafe實驗框架
        integrate_with_gradsafe_experiments(attack_results, args)
        
        logger.info("適應性攻擊整合完成！")
        
    except Exception as e:
        logger.error(f"執行過程中發生錯誤: {e}")
        raise

if __name__ == "__main__":
    main() 