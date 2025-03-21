# GradSafe 實驗框架

本目錄包含用於測試和評估 GradSafe 防禦系統的各種實驗腳本。這些腳本可以用來測量防禦系統的有效性、效率以及與其他基線方法的比較。

## 檔案結構

- `adaptive_attack_evaluation.py`: 評估 GradSafe 框架對抵抗適應性攻擊的有效性
- `adaptive_attack_integration.py`: 將 AdaptiveAttackAgent 中的攻擊方法整合到 GradSafe 測試框架中
- `run_baseline_comparison.py`: 比較 GradSafe 與其他基線方法 (如 Gradient Cuff, STPC) 的防禦效果
- `create_benchmark.py`: 生成評估 GradSafe 框架的標準測試數據集
- `gradient_experiments.py`: 對 GradSafe 的梯度特徵進行實驗分析
- `syntax_features.py`: 對 GradSafe 的語法特徵進行實驗分析
- `tool_call_experiments.py`: 對 GradSafe 的工具調用行為進行實驗分析

## 測試流程

完整的 GradSafe 測試框架評估流程如下：

1. 生成基準測試數據集
2. 集成適應性攻擊方法
3. 運行適應性攻擊評估
4. 運行基線方法比較
5. 分析結果

### 1. 生成基準測試數據集

使用 `create_benchmark.py` 腳本生成基準測試數據集：

```bash
python experiments/create_benchmark.py --output_dir data/benchmark --num_samples 1000 --benign_ratio 0.6
```

參數說明：
- `--output_dir`: 輸出目錄
- `--num_samples`: 生成的樣本總數
- `--benign_ratio`: 良性樣本比例 (0.0-1.0)
- `--use_injecagent`: 是否使用 InjecAgent 數據
- `--use_adaptive`: 是否包含適應性攻擊樣本
- `--random_seed`: 隨機種子

### 2. 集成適應性攻擊方法

使用 `adaptive_attack_integration.py` 腳本將 AdaptiveAttackAgent 中的攻擊方法整合到 GradSafe 測試框架中：

```bash
python experiments/adaptive_attack_integration.py --model_path vicuna-7b-v1.5 --attack_method GCG --dataset dh --num_samples 10
```

參數說明：
- `--model_path`: 模型路徑
- `--attack_method`: 攻擊方法 (GCG, MGCG_ST, MGCG_DT)
- `--dataset`: 數據集類型 (dh, ds)
- `--data_setting`: 數據設置 (base, base_subset, enhanced)
- `--num_steps`: 攻擊優化步數
- `--adv_string_position`: 對抗字符串位置 (prefix, suffix)
- `--adv_string_init`: 對抗字符串初始值
- `--output_dir`: 輸出目錄
- `--num_samples`: 處理的樣本數

### 3. 運行適應性攻擊評估

使用 `adaptive_attack_evaluation.py` 腳本評估 GradSafe 框架對抵抗適應性攻擊的有效性：

```bash
python experiments/adaptive_attack_evaluation.py --mode asr_comparison --attack_method GCG --defense_mode all --dataset both
```

參數說明：
- `--mode`: 實驗模式 (asr_comparison, efficiency, ablation)
- `--device`: 運行設備
- `--model`: 使用的基礎 LLM 模型
- `--attack_method`: 攻擊方法 (GCG, MGCG_ST, MGCG_DT, TGCG, all)
- `--defense_mode`: 防禦模式 (gradient_only, behavior_only, full, all)
- `--dataset`: 數據集類型 (ds, dh, both)
- `--data_subset`: 數據子集 (base, base_subset, enhanced)
- `--output_dir`: 輸出目錄
- `--num_samples`: 每種攻擊方法使用的樣本數
- `--max_steps`: 攻擊優化的最大步數
- `--enable_early_exit`: 是否啟用提前退出機制
- `--enable_resource_adaption`: 是否啟用資源自適應機制
- `--baseline_comparison`: 是否與基線方法進行比較

### 4. 運行基線方法比較

使用 `run_baseline_comparison.py` 腳本比較 GradSafe 與其他基線方法的防禦效果：

```bash
python experiments/run_baseline_comparison.py --dataset both --data_subset base_subset --num_samples 100
```

參數說明：
- `--device`: 運行設備
- `--dataset`: 數據集類型 (ds, dh, both)
- `--data_subset`: 數據子集 (base, base_subset, enhanced)
- `--num_samples`: 每個數據集使用的樣本數
- `--output_dir`: 輸出目錄

### 5. 特徵分析實驗

運行特徵分析實驗以評估不同特徵對防禦效果的影響：

```bash
# 梯度特徵實驗
python experiments/gradient_experiments.py

# 語法特徵實驗
python experiments/syntax_features.py

# 工具調用行為實驗
python experiments/tool_call_experiments.py
```

## 結果分析

實驗結果將保存在指定的輸出目錄中，包括：

1. 詳細的 JSON 格式結果文件
2. 攻擊成功率比較圖
3. 計算效率比較圖
4. 特徵重要性分析圖

您可以使用這些結果來評估 GradSafe 框架的性能和有效性，並與其他防禦方法進行比較。

## 運行完整評估流程

以下是運行完整評估流程的一個範例：

```bash
# 1. 生成基準測試數據集
python experiments/create_benchmark.py --output_dir data/benchmark --num_samples 1000

# 2. 集成 GCG 攻擊方法
python experiments/adaptive_attack_integration.py --model_path vicuna-7b-v1.5 --attack_method GCG --dataset dh --num_samples 50

# 3. 集成 MGCG_ST 攻擊方法
python experiments/adaptive_attack_integration.py --model_path vicuna-7b-v1.5 --attack_method MGCG_ST --dataset dh --num_samples 50

# 4. 運行適應性攻擊評估
python experiments/adaptive_attack_evaluation.py --mode asr_comparison --attack_method all --defense_mode all --dataset both

# 5. 運行基線方法比較
python experiments/run_baseline_comparison.py --dataset both --data_subset base_subset --num_samples 100

# 6. 運行特徵分析實驗
python experiments/gradient_experiments.py
python experiments/syntax_features.py
python experiments/tool_call_experiments.py
```

## 注意事項

- 確保所有必要的依賴庫已正確安裝
- 運行適應性攻擊生成可能需要較長時間和較多計算資源
- 調整參數以適應您的計算環境和研究需求
- 對於生產環境，建議增加更多測試樣本以獲得更可靠的結果 