# GradSafe: 基於梯度特徵的LLM攻擊防護框架

GradSafe是一個強大的框架，用於檢測和防禦針對大型語言模型(LLM)的惡意注入和提示詞攻擊。本框架基於模型梯度特徵分析，同時結合語法分析和工具調用序列監控，提供全方位的防護能力。

## 主要特點

- **梯度特徵分析**：通過分析輸入文本的梯度敏感性，有效識別惡意提示詞和注入攻擊
- **優化的零階梯度估計**：減少查詢次數的同時保持高檢測能力
- **輕量級語法分析**：專注於提取與惡意行為相關的關鍵語法特徵
- **工具調用序列監控**：識別可疑的工具調用模式，防止數據洩露和系統破壞
- **批處理優化與早退機制**：減少處理延遲，提高LLM代理的響應能力
- **資源自適應調整**：根據系統資源使用情況動態調整處理策略

## 系統架構

GradSafe框架由以下主要模塊組成：

1. **特徵提取模塊**
   - `gradient_features.py`: 實現梯度特徵提取，包含優化的零階梯度估計
   - `syntax_analyzer.py`: 輕量級語法分析器，專注於提取關鍵語法特徵
   - `tool_call_analyzer.py`: 工具調用序列分析系統，識別可疑的工具調用模式

2. **處理優化模塊**
   - `batch_processor.py`: 實現批處理優化和提前退出機制，減少處理延遲

3. **防禦機制**
   - `defense_strategies.py`: 實現多層防禦策略
   - `detector.py`: 綜合特徵檢測器，融合多種特徵分析結果

4. **評估框架**
   - `adaptive_attack_evaluation.py`: 評估框架對自適應攻擊的防禦效果
   - `run_baseline_comparison.py`: 與基準方法的比較評估
   - `adaptive_attack_integration.py`: 集成自適應攻擊方法

## 關鍵技術

### 優化的零階梯度估計

GradSafe框架使用優化的零階梯度估計技術，通過以下策略顯著減少所需的查詢次數：

- **自適應抽樣**：根據初始梯度估計動態調整抽樣點
- **層次抽樣**：專注於重要位置的梯度特徵
- **早退機制**：當梯度特徵足夠明確時提前停止估計
- **批處理加速**：減少整體延遲

### 輕量級語法分析

GradSafe的語法分析器專注於提取與惡意行為相關的關鍵語法特徵：

- **命令模式檢測**：識別可能含有系統命令的結構
- **注入模式檢測**：識別SQL注入、代碼注入等模式
- **工具調用模式**：分析工具調用的結構和語法特徵
- **上下文一致性分析**：檢測輸入與上下文的不協調性

### 工具調用序列分析

GradSafe的工具調用分析系統能夠識別可疑的工具調用模式：

- **高風險組合識別**：檢測潛在危險的工具調用組合
- **數據洩露模式檢測**：識別可能導致敏感信息洩露的調用序列
- **系統破壞模式識別**：檢測可能導致系統破壞的操作序列
- **查詢-工具一致性分析**：檢查用戶查詢與工具調用的一致性

### 批處理優化與資源自適應

GradSafe框架通過批處理優化和資源自適應機制，提高處理效率：

- **動態批量大小調整**：根據資源使用情況調整批量大小
- **提前退出機制**：根據置信度提前結束處理
- **結果快取**：避免重複計算，提高效率
- **資源監控**：實時監控系統資源使用情況，優化處理策略

## 安裝與使用

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from gradsafe import GradSafeDefense

# 初始化防禦系統
defense = GradSafeDefense(model="vicuna-7b-v1.5")

# 檢查輸入是否為潛在攻擊
result = defense.check_input("用戶輸入文本")

if result.is_potential_attack:
    print(f"警告: 檢測到潛在攻擊, 風險分數: {result.risk_score}")
    print(f"攻擊類型: {result.attack_type}")
    print(f"建議: {result.suggested_action}")
else:
    print("輸入安全")
```

### 高級配置

```python
from gradsafe import GradSafeDefense, GradSafeConfig

# 自定義配置
config = GradSafeConfig(
    gradient_estimation_queries=50,  # 設置梯度估計的最大查詢次數
    enable_syntax_analysis=True,     # 啟用語法分析
    enable_tool_call_analysis=True,  # 啟用工具調用分析
    early_exit_threshold=0.95,       # 設置提前退出閾值
    batch_size=8,                    # 設置批處理大小
    adapt_to_resources=True          # 啟用資源自適應
)

# 使用自定義配置初始化防禦系統
defense = GradSafeDefense(model="vicuna-7b-v1.5", config=config)
```

## 實驗評估

### 評估自適應攻擊防禦效果

```bash
python experiments/adaptive_attack_evaluation.py --model vicuna-7b-v1.5 --attack_method GCG --defense_mode full
```

### 與基準方法比較

```bash
python experiments/run_baseline_comparison.py --defense_methods gradsafe gradient-cuff --dataset dh --num_samples 100
```

## 貢獻

歡迎提交問題報告、功能請求和代碼貢獻！詳細信息請參見 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 引用

如果您在研究中使用了GradSafe，請引用我們的論文：

```
@article{gradsafe2023,
  title={GradSafe: 基於梯度特徵的LLM攻擊防護框架},
  author={XXX},
  journal={XXX},
  year={2023}
}
```

## 許可證

本項目基於MIT許可證。詳見[LICENSE](LICENSE)文件。 