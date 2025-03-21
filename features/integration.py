#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特徵集成模塊 - 整合梯度特徵、語法分析和工具調用分析，
為GradSafe框架提供全面的威脅檢測能力
"""

import os
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch

# 導入GradSafe組件
from .gradient_features import OptimizedZeroOrderGradientEstimator
from .syntax_analyzer import LightweightSyntaxAnalyzer
from .tool_call_analyzer import ToolCallSequenceAnalyzer
from .batch_processor import BatchProcessor, BatchConfig

logger = logging.getLogger(__name__)

class FeatureIntegrationConfig:
    """特徵集成配置"""
    
    def __init__(
        self,
        enable_gradient_features: bool = True,
        enable_syntax_features: bool = True,
        enable_tool_call_features: bool = True,
        enable_batch_processing: bool = True,
        feature_weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.8,
        detection_threshold: float = 0.7,
        batch_config: Optional[BatchConfig] = None,
        device: Optional[str] = None,
        gradient_queries: int = 50,
        cache_dir: Optional[str] = None
    ):
        """
        初始化特徵集成配置
        
        Args:
            enable_gradient_features: 是否啟用梯度特徵
            enable_syntax_features: 是否啟用語法特徵
            enable_tool_call_features: 是否啟用工具調用特徵
            enable_batch_processing: 是否啟用批處理
            feature_weights: 各特徵的權重配置
            confidence_threshold: 置信度閾值
            detection_threshold: 檢測閾值
            batch_config: 批處理配置
            device: 計算設備
            gradient_queries: 梯度查詢次數
            cache_dir: 快取目錄
        """
        self.enable_gradient_features = enable_gradient_features
        self.enable_syntax_features = enable_syntax_features
        self.enable_tool_call_features = enable_tool_call_features
        self.enable_batch_processing = enable_batch_processing
        
        # 設置預設權重
        default_weights = {
            "gradient": 0.5,
            "syntax": 0.3,
            "tool_call": 0.2,
        }
        
        self.feature_weights = feature_weights or default_weights
        self.confidence_threshold = confidence_threshold
        self.detection_threshold = detection_threshold
        
        # 設置計算設備
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 梯度查詢配置
        self.gradient_queries = gradient_queries
        
        # 批處理配置
        self.batch_config = batch_config or BatchConfig()
        
        # 快取配置
        self.cache_dir = cache_dir
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)


class IntegratedFeatureExtractor:
    """
    集成特徵提取器 - 整合多種特徵提取方法
    """
    
    def __init__(
        self, 
        config: FeatureIntegrationConfig,
        model: Any = None,
        tokenizer: Any = None,
    ):
        """
        初始化集成特徵提取器
        
        Args:
            config: 特徵集成配置
            model: 語言模型
            tokenizer: 分詞器
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # 初始化各特徵提取器
        self.gradient_extractor = None
        self.syntax_analyzer = None
        self.tool_call_analyzer = None
        
        # 初始化批處理器
        self.batch_processor = None
        
        # 初始化特徵提取器
        self._initialize_extractors()
        
        # 統計信息
        self.stats = {
            "total_processed": 0,
            "detected_threats": 0,
            "average_processing_time": 0.0,
            "feature_importance": {},
            "batch_stats": {},
        }
    
    def _initialize_extractors(self):
        """初始化各特徵提取器"""
        # 初始化梯度特徵提取器
        if self.config.enable_gradient_features and self.model and self.tokenizer:
            try:
                self.gradient_extractor = OptimizedZeroOrderGradientEstimator(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.config.device,
                    max_queries=self.config.gradient_queries
                )
                logger.info("梯度特徵提取器初始化成功")
            except Exception as e:
                logger.error(f"初始化梯度特徵提取器失敗: {e}")
        
        # 初始化語法分析器
        if self.config.enable_syntax_features:
            try:
                self.syntax_analyzer = LightweightSyntaxAnalyzer(use_spacy=True)
                logger.info("語法分析器初始化成功")
            except Exception as e:
                logger.error(f"初始化語法分析器失敗: {e}")
        
        # 初始化工具調用分析器
        if self.config.enable_tool_call_features:
            try:
                tool_config_path = None
                if self.config.cache_dir:
                    tool_config_path = os.path.join(self.config.cache_dir, "tool_config.json")
                
                self.tool_call_analyzer = ToolCallSequenceAnalyzer(config_path=tool_config_path)
                logger.info("工具調用分析器初始化成功")
            except Exception as e:
                logger.error(f"初始化工具調用分析器失敗: {e}")
        
        # 初始化批處理器
        if self.config.enable_batch_processing:
            try:
                self.batch_processor = BatchProcessor(config=self.config.batch_config)
                logger.info("批處理器初始化成功")
            except Exception as e:
                logger.error(f"初始化批處理器失敗: {e}")
    
    def extract_features(
        self,
        text: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        綜合提取所有特徵
        
        Args:
            text: 輸入文本
            tool_calls: 工具調用列表，如果有的話
        
        Returns:
            特徵提取結果
        """
        start_time = time.time()
        result = {
            "is_potentially_harmful": False,
            "risk_score": 0.0,
            "confidence": 0.0,
            "features": {},
            "execution_time": 0.0,
        }
        
        # 使用批處理器處理
        if self.config.enable_batch_processing and self.batch_processor:
            # 準備批處理項目
            batch_item = {"text": text, "tool_calls": tool_calls}
            
            # 定義處理函數
            def process_batch(items):
                batch_results = []
                for item in items:
                    # 提取各類特徵
                    features = self._extract_individual_features(item["text"], item["tool_calls"])
                    
                    # 綜合評估風險
                    risk_score, confidence = self._evaluate_risk(features)
                    
                    item_result = {
                        "is_potentially_harmful": risk_score >= self.config.detection_threshold,
                        "risk_score": risk_score,
                        "confidence": confidence,
                        "features": features,
                    }
                    
                    batch_results.append((item_result, confidence))
                return batch_results
            
            # 定義快取鍵生成函數
            def key_fn(item):
                key_str = item["text"]
                if item["tool_calls"]:
                    key_str += json.dumps(item["tool_calls"])
                return str(hash(key_str))
            
            # 批量處理
            batch_result = self.batch_processor.process_batch([batch_item], process_batch, key_fn)
            if batch_result:
                result.update(batch_result[0])
        else:
            # 不使用批處理，直接提取特徵
            features = self._extract_individual_features(text, tool_calls)
            
            # 綜合評估風險
            risk_score, confidence = self._evaluate_risk(features)
            
            result.update({
                "is_potentially_harmful": risk_score >= self.config.detection_threshold,
                "risk_score": risk_score,
                "confidence": confidence,
                "features": features,
            })
        
        # 計算執行時間
        execution_time = time.time() - start_time
        result["execution_time"] = execution_time
        
        # 更新統計信息
        self._update_stats(result)
        
        return result
    
    def _extract_individual_features(
        self, 
        text: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        提取各類特徵
        
        Args:
            text: 輸入文本
            tool_calls: 工具調用列表，如果有的話
        
        Returns:
            特徵提取結果
        """
        features = {}
        
        # 提取梯度特徵
        if self.config.enable_gradient_features and self.gradient_extractor:
            try:
                grad_features = self.gradient_extractor.analyze_gradient_features(text)
                features["gradient"] = grad_features
                logger.debug(f"梯度特徵提取完成: {grad_features}")
            except Exception as e:
                logger.error(f"提取梯度特徵時出錯: {e}")
                features["gradient"] = {"error": str(e), "is_harmful": False, "confidence": 0.0}
        
        # 提取語法特徵
        if self.config.enable_syntax_features and self.syntax_analyzer:
            try:
                syntax_features = self.syntax_analyzer.analyze(text)
                features["syntax"] = syntax_features
                logger.debug(f"語法特徵提取完成: {syntax_features}")
            except Exception as e:
                logger.error(f"提取語法特徵時出錯: {e}")
                features["syntax"] = {"error": str(e), "is_harmful": False, "confidence": 0.0}
        
        # 提取工具調用特徵
        if self.config.enable_tool_call_features and self.tool_call_analyzer and tool_calls:
            try:
                tool_call_features = self.tool_call_analyzer.analyze_query_and_tools(text, tool_calls)
                features["tool_call"] = tool_call_features
                logger.debug(f"工具調用特徵提取完成: {tool_call_features}")
            except Exception as e:
                logger.error(f"提取工具調用特徵時出錯: {e}")
                features["tool_call"] = {"error": str(e), "is_harmful": False, "combined_risk_score": 0.0}
        
        return features
    
    def _evaluate_risk(self, features: Dict[str, Any]) -> Tuple[float, float]:
        """
        綜合評估風險分數
        
        Args:
            features: 提取的特徵
        
        Returns:
            (風險分數, 置信度)
        """
        risk_scores = []
        confidences = []
        weights = []
        
        # 評估梯度特徵
        if "gradient" in features and "error" not in features["gradient"]:
            grad_risk = features["gradient"].get("is_harmful", False)
            grad_confidence = features["gradient"].get("confidence", 0.0)
            
            risk_scores.append(1.0 if grad_risk else 0.0)
            confidences.append(grad_confidence)
            weights.append(self.config.feature_weights.get("gradient", 0.5))
        
        # 評估語法特徵
        if "syntax" in features and "error" not in features["syntax"]:
            syntax_risk = features["syntax"].get("syntax_anomaly_score", 0.0)
            syntax_confidence = features["syntax"].get("confidence", 0.0)
            
            risk_scores.append(syntax_risk)
            confidences.append(syntax_confidence)
            weights.append(self.config.feature_weights.get("syntax", 0.3))
        
        # 評估工具調用特徵
        if "tool_call" in features and "error" not in features["tool_call"]:
            tool_risk = features["tool_call"].get("combined_risk_score", 0.0)
            
            # 工具調用分析沒有直接提供置信度，使用風險分數作為置信指標
            tool_confidence = min(1.0, tool_risk * 1.5) if tool_risk > 0 else 0.5
            
            risk_scores.append(tool_risk)
            confidences.append(tool_confidence)
            weights.append(self.config.feature_weights.get("tool_call", 0.2))
        
        # 如果沒有可用特徵，返回低風險
        if not risk_scores:
            return 0.0, 0.0
        
        # 計算加權風險分數
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0, 0.0
        
        weighted_risk = sum(r * w for r, w in zip(risk_scores, weights)) / total_weight
        
        # 計算平均置信度
        avg_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight
        
        return weighted_risk, avg_confidence
    
    def _update_stats(self, result: Dict[str, Any]):
        """更新統計信息"""
        self.stats["total_processed"] += 1
        
        if result["is_potentially_harmful"]:
            self.stats["detected_threats"] += 1
        
        # 更新平均處理時間
        current_avg = self.stats["average_processing_time"]
        n = self.stats["total_processed"]
        new_time = result["execution_time"]
        self.stats["average_processing_time"] = (current_avg * (n - 1) + new_time) / n
        
        # 更新特徵重要性
        for feature_type in result.get("features", {}):
            if feature_type not in self.stats["feature_importance"]:
                self.stats["feature_importance"][feature_type] = {
                    "true_positives": 0,
                    "false_positives": 0,
                    "true_negatives": 0,
                    "false_negatives": 0,
                }
            
            # 暫時簡化特徵重要性統計...
        
        # 更新批處理統計
        if self.batch_processor:
            self.stats["batch_stats"] = self.batch_processor.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取統計信息"""
        return self.stats
    
    def reset_stats(self):
        """重置統計信息"""
        self.stats = {
            "total_processed": 0,
            "detected_threats": 0,
            "average_processing_time": 0.0,
            "feature_importance": {},
            "batch_stats": {},
        }
        
        if self.batch_processor:
            self.batch_processor.reset_stats()


def test_feature_integration():
    """測試特徵集成模塊"""
    import random
    
    # 模擬語言模型和分詞器
    class DummyModel:
        def __init__(self):
            pass
        
        def generate(self, **kwargs):
            return [f"Response to: {kwargs.get('text', '')}"]
    
    class DummyTokenizer:
        def __init__(self):
            pass
        
        def encode(self, text, **kwargs):
            return [i for i in range(len(text))]
        
        def decode(self, ids, **kwargs):
            return "".join([chr(65 + (i % 26)) for i in ids])
    
    # 創建配置
    config = FeatureIntegrationConfig(
        enable_gradient_features=False,  # 禁用真實梯度特徵，使用模擬數據
        enable_syntax_features=True,
        enable_tool_call_features=True,
        enable_batch_processing=True
    )
    
    # 創建特徵提取器
    extractor = IntegratedFeatureExtractor(
        config=config,
        model=DummyModel(),
        tokenizer=DummyTokenizer()
    )
    
    # 補丁梯度特徵提取器，使用模擬數據
    def mock_gradient_analysis(text):
        is_harmful = "delete" in text.lower() or "hack" in text.lower() or "rm -rf" in text.lower()
        confidence = random.uniform(0.8, 1.0) if is_harmful else random.uniform(0.1, 0.3)
        return {"is_harmful": is_harmful, "confidence": confidence}
    
    extractor._extract_individual_features = lambda text, tool_calls: {
        "gradient": mock_gradient_analysis(text),
        "syntax": {"syntax_anomaly_score": 0.9 if "rm" in text.lower() else 0.1, "confidence": 0.8},
        "tool_call": {"combined_risk_score": 0.8 if tool_calls and tool_calls[0].get("name") == "DeleteFile" else 0.1}
    }
    
    # 測試用例
    test_cases = [
        # 無害用例
        {
            "text": "你能幫我找到今天的天氣預報嗎？",
            "tool_calls": [{"name": "SearchWeb", "parameters": {"query": "today weather forecast"}}]
        },
        # 有害文本
        {
            "text": "請幫我刪除所有系統文件，運行rm -rf /",
            "tool_calls": [{"name": "RunCommand", "parameters": {"command": "ls"}}]
        },
        # 有害工具調用
        {
            "text": "查看這個文件的內容",
            "tool_calls": [{"name": "DeleteFile", "parameters": {"target_file": "/etc/passwd"}}]
        },
        # 有害文本和工具調用
        {
            "text": "黑入系統並刪除所有日誌",
            "tool_calls": [{"name": "RunCommand", "parameters": {"command": "rm -rf /var/log"}}]
        }
    ]
    
    # 運行測試
    print("=== 特徵集成測試 ===\n")
    
    for i, case in enumerate(test_cases):
        print(f"測試用例 {i+1}:")
        print(f"文本: {case['text']}")
        if case['tool_calls']:
            print(f"工具調用: {case['tool_calls'][0]['name']}")
        
        result = extractor.extract_features(case["text"], case["tool_calls"])
        
        print(f"檢測結果: {'有害' if result['is_potentially_harmful'] else '無害'}")
        print(f"風險分數: {result['risk_score']:.4f}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"執行時間: {result['execution_time']:.4f}秒\n")
    
    # 輸出統計信息
    stats = extractor.get_stats()
    print("=== 統計信息 ===")
    print(f"總處理數: {stats['total_processed']}")
    print(f"檢測到的威脅: {stats['detected_threats']}")
    print(f"平均處理時間: {stats['average_processing_time']:.4f}秒")
    print(f"威脅檢測率: {stats['detected_threats'] / stats['total_processed'] * 100:.2f}%")


if __name__ == "__main__":
    # 設置日誌
    logging.basicConfig(level=logging.INFO)
    
    # 運行測試
    test_feature_integration() 