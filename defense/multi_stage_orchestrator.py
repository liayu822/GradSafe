#實現動態防禦流程管理
import logging
import time
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass
import traceback
import psutil

from ..features.gradient_features import GradientFeatureEnhancer
from ..features.behavior_features import BehaviorFeatureExtractor
from ..features.syntax_features import SyntaxFeatureExtractor

@dataclass
class DefenseConfig:
    """防禦機制的配置參數"""
    # 基本設置
    enable_grad_features: bool = True
    enable_behavior_features: bool = True
    enable_early_exit: bool = True
    enable_resource_adaption: bool = True
    
    # 閾值設置
    low_risk_threshold: float = 0.2
    medium_risk_threshold: float = 0.5
    high_risk_threshold: float = 0.8
    
    # 資源限制
    max_memory_usage_percent: float = 90.0  # 最大內存使用百分比
    max_compute_time_ms: int = 300  # 最大計算時間(毫秒)
    
    # 批處理設置
    batch_size: int = 32
    
    # 特徵提取器設置
    feature_dim: int = 768
    num_syntax_features: int = 50
    
    # 梯度擾動參數
    mu: float = 0.02
    sample_times: int = 10
    
    # 設備設置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 早退策略
    exit_after_gradient: bool = True  # 是否在梯度分析後可以提前退出
    exit_after_behavior: bool = True  # 是否在行為分析後可以提前退出
    
    # 記錄時間戳的設置
    log_timings: bool = True


class MultiStageDefenseOrchestrator:
    """
    輕量級階段性防禦架構 - 動態協調多層防禦
    
    主要功能：
    1. 風險適應性防禦：根據初步風險評估動態調整防禦深度
    2. 提前退出機制：在低風險情境下跳過部分防禦層
    3. 批量處理優化：實現梯度特徵和行為分析的批量處理
    4. 資源自適應調整：根據可用計算資源自動調整防禦強度
    """
    
    def __init__(
        self, 
        config: Optional[DefenseConfig] = None,
        gradient_enhancer: Optional[GradientFeatureEnhancer] = None,
        behavior_extractor: Optional[BehaviorFeatureExtractor] = None,
        syntax_extractor: Optional[SyntaxFeatureExtractor] = None
    ):
        """初始化防禦協調器
        
        Args:
            config: 防禦配置
            gradient_enhancer: 梯度特徵增強器
            behavior_extractor: 行為特徵提取器
            syntax_extractor: 語法特徵提取器
        """
        # 設置配置
        self.config = config or DefenseConfig()
        
        # 設置日誌記錄器
        self.logger = logging.getLogger(__name__)
        
        # 初始化特徵提取器和增強器
        if syntax_extractor is None:
            self.syntax_extractor = SyntaxFeatureExtractor(
                max_features=self.config.num_syntax_features,
                device=self.config.device
            )
        else:
            self.syntax_extractor = syntax_extractor
            
        if gradient_enhancer is None:
            self.gradient_enhancer = GradientFeatureEnhancer(
                feature_dim=self.config.feature_dim,
                num_syntax_features=self.config.num_syntax_features,
                batch_size=self.config.batch_size,
                mu=self.config.mu,
                sample_times=self.config.sample_times,
                device=self.config.device
            )
        else:
            self.gradient_enhancer = gradient_enhancer
            
        if behavior_extractor is None:
            self.behavior_extractor = BehaviorFeatureExtractor(
                max_features=20,
                device=self.config.device
            )
        else:
            self.behavior_extractor = behavior_extractor
            
        # 跟蹤執行時間和資源使用情況
        self.timings = {
            'gradient_analysis': [],
            'behavior_analysis': [],
            'total_defense': []
        }
        
        # 已處理的工具調用數量
        self.processed_calls = 0
        
        # 重置計數
        self.early_exits = 0
        self.full_defenses = 0
        
        self.logger.info("多階段防禦協調器初始化完成")
    
    def _check_resources(self) -> Dict[str, float]:
        """檢查當前可用的計算資源
        
        Returns:
            Dict[str, float]: 資源使用情況
        """
        try:
            # 獲取CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 獲取記憶體使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 獲取GPU使用率（如果有）
            gpu_percent = 0.0
            if torch.cuda.is_available():
                try:
                    # 這需要安裝pynvml或nvidia-ml-py
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = util.gpu
                except:
                    self.logger.warning("無法獲取GPU使用率，可能需要安裝pynvml")
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'gpu_percent': gpu_percent
            }
        except Exception as e:
            self.logger.error(f"檢查資源時出錯: {str(e)}")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'gpu_percent': 0.0
            }
    
    def _adjust_defense_parameters(self, resources: Dict[str, float]) -> None:
        """根據可用資源調整防禦參數
        
        Args:
            resources: 資源使用情況
        """
        if not self.config.enable_resource_adaption:
            return
        
        memory_percent = resources.get('memory_percent', 0.0)
        cpu_percent = resources.get('cpu_percent', 0.0)
        
        # 如果記憶體使用率過高，減少批處理大小
        if memory_percent > self.config.max_memory_usage_percent:
            old_batch_size = self.config.batch_size
            self.config.batch_size = max(1, self.config.batch_size // 2)
            self.logger.warning(f"記憶體使用率過高 ({memory_percent}%)，減少批處理大小: {old_batch_size} -> {self.config.batch_size}")
        
        # 如果CPU使用率過高，減少採樣次數
        if cpu_percent > 80:
            old_sample_times = self.config.sample_times
            self.config.sample_times = max(3, self.config.sample_times // 2)
            self.logger.warning(f"CPU使用率過高 ({cpu_percent}%)，減少採樣次數: {old_sample_times} -> {self.config.sample_times}")
        
        # 更新特徵提取器的參數
        if hasattr(self.gradient_enhancer, 'batch_size'):
            self.gradient_enhancer.batch_size = self.config.batch_size
        
        if hasattr(self.gradient_enhancer, 'sample_times'):
            self.gradient_enhancer.sample_times = self.config.sample_times
    
    def _should_exit_early(self, risk_level: str, stage: str) -> bool:
        """判斷是否應該提前退出防禦流程
        
        Args:
            risk_level: 風險等級
            stage: 當前防禦階段
            
        Returns:
            bool: 是否應該提前退出
        """
        # 如果未啟用提前退出，直接返回False
        if not self.config.enable_early_exit:
            return False
        
        # 根據風險等級和當前階段決定是否提前退出
        if risk_level == "low":
            return True
        
        if risk_level == "medium" and stage == "gradient":
            if not self.config.exit_after_gradient:
                return False
            
            # 隨機決定是否對中等風險進行深入檢查（保持20%的深入檢查率）
            return np.random.random() > 0.2
            
        if risk_level == "medium" and stage == "behavior":
            if not self.config.exit_after_behavior:
                return False
            
            # 隨機決定是否對中等風險進行深入檢查（保持10%的深入檢查率）
            return np.random.random() > 0.1
        
        # 高風險或關鍵風險不提前退出
        return False
    
    def analyze_gradient_features(
        self, 
        user_query: str,
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """分析梯度特徵
        
        Args:
            user_query: 用戶查詢
            loss_fn: 用於更新特徵重要性的損失函數
            
        Returns:
            Dict[str, Any]: 分析結果
        """
        start_time = time.time()
        
        try:
            # 提取語法特徵
            syntax_features = self.syntax_extractor.extract_features_tensor(user_query)
            
            # 生成梯度嵌入（假設為隨機向量，實際應從模型獲取）
            gradient_embedding = torch.randn(1, self.config.feature_dim, device=self.config.device)
            
            # 使用梯度特徵增強器處理
            enhanced_features = self.gradient_enhancer.process_batch(
                gradient_embedding,
                syntax_features.unsqueeze(0),
                loss_fn
            )
            
            # 計算風險分數（簡化版本）
            # 這裡可以實現更複雜的風險評估邏輯
            risk_score = torch.norm(enhanced_features).item() / self.config.feature_dim
            
            # 確定風險等級
            if risk_score < self.config.low_risk_threshold:
                risk_level = "low"
            elif risk_score < self.config.medium_risk_threshold:
                risk_level = "medium"
            elif risk_score < self.config.high_risk_threshold:
                risk_level = "high"
            else:
                risk_level = "critical"
            
            result = {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "enhanced_features": enhanced_features,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"梯度特徵分析出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            result = {
                "risk_score": 0.0,
                "risk_level": "unknown",
                "error": str(e),
                "success": False
            }
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        if self.config.log_timings:
            self.timings['gradient_analysis'].append(elapsed_ms)
            result["elapsed_ms"] = elapsed_ms
            
        return result
    
    def analyze_behavior_features(
        self, 
        tool_calls: List[Dict],
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """分析行為特徵
        
        Args:
            tool_calls: 工具調用列表
            user_query: 用戶查詢
            
        Returns:
            Dict[str, Any]: 分析結果
        """
        start_time = time.time()
        
        try:
            # 設置原始意圖
            if user_query:
                self.behavior_extractor.set_original_intent(user_query)
            
            # 計算安全評分
            safety_scores = self.behavior_extractor.calculate_safety_score(tool_calls)
            
            result = {
                "safety_score": safety_scores["overall_safety_score"],
                "risk_level": safety_scores["risk_level"],
                "anomaly_score": safety_scores["anomaly_score"],
                "intent_deviation": safety_scores["intent_deviation"],
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"行為特徵分析出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            result = {
                "safety_score": 0.0,
                "risk_level": "unknown",
                "error": str(e),
                "success": False
            }
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        if self.config.log_timings:
            self.timings['behavior_analysis'].append(elapsed_ms)
            result["elapsed_ms"] = elapsed_ms
            
        return result
    
    def run_defense(
        self, 
        user_query: str,
        tool_calls: List[Dict],
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """運行完整的防禦流程
        
        Args:
            user_query: 用戶查詢
            tool_calls: 工具調用列表
            loss_fn: 損失函數
            
        Returns:
            Dict[str, Any]: 防禦結果
        """
        overall_start_time = time.time()
        
        # 檢查資源並調整參數
        resources = self._check_resources()
        self._adjust_defense_parameters(resources)
        
        result = {
            "is_malicious": False,
            "risk_level": "unknown",
            "defense_path": [],
            "details": {}
        }
        
        # 階段1: 梯度特徵分析
        if self.config.enable_grad_features:
            gradient_result = self.analyze_gradient_features(user_query, loss_fn)
            result["details"]["gradient_analysis"] = gradient_result
            result["defense_path"].append("gradient")
            
            if gradient_result["success"]:
                result["risk_level"] = gradient_result["risk_level"]
                
                # 根據梯度分析決定是否提前退出
                if self._should_exit_early(gradient_result["risk_level"], "gradient"):
                    result["early_exit"] = True
                    result["exit_stage"] = "gradient"
                    self.early_exits += 1
                    
                    # 如果風險等級為高或關鍵，標記為惡意
                    result["is_malicious"] = gradient_result["risk_level"] in ["high", "critical"]
                    
                    overall_end_time = time.time()
                    overall_elapsed_ms = (overall_end_time - overall_start_time) * 1000
                    
                    if self.config.log_timings:
                        self.timings['total_defense'].append(overall_elapsed_ms)
                        result["total_elapsed_ms"] = overall_elapsed_ms
                        
                    return result
        
        # 階段2: 行為特徵分析
        if self.config.enable_behavior_features and tool_calls:
            behavior_result = self.analyze_behavior_features(tool_calls, user_query)
            result["details"]["behavior_analysis"] = behavior_result
            result["defense_path"].append("behavior")
            
            if behavior_result["success"]:
                # 更新風險等級為較高的級別
                behavior_risk = behavior_result["risk_level"]
                current_risk = result["risk_level"]
                
                risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3, "unknown": -1}
                if risk_order.get(behavior_risk, -1) > risk_order.get(current_risk, -1):
                    result["risk_level"] = behavior_risk
                
                # 根據行為分析決定是否提前退出
                if self._should_exit_early(behavior_result["risk_level"], "behavior"):
                    result["early_exit"] = True
                    result["exit_stage"] = "behavior"
                    self.early_exits += 1
                    
                    # 如果風險等級為高或關鍵，標記為惡意
                    result["is_malicious"] = result["risk_level"] in ["high", "critical"]
                    
                    overall_end_time = time.time()
                    overall_elapsed_ms = (overall_end_time - overall_start_time) * 1000
                    
                    if self.config.log_timings:
                        self.timings['total_defense'].append(overall_elapsed_ms)
                        result["total_elapsed_ms"] = overall_elapsed_ms
                        
                    return result
        
        # 如果沒有提前退出，執行完整防禦邏輯
        self.full_defenses += 1
        
        # 結合所有指標，判斷是否為惡意
        # 這裡可以實現更複雜的決策邏輯
        result["is_malicious"] = result["risk_level"] in ["high", "critical"]
        
        # 記錄總時間
        overall_end_time = time.time()
        overall_elapsed_ms = (overall_end_time - overall_start_time) * 1000
        
        if self.config.log_timings:
            self.timings['total_defense'].append(overall_elapsed_ms)
            result["total_elapsed_ms"] = overall_elapsed_ms
        
        # 更新處理計數
        self.processed_calls += 1
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """獲取性能指標
        
        Returns:
            Dict[str, Any]: 性能指標
        """
        metrics = {
            "processed_calls": self.processed_calls,
            "early_exits": self.early_exits,
            "full_defenses": self.full_defenses,
            "early_exit_ratio": self.early_exits / max(1, self.processed_calls),
            "current_batch_size": self.config.batch_size,
            "current_sample_times": self.config.sample_times
        }
        
        if self.timings['gradient_analysis']:
            metrics["avg_gradient_time_ms"] = sum(self.timings['gradient_analysis']) / len(self.timings['gradient_analysis'])
            
        if self.timings['behavior_analysis']:
            metrics["avg_behavior_time_ms"] = sum(self.timings['behavior_analysis']) / len(self.timings['behavior_analysis'])
            
        if self.timings['total_defense']:
            metrics["avg_total_time_ms"] = sum(self.timings['total_defense']) / len(self.timings['total_defense'])
            
        # 計算資源使用情況
        resources = self._check_resources()
        metrics.update({
            "current_cpu_percent": resources.get('cpu_percent', 0.0),
            "current_memory_percent": resources.get('memory_percent', 0.0),
            "current_gpu_percent": resources.get('gpu_percent', 0.0)
        })
        
        # 計算計算效率
        if hasattr(self.gradient_enhancer, 'get_compute_efficiency'):
            compute_efficiency = self.gradient_enhancer.get_compute_efficiency()
            metrics["gradient_compute_savings_percent"] = compute_efficiency.get("compute_savings_percent", 0.0)
        
        return metrics
    
    def reset_metrics(self) -> None:
        """重置性能指標"""
        self.timings = {
            'gradient_analysis': [],
            'behavior_analysis': [],
            'total_defense': []
        }
        self.early_exits = 0
        self.full_defenses = 0
        self.processed_calls = 0