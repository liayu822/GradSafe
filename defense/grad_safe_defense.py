# GradSafe防禦實現
import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass

from .multi_stage_orchestrator import MultiStageDefenseOrchestrator, DefenseConfig
from ..features.gradient_features import GradientFeatureEnhancer
from ..features.behavior_features import BehaviorFeatureExtractor
from ..features.syntax_features import SyntaxFeatureExtractor

class GradSafeDefense:
    """
    GradSafe 防禦系統 - 統一介面
    
    集成了梯度特徵增強層和行為一致性分析層的輕量級防禦框架，
    專注於保護LLM代理免受惡意工具調用攻擊。
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_interface: Optional[Any] = None
    ):
        """初始化 GradSafe 防禦系統
        
        Args:
            config: 配置參數
            model_interface: 與LLM模型交互的介面對象
        """
        # 配置初始化
        self.defense_config = DefenseConfig()
        if config:
            # 更新配置
            for key, value in config.items():
                if hasattr(self.defense_config, key):
                    setattr(self.defense_config, key, value)
        
        # 設置日誌記錄器
        self.logger = logging.getLogger(__name__)
        
        # 保存模型介面引用
        self.model_interface = model_interface
        
        # 初始化提取器
        self.syntax_extractor = SyntaxFeatureExtractor(
            max_features=self.defense_config.num_syntax_features,
            device=self.defense_config.device
        )
        
        self.gradient_enhancer = GradientFeatureEnhancer(
            feature_dim=self.defense_config.feature_dim,
            num_syntax_features=self.defense_config.num_syntax_features,
            batch_size=self.defense_config.batch_size,
            mu=self.defense_config.mu,
            sample_times=self.defense_config.sample_times,
            device=self.defense_config.device
        )
        
        self.behavior_extractor = BehaviorFeatureExtractor(
            max_features=20,
            device=self.defense_config.device
        )
        
        # 初始化防禦協調器
        self.orchestrator = MultiStageDefenseOrchestrator(
            config=self.defense_config,
            gradient_enhancer=self.gradient_enhancer,
            behavior_extractor=self.behavior_extractor,
            syntax_extractor=self.syntax_extractor
        )
        
        self.logger.info("GradSafe 防禦系統初始化完成")
    
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """分析用戶查詢
        
        Args:
            user_query: 用戶查詢文本
            
        Returns:
            Dict[str, Any]: 分析結果
        """
        # 設置原始意圖
        self.behavior_extractor.set_original_intent(user_query)
        
        # 提取語法特徵
        syntax_features = self.syntax_extractor.extract_features_tensor(user_query)
        
        # 創建臨時損失函數
        def dummy_loss_fn(features):
            # 簡單返回特徵L2範數作為損失
            return torch.norm(features, dim=1)
        
        # 只執行梯度特徵分析
        gradient_result = self.orchestrator.analyze_gradient_features(
            user_query=user_query,
            loss_fn=dummy_loss_fn
        )
        
        return {
            "query_risk_level": gradient_result.get("risk_level", "unknown"),
            "query_risk_score": gradient_result.get("risk_score", 0.0),
            "is_potentially_harmful": gradient_result.get("risk_level") in ["high", "critical"],
            "details": gradient_result
        }
    
    def analyze_tool_calls(
        self, 
        tool_calls: List[Dict],
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """分析工具調用
        
        Args:
            tool_calls: 工具調用列表
            user_query: 原始用戶查詢
            
        Returns:
            Dict[str, Any]: 分析結果
        """
        # 設置原始意圖（如果提供）
        if user_query:
            self.behavior_extractor.set_original_intent(user_query)
        
        # 執行行為特徵分析
        behavior_result = self.orchestrator.analyze_behavior_features(
            tool_calls=tool_calls,
            user_query=user_query
        )
        
        return {
            "tool_risk_level": behavior_result.get("risk_level", "unknown"),
            "safety_score": behavior_result.get("safety_score", 0.0),
            "is_potentially_harmful": behavior_result.get("risk_level") in ["high", "critical"],
            "details": behavior_result
        }
    
    def defend(
        self,
        user_query: str,
        tool_calls: List[Dict],
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """執行完整防禦流程
        
        Args:
            user_query: 用戶查詢
            tool_calls: 工具調用列表
            callback: 可選的回調函數，用於進行自定義處理
            
        Returns:
            Dict[str, Any]: 防禦結果
        """
        # 創建臨時損失函數
        def dummy_loss_fn(features):
            return torch.norm(features, dim=1)
        
        # 執行多階段防禦
        result = self.orchestrator.run_defense(
            user_query=user_query,
            tool_calls=tool_calls,
            loss_fn=dummy_loss_fn
        )
        
        # 執行回調（如果存在）
        if callback and callable(callback):
            callback(result)
        
        return result
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """更新防禦配置
        
        Args:
            config_updates: 需要更新的配置項
        """
        for key, value in config_updates.items():
            if hasattr(self.defense_config, key):
                setattr(self.defense_config, key, value)
                self.logger.info(f"更新配置: {key} = {value}")
            else:
                self.logger.warning(f"未知配置項: {key}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """獲取性能指標
        
        Returns:
            Dict[str, Any]: 性能指標
        """
        return self.orchestrator.get_performance_metrics()
    
    def reset(self) -> None:
        """重置防禦系統狀態"""
        self.orchestrator.reset_metrics()
        self.behavior_extractor.reset_state()


# 快速創建防禦系統的工廠函數
def create_defense(
    config: Optional[Dict[str, Any]] = None,
    model_interface: Optional[Any] = None
) -> GradSafeDefense:
    """創建GradSafe防禦系統
    
    Args:
        config: 配置參數
        model_interface: 模型介面
        
    Returns:
        GradSafeDefense: 防禦系統實例
    """
    return GradSafeDefense(config=config, model_interface=model_interface)