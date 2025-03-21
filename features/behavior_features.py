#實現工具調用行為特徵提取
import numpy as np
import torch
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
import re
import json
from collections import Counter, defaultdict
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BehaviorFeatureExtractor:
    """
    行為特徵提取器 - 從LLM輸出中提取工具調用行為特徵
    
    主要提取：
    1. 工具調用模式特徵
    2. 參數使用特徵
    3. 調用序列特徵
    4. 異常行為模式
    """
    
    def __init__(
        self,
        max_features: int = 20,
        device: str = "cpu",
        sensitive_keywords: Optional[List[str]] = None,
        high_risk_tools: Optional[List[str]] = None,
        sequence_memory: int = 5,
        intent_embedding_dim: int = 256
    ):
        """初始化行為特徵提取器
        
        Args:
            max_features: 最大特徵數量
            device: 運算裝置
            sensitive_keywords: 敏感關鍵詞列表
            high_risk_tools: 高風險工具列表
            sequence_memory: 序列記憶長度
            intent_embedding_dim: 意圖嵌入維度
        """
        self.max_features = max_features
        self.device = device
        self.sequence_memory = sequence_memory
        self.intent_embedding_dim = intent_embedding_dim
        
        # 設置敏感關鍵詞和高風險工具列表
        self.sensitive_keywords = sensitive_keywords or [
            "password", "token", "key", "secret", "credential", "passwd", 
            "api_key", "apikey", "ssh", "private", "confidential", "admin",
            "密碼", "令牌", "憑證", "密鑰", "秘密", "安全", "帳號", "管理員"
        ]
        
        self.high_risk_tools = high_risk_tools or [
            "run_terminal_cmd", "edit_file", "delete_file", "create_file",
            "run_shell", "execute_code", "http_request", "send_email", 
            "web_search", "sql_query", "api_call"
        ]
        
        # 工具轉移矩陣 (用於計算工具調用序列的概率)
        self.tool_transition_matrix = defaultdict(lambda: defaultdict(float))
        self.tool_frequency = Counter()
        
        # 維護一個歷史工具調用序列
        self.tool_call_history = []
        
        # TF-IDF向量化器用於意圖分析
        self.tfidf = TfidfVectorizer(
            max_features=100, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # 記錄原始意圖的向量表示
        self.original_intent_vector = None
        
        # 特徵名稱到索引的映射
        self._feature_names = []
        self._build_feature_names()
        
        logging.info(f"行為特徵提取器初始化完成，特徵數量: {len(self._feature_names)}")
    
    def _build_feature_names(self) -> None:
        """建立特徵名稱列表"""
        # 工具使用頻率特徵
        tool_features = [
            'file_access_freq',
            'code_execution_freq',
            'web_access_freq',
            'system_cmd_freq',
            'api_call_freq'
        ]
        self._feature_names.extend(tool_features)
        
        # 參數特徵
        param_features = [
            'param_count_avg',
            'param_length_avg',
            'special_param_ratio',
            'numeric_param_ratio',
            'string_param_ratio'
        ]
        self._feature_names.extend(param_features)
        
        # 序列特徵
        sequence_features = [
            'tool_sequence_entropy',
            'repetition_score',
            'tool_diversity',
            'sequence_length',
            'transition_probability'
        ]
        self._feature_names.extend(sequence_features)
        
        # 異常特徵
        anomaly_features = [
            'unusual_pattern_score',
            'permission_escalation_attempt',
            'resource_abuse_score',
            'sensitive_data_access',
            'evasion_technique_score'
        ]
        self._feature_names.extend(anomaly_features)
        
        # 意圖相關特徵
        intent_features = [
            'intent_deviation_score',
            'context_relevance',
            'semantic_coherence',
            'goal_alignment',
            'progression_logic'
        ]
        self._feature_names.extend(intent_features)
        
        # 限制特徵數量
        if len(self._feature_names) > self.max_features:
            self._feature_names = self._feature_names[:self.max_features]
    
    def get_feature_names(self) -> List[str]:
        """獲取特徵名稱列表
        
        Returns:
            List[str]: 特徵名稱列表
        """
        return self._feature_names
    
    def set_original_intent(self, user_query: str) -> None:
        """設置原始使用者意圖
        
        Args:
            user_query: 用戶原始查詢文本
        """
        if not user_query:
            return
        
        # 使用TF-IDF向量化用戶查詢
        try:
            # 必須包含至少一個文檔來訓練向量化器
            self.tfidf.fit([user_query])
            self.original_intent_vector = self.tfidf.transform([user_query]).toarray()[0]
            logging.info(f"已設置原始意圖向量，維度: {len(self.original_intent_vector)}")
        except Exception as e:
            logging.error(f"設置原始意圖時出錯: {str(e)}")
            self.original_intent_vector = np.zeros(self.tfidf.max_features or 100)
    
    def _extract_tool_frequency_features(self, tool_calls: List[Dict]) -> Dict[str, float]:
        """提取工具調用頻率特徵
        
        Args:
            tool_calls: 工具調用列表
            
        Returns:
            Dict[str, float]: 工具頻率特徵
        """
        features = {
            'file_access_freq': 0.0,
            'code_execution_freq': 0.0,
            'web_access_freq': 0.0,
            'system_cmd_freq': 0.0,
            'api_call_freq': 0.0
        }
        
        if not tool_calls:
            return features
        
        total_calls = len(tool_calls)
        
        # 計算各類工具的使用頻率
        file_tools = ['read_file', 'edit_file', 'delete_file', 'create_file', 'file_search', 'list_dir']
        code_tools = ['run_terminal_cmd', 'execute_code', 'python_repl']
        web_tools = ['web_search', 'http_request', 'browse_website', 'curl']
        system_tools = ['run_shell', 'run_terminal_cmd', 'grep_search']
        api_tools = ['api_call', 'search_symbols', 'codebase_search']
        
        for tool_call in tool_calls:
            tool_name = tool_call.get('name', '')
            
            # 更新工具調用歷史和頻率計數
            self.tool_call_history.append(tool_name)
            self.tool_frequency[tool_name] += 1
            
            # 計算工具類別的頻率
            if tool_name in file_tools:
                features['file_access_freq'] += 1
            
            if tool_name in code_tools:
                features['code_execution_freq'] += 1
            
            if tool_name in web_tools:
                features['web_access_freq'] += 1
            
            if tool_name in system_tools:
                features['system_cmd_freq'] += 1
            
            if tool_name in api_tools:
                features['api_call_freq'] += 1
        
        # 保持歷史長度限制
        if len(self.tool_call_history) > self.sequence_memory:
            self.tool_call_history = self.tool_call_history[-self.sequence_memory:]
        
        # 正規化頻率
        for key in features:
            features[key] = features[key] / total_calls if total_calls > 0 else 0.0
        
        return features
    
    def _extract_param_features(self, tool_calls: List[Dict]) -> Dict[str, float]:
        """提取參數使用特徵
        
        Args:
            tool_calls: 工具調用列表
            
        Returns:
            Dict[str, float]: 參數特徵
        """
        features = {
            'param_count_avg': 0.0,
            'param_length_avg': 0.0,
            'special_param_ratio': 0.0,
            'numeric_param_ratio': 0.0,
            'string_param_ratio': 0.0
        }
        
        if not tool_calls:
            return features
        
        total_params = 0
        total_param_length = 0
        special_params = 0
        numeric_params = 0
        string_params = 0
        
        # 分析參數使用情況
        for tool_call in tool_calls:
            params = tool_call.get('parameters', {})
            
            if params:
                param_count = len(params)
                total_params += param_count
                
                for param_name, param_value in params.items():
                    # 分析參數值
                    param_str = str(param_value)
                    total_param_length += len(param_str)
                    
                    # 檢測特殊參數（包含特殊字符或敏感關鍵詞）
                    if any(keyword in param_str.lower() for keyword in self.sensitive_keywords) or \
                       re.search(r'[;|&`$!*?]', param_str):
                        special_params += 1
                    
                    # 檢測數值參數
                    if isinstance(param_value, (int, float)) or param_str.isdigit():
                        numeric_params += 1
                    
                    # 檢測字符串參數
                    if isinstance(param_value, str) and not param_str.isdigit():
                        string_params += 1
        
        # 計算平均值和比率
        if total_params > 0:
            features['param_count_avg'] = total_params / len(tool_calls)
            features['param_length_avg'] = total_param_length / total_params
            features['special_param_ratio'] = special_params / total_params
            features['numeric_param_ratio'] = numeric_params / total_params
            features['string_param_ratio'] = string_params / total_params
        
        return features
    
    def _extract_sequence_features(self, tool_calls: List[Dict]) -> Dict[str, float]:
        """提取工具調用序列特徵
        
        Args:
            tool_calls: 工具調用列表
            
        Returns:
            Dict[str, float]: 序列特徵
        """
        features = {
            'tool_sequence_entropy': 0.0,
            'repetition_score': 0.0,
            'tool_diversity': 0.0,
            'sequence_length': 0.0,
            'transition_probability': 0.0
        }
        
        # 獲取當前工具序列
        current_sequence = [tool_call.get('name', '') for tool_call in tool_calls]
        
        # 如果沒有足夠的工具調用記錄，返回默認特徵
        if not current_sequence:
            return features
        
        # 計算序列特徵
        # 1. 序列長度
        features['sequence_length'] = len(current_sequence) / self.sequence_memory
        
        # 2. 工具多樣性（使用不同工具的數量除以序列長度）
        unique_tools = len(set(current_sequence))
        features['tool_diversity'] = unique_tools / len(current_sequence) if len(current_sequence) > 0 else 0.0
        
        # 3. 重複分數（連續重複調用同一工具的次數）
        repetitions = 0
        for i in range(1, len(current_sequence)):
            if current_sequence[i] == current_sequence[i-1]:
                repetitions += 1
        features['repetition_score'] = repetitions / (len(current_sequence) - 1) if len(current_sequence) > 1 else 0.0
        
        # 4. 序列熵（衡量序列的不確定性）
        tool_counts = Counter(current_sequence)
        entropy = 0.0
        for tool, count in tool_counts.items():
            p = count / len(current_sequence)
            entropy -= p * math.log2(p) if p > 0 else 0
        # 正規化熵值
        max_entropy = math.log2(len(tool_counts)) if len(tool_counts) > 0 else 1
        features['tool_sequence_entropy'] = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # 5. 轉移概率（從當前工具序列的轉移概率與常見模式的一致性）
        # 先更新轉移矩陣
        for i in range(1, len(current_sequence)):
            prev_tool = current_sequence[i-1]
            curr_tool = current_sequence[i]
            self.tool_transition_matrix[prev_tool][curr_tool] += 1
        
        # 計算當前序列的轉移概率
        if len(current_sequence) > 1:
            transition_prob = 1.0
            for i in range(1, len(current_sequence)):
                prev_tool = current_sequence[i-1]
                curr_tool = current_sequence[i]
                
                # 計算此轉移的概率
                total_transitions = sum(self.tool_transition_matrix[prev_tool].values())
                if total_transitions > 0:
                    prob = self.tool_transition_matrix[prev_tool][curr_tool] / total_transitions
                    transition_prob *= prob if prob > 0 else 0.01  # 避免乘以0
                
            # 正規化轉移概率
            features['transition_probability'] = transition_prob
            
            # 防止概率為0或過小導致的數值問題
            if features['transition_probability'] < 1e-10:
                features['transition_probability'] = 0.0
            else:
                # 應用對數變換以使分布更平衡
                features['transition_probability'] = -math.log10(features['transition_probability']) / len(current_sequence)
                # 正規化到0-1範圍
                features['transition_probability'] = min(features['transition_probability'], 1.0)
        
        return features
    
    def _extract_anomaly_features(self, tool_calls: List[Dict]) -> Dict[str, float]:
        """提取異常行為特徵
        
        Args:
            tool_calls: 工具調用列表
            
        Returns:
            Dict[str, float]: 異常特徵
        """
        features = {
            'unusual_pattern_score': 0.0,
            'permission_escalation_attempt': 0.0,
            'resource_abuse_score': 0.0,
            'sensitive_data_access': 0.0,
            'evasion_technique_score': 0.0
        }
        
        if not tool_calls:
            return features
        
        # 檢測異常模式
        unusual_patterns = 0
        perm_escalation = 0
        resource_abuse = 0
        sensitive_access = 0
        evasion_techniques = 0
        
        # 可疑命令和參數模式
        suspicious_cmd_patterns = [
            r'(sudo|su)\s',
            r'chmod\s.*777',
            r'rm\s+-rf\s+/',
            r'(wget|curl)\s+.*\s+\|\s+(bash|sh)',
            r'eval\s+.*',
            r'exec\s+.*',
            r'(base64|hex)\s+.*'
        ]
        
        # 資源濫用模式
        resource_abuse_patterns = [
            r'while\s+true',
            r'fork\s*\(',
            r':\(\)\s*{\s*:\s*\|\s*:\s*&\s*}\s*;',  # 惡意fork炸彈模式
            r'dd\s+if=/dev/zero',
            r'yes\s+>\s+/dev'
        ]
        
        # 敏感數據訪問模式
        sensitive_data_patterns = [
            r'(\/etc\/passwd|\/etc\/shadow)',
            r'\.ssh\/.*key',
            r'(token|api.?key|secret|password|credential)',
            r'(\/var\/log\/|\/proc\/|\/sys\/)'
        ]
        
        # 規避檢測技術
        evasion_patterns = [
            r'(base64|hex)\s+(decode|encode)',
            r'(sleep|timeout|wait)\s+[0-9]+',
            r'(\$\(|\`|\|)',
            r'(\/dev\/null|>/dev/null|2>&1)',
            r'\$\{IFS\}'
        ]
        
        for tool_call in tool_calls:
            tool_name = tool_call.get('name', '')
            params = tool_call.get('parameters', {})
            
            # 轉換參數為字符串以便檢測模式
            param_str = json.dumps(params)
            
            # 檢測高風險工具使用
            if tool_name in self.high_risk_tools:
                unusual_patterns += 1
            
            # 檢測可疑命令模式
            for pattern in suspicious_cmd_patterns:
                if re.search(pattern, param_str, re.IGNORECASE):
                    perm_escalation += 1
                    break
            
            # 檢測資源濫用
            for pattern in resource_abuse_patterns:
                if re.search(pattern, param_str, re.IGNORECASE):
                    resource_abuse += 1
                    break
                    
            # 檢測敏感數據訪問
            for pattern in sensitive_data_patterns:
                if re.search(pattern, param_str, re.IGNORECASE):
                    sensitive_access += 1
                    break
            
            # 檢測規避技術
            for pattern in evasion_patterns:
                if re.search(pattern, param_str, re.IGNORECASE):
                    evasion_techniques += 1
                    break
        
        # 正規化特徵值
        total_calls = len(tool_calls)
        features['unusual_pattern_score'] = unusual_patterns / total_calls if total_calls > 0 else 0.0
        features['permission_escalation_attempt'] = perm_escalation / total_calls if total_calls > 0 else 0.0
        features['resource_abuse_score'] = resource_abuse / total_calls if total_calls > 0 else 0.0
        features['sensitive_data_access'] = sensitive_access / total_calls if total_calls > 0 else 0.0
        features['evasion_technique_score'] = evasion_techniques / total_calls if total_calls > 0 else 0.0
        
        return features
    
    def _extract_intent_features(self, tool_calls: List[Dict]) -> Dict[str, float]:
        """提取意圖相關特徵
        
        Args:
            tool_calls: 工具調用列表
            
        Returns:
            Dict[str, float]: 意圖特徵
        """
        features = {
            'intent_deviation_score': 0.0,
            'context_relevance': 1.0,  # 默認為最高相關性
            'semantic_coherence': 1.0,  # 默認為最高一致性
            'goal_alignment': 1.0,      # 默認為最高對齊度
            'progression_logic': 1.0    # 默認為最高邏輯性
        }
        
        # 如果沒有原始意圖或工具調用，返回默認特徵
        if self.original_intent_vector is None or not tool_calls:
            return features
        
        # 將工具調用轉換為文本
        tool_text = ""
        for tool_call in tool_calls:
            tool_name = tool_call.get('name', '')
            params = tool_call.get('parameters', {})
            tool_text += f"{tool_name} {json.dumps(params)} "
        
        try:
            # 使用TF-IDF向量化工具調用
            tool_vector = self.tfidf.transform([tool_text]).toarray()[0]
            
            # 計算工具調用與原始意圖的餘弦相似度
            similarity = cosine_similarity(
                [self.original_intent_vector], 
                [tool_vector]
            )[0][0]
            
            # 計算意圖偏離分數
            features['intent_deviation_score'] = 1.0 - max(0.0, min(1.0, similarity))
            
            # 其他意圖相關特徵
            features['context_relevance'] = max(0.1, similarity)
            features['semantic_coherence'] = max(0.1, similarity)
            
            # 計算目標對齊度（根據工具類型與意圖的匹配度）
            # 這需要更複雜的實現，這裡只給一個簡化版本
            intent_tools_match = 0.7  # 假設有70%的工具與意圖相匹配
            features['goal_alignment'] = intent_tools_match
            
            # 計算進展邏輯性（工具調用順序是否符合常見任務的邏輯流程）
            # 這需要更複雜的實現，這裡只給一個簡化版本
            progression_logic = 0.8  # 假設工具調用順序80%符合邏輯
            features['progression_logic'] = progression_logic
        
        except Exception as e:
            logging.error(f"提取意圖特徵時出錯: {str(e)}")
        
        return features
    
    def extract_features(self, tool_calls: List[Dict]) -> np.ndarray:
        """從工具調用列表中提取特徵
        
        Args:
            tool_calls: 工具調用列表
            
        Returns:
            np.ndarray: 特徵向量
        """
        # 提取各類特徵
        tool_freq_features = self._extract_tool_frequency_features(tool_calls)
        param_features = self._extract_param_features(tool_calls)
        sequence_features = self._extract_sequence_features(tool_calls)
        anomaly_features = self._extract_anomaly_features(tool_calls)
        intent_features = self._extract_intent_features(tool_calls)
        
        # 合併特徵
        all_features = {}
        all_features.update(tool_freq_features)
        all_features.update(param_features)
        all_features.update(sequence_features)
        all_features.update(anomaly_features)
        all_features.update(intent_features)
        
        # 轉換為向量
        feature_vector = np.zeros(len(self._feature_names))
        for i, feature_name in enumerate(self._feature_names):
            feature_vector[i] = all_features.get(feature_name, 0.0)
        
        return feature_vector
    
    def extract_features_tensor(self, tool_calls: List[Dict]) -> torch.Tensor:
        """從工具調用列表中提取特徵並返回PyTorch張量
        
        Args:
            tool_calls: 工具調用列表
            
        Returns:
            torch.Tensor: 特徵張量
        """
        features = self.extract_features(tool_calls)
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def batch_extract_features(self, batch_tool_calls: List[List[Dict]]) -> np.ndarray:
        """批量提取特徵
        
        Args:
            batch_tool_calls: 批量工具調用列表
            
        Returns:
            np.ndarray: 特徵向量批次
        """
        features = []
        for tool_calls in batch_tool_calls:
            feature_vector = self.extract_features(tool_calls)
            features.append(feature_vector)
        
        return np.array(features)
    
    def batch_extract_features_tensor(self, batch_tool_calls: List[List[Dict]]) -> torch.Tensor:
        """批量提取特徵並返回PyTorch張量
        
        Args:
            batch_tool_calls: 批量工具調用列表
            
        Returns:
            torch.Tensor: 特徵張量批次
        """
        features = self.batch_extract_features(batch_tool_calls)
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def calculate_safety_score(self, tool_calls: List[Dict]) -> Dict[str, float]:
        """計算工具調用的安全評分
        
        Args:
            tool_calls: 工具調用列表
            
        Returns:
            Dict[str, float]: 安全評分指標
        """
        # 提取特徵
        features = self.extract_features(tool_calls)
        
        # 計算風險相關指標
        anomaly_indices = [self._feature_names.index(f) for f in self._feature_names 
                          if f in ['unusual_pattern_score', 'permission_escalation_attempt', 
                                'resource_abuse_score', 'sensitive_data_access', 
                                'evasion_technique_score']]
        
        intent_indices = [self._feature_names.index(f) for f in self._feature_names
                         if f in ['intent_deviation_score']]
        
        # 從特徵向量獲取指標值
        anomaly_scores = [features[i] for i in anomaly_indices if i < len(features)]
        intent_scores = [features[i] for i in intent_indices if i < len(features)]
        
        # 計算綜合安全分數
        anomaly_score = sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0
        intent_deviation = sum(intent_scores) / len(intent_scores) if intent_scores else 0.0
        
        # 計算最終安全分數（越低越安全）
        safety_score = 0.7 * anomaly_score + 0.3 * intent_deviation
        
        # 返回詳細評分
        return {
            "overall_safety_score": 1.0 - safety_score,  # 轉換為正向分數（越高越安全）
            "anomaly_score": anomaly_score,
            "intent_deviation": intent_deviation,
            "risk_level": self._classify_risk_level(safety_score)
        }
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """根據風險分數分類風險等級
        
        Args:
            risk_score: 風險分數
            
        Returns:
            str: 風險等級
        """
        if risk_score < 0.2:
            return "low"
        elif risk_score < 0.5:
            return "medium"
        elif risk_score < 0.8:
            return "high"
        else:
            return "critical"
    
    def reset_state(self) -> None:
        """重置提取器的內部狀態"""
        self.tool_call_history = []
        self.tool_frequency = Counter()
        # 保留轉移矩陣，因為它包含有價值的長期模式