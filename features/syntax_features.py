# 語法特徵提取
import re
import numpy as np
import torch
from typing import List, Dict, Optional, Union, Tuple
import logging
import spacy
from spacy.tokens import Doc
from collections import Counter
import nltk
from nltk.tree import Tree
from nltk.parse import CoreNLPParser
from nltk.tokenize import word_tokenize

class SyntaxFeatureExtractor:
    """
    語法特徵提取器 - 從輸入文本中提取語法特徵
    
    主要提取：
    1. 詞性分布特徵
    2. 語法依賴結構特徵
    3. 語法樹複雜度指標
    4. 特殊語法模式識別
    5. 歧義性指標
    """
    
    def __init__(
        self,
        use_spacy: bool = True, 
        use_corenlp: bool = False,
        spacy_model: str = "zh_core_web_sm",  # 使用中文模型，也可使用其他語言模型
        max_features: int = 50,
        device: str = "cpu",  # spaCy 主要在 CPU 上運行
        corenlp_url: Optional[str] = None
    ):
        """初始化語法特徵提取器
        
        Args:
            use_spacy: 是否使用 spaCy 進行語法分析
            use_corenlp: 是否使用 Stanford CoreNLP 進行語法分析
            spacy_model: 要使用的 spaCy 模型
            max_features: 最大特徵數量
            device: 運算裝置
            corenlp_url: CoreNLP 服務器 URL，只在 use_corenlp=True 時使用
        """
        self.use_spacy = use_spacy
        self.use_corenlp = use_corenlp
        self.max_features = max_features
        self.device = device
        self.corenlp_url = corenlp_url
        
        # 初始化 spaCy
        if self.use_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
                logging.info(f"已加載 spaCy 模型: {spacy_model}")
            except OSError:
                logging.warning(f"無法加載 spaCy 模型: {spacy_model}，嘗試下載...")
                spacy.cli.download(spacy_model)
                self.nlp = spacy.load(spacy_model)
                logging.info(f"已下載並加載 spaCy 模型: {spacy_model}")
        
        # 初始化 CoreNLP
        if self.use_corenlp:
            try:
                self.parser = CoreNLPParser(url=self.corenlp_url)
                logging.info(f"已連接 CoreNLP 服務器: {self.corenlp_url}")
            except Exception as e:
                logging.error(f"無法連接 CoreNLP 服務器: {str(e)}")
                self.use_corenlp = False
        
        # 確保 NLTK 資源可用
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logging.info("下載 NLTK punkt 資源...")
            nltk.download('punkt')
        
        # 特徵名稱到索引的映射
        self._feature_names = []
        self._build_feature_names()
        
        logging.info(f"語法特徵提取器初始化完成，特徵數量: {len(self._feature_names)}")
    
    def _build_feature_names(self) -> None:
        """建立特徵名稱列表"""
        # 詞性標籤分布
        pos_tags = [
            'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'PART', 'INTJ', 
            'CONJ', 'PUNCT', 'NUM', 'ADP', 'SYM', 'X'
        ]
        self._feature_names.extend([f"pos_ratio_{tag}" for tag in pos_tags])
        
        # 依存關係分布
        dep_types = [
            'nsubj', 'obj', 'iobj', 'amod', 'advmod', 'conj', 'cc', 'aux',
            'root', 'compound', 'prep', 'det', 'mark', 'case'
        ]
        self._feature_names.extend([f"dep_ratio_{dep}" for dep in dep_types])
        
        # 句法複雜度指標
        complexity_metrics = [
            'avg_sentence_length', 'avg_tree_depth', 'avg_noun_chunks',
            'noun_chunk_ratio', 'verb_ratio', 'function_word_ratio',
            'content_word_ratio', 'special_char_ratio', 'nested_clause_count'
        ]
        self._feature_names.extend(complexity_metrics)
        
        # 特殊語法模式
        special_patterns = [
            'imp_command_pattern', 'question_pattern', 'negation_pattern', 
            'conditional_pattern', 'ambiguity_score'
        ]
        self._feature_names.extend(special_patterns)
        
        # 限制特徵數量
        if len(self._feature_names) > self.max_features:
            self._feature_names = self._feature_names[:self.max_features]
    
    def get_feature_names(self) -> List[str]:
        """獲取特徵名稱列表
        
        Returns:
            List[str]: 特徵名稱列表
        """
        return self._feature_names
    
    def _extract_pos_features(self, doc: Doc) -> Dict[str, float]:
        """提取詞性特徵
        
        Args:
            doc: spaCy 文檔對象
            
        Returns:
            Dict[str, float]: 詞性特徵
        """
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)
        
        features = {}
        for pos in set([token.pos_ for token in doc]):
            feature_name = f"pos_ratio_{pos}"
            if feature_name in self._feature_names:
                features[feature_name] = pos_counts[pos] / total_tokens if total_tokens > 0 else 0
        
        return features
    
    def _extract_dependency_features(self, doc: Doc) -> Dict[str, float]:
        """提取依存關係特徵
        
        Args:
            doc: spaCy 文檔對象
            
        Returns:
            Dict[str, float]: 依存關係特徵
        """
        dep_counts = Counter([token.dep_ for token in doc])
        total_tokens = len(doc)
        
        features = {}
        for dep in set([token.dep_ for token in doc]):
            feature_name = f"dep_ratio_{dep}"
            if feature_name in self._feature_names:
                features[feature_name] = dep_counts[dep] / total_tokens if total_tokens > 0 else 0
        
        return features
    
    def _extract_complexity_features(self, doc: Doc) -> Dict[str, float]:
        """提取複雜度特徵
        
        Args:
            doc: spaCy 文檔對象
            
        Returns:
            Dict[str, float]: 複雜度特徵
        """
        features = {}
        
        # 平均句子長度
        sentences = list(doc.sents)
        if len(sentences) > 0:
            avg_sent_len = sum(len(sent) for sent in sentences) / len(sentences)
            features['avg_sentence_length'] = avg_sent_len
        else:
            features['avg_sentence_length'] = 0
        
        # 名詞短語統計
        noun_chunks = list(doc.noun_chunks)
        features['avg_noun_chunks'] = len(noun_chunks) / len(sentences) if len(sentences) > 0 else 0
        features['noun_chunk_ratio'] = len(noun_chunks) / len(doc) if len(doc) > 0 else 0
        
        # 詞性比例
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)
        features['verb_ratio'] = pos_counts['VERB'] / total_tokens if total_tokens > 0 else 0
        
        # 功能詞與內容詞比例
        function_pos = {'ADP', 'AUX', 'CONJ', 'DET', 'PART', 'PRON', 'SCONJ'}
        content_pos = {'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB'}
        
        function_words = sum(1 for token in doc if token.pos_ in function_pos)
        content_words = sum(1 for token in doc if token.pos_ in content_pos)
        
        features['function_word_ratio'] = function_words / total_tokens if total_tokens > 0 else 0
        features['content_word_ratio'] = content_words / total_tokens if total_tokens > 0 else 0
        
        # 特殊字符比例
        special_chars = sum(1 for token in doc if not token.is_alpha)
        features['special_char_ratio'] = special_chars / total_tokens if total_tokens > 0 else 0
        
        # 嵌套子句計數（近似估計）
        # 這裡使用依存關係中的'advcl', 'acl', 'ccomp', 'xcomp', 'rcmod'等作為指標
        nested_deps = {'advcl', 'acl', 'ccomp', 'xcomp', 'relcl'}
        nested_clauses = sum(1 for token in doc if token.dep_ in nested_deps)
        features['nested_clause_count'] = nested_clauses
        
        return features
    
    def _extract_pattern_features(self, doc: Doc) -> Dict[str, float]:
        """提取特殊語法模式特徵
        
        Args:
            doc: spaCy 文檔對象
            
        Returns:
            Dict[str, float]: 特殊語法模式特徵
        """
        features = {}
        text = doc.text.lower()
        
        # 命令句模式（通常以動詞開頭）
        first_tokens = [sent[0] for sent in doc.sents if len(sent) > 0]
        imp_command_score = sum(1 for token in first_tokens if token.pos_ == 'VERB') / len(first_tokens) if first_tokens else 0
        features['imp_command_pattern'] = imp_command_score
        
        # 問句模式
        question_markers = {'?', '什麼', '為什麼', '怎麼', '如何', '是否'}
        question_score = any(marker in text for marker in question_markers)
        features['question_pattern'] = float(question_score)
        
        # 否定模式
        negation_markers = {'不', '沒', '無', '非', '莫', '勿'}
        negation_count = sum(1 for token in doc if token.text in negation_markers)
        features['negation_pattern'] = negation_count / len(doc) if len(doc) > 0 else 0
        
        # 條件句模式
        conditional_markers = {'如果', '若', '假如', '要是', '除非', '只要', '倘若'}
        conditional_score = any(marker in text for marker in conditional_markers)
        features['conditional_pattern'] = float(conditional_score)
        
        # 歧義性分數（簡單估計）
        # 這裡使用具有多義詞性標籤的單詞比例作為指標
        pos_per_token = {}
        for token in doc:
            if token.text not in pos_per_token:
                pos_per_token[token.text] = set()
            pos_per_token[token.text].add(token.pos_)
        
        ambiguous_tokens = sum(1 for tokens in pos_per_token.values() if len(tokens) > 1)
        features['ambiguity_score'] = ambiguous_tokens / len(pos_per_token) if len(pos_per_token) > 0 else 0
        
        return features
    
    def _parse_with_corenlp(self, text: str) -> Dict[str, float]:
        """使用 CoreNLP 解析文本（獲取語法樹深度等特徵）
        
        Args:
            text: 輸入文本
            
        Returns:
            Dict[str, float]: 解析特徵
        """
        features = {'avg_tree_depth': 0.0}
        
        if not self.use_corenlp:
            return features
        
        try:
            # 使用 CoreNLP 解析文本
            tree_iter = self.parser.raw_parse(text)
            trees = list(tree_iter)
            
            if trees:
                # 計算語法樹深度
                depths = []
                for tree in trees:
                    depth = self._get_tree_depth(tree)
                    depths.append(depth)
                
                features['avg_tree_depth'] = sum(depths) / len(depths) if depths else 0
        except Exception as e:
            logging.warning(f"CoreNLP 解析錯誤: {str(e)}")
        
        return features
    
    def _get_tree_depth(self, tree: Tree) -> int:
        """獲取語法樹的深度
        
        Args:
            tree: NLTK Tree 對象
            
        Returns:
            int: 樹的深度
        """
        if not isinstance(tree, Tree):
            return 0
        
        if len(tree) == 0:
            return 1
        
        return 1 + max(self._get_tree_depth(subtree) for subtree in tree)
    
    def extract_features(self, text: str) -> np.ndarray:
        """從文本中提取所有語法特徵
        
        Args:
            text: 輸入文本
            
        Returns:
            np.ndarray: 特徵向量
        """
        features_dict = {}
        
        # spaCy 分析
        if self.use_spacy:
            doc = self.nlp(text)
            
            # 提取各類特徵
            pos_features = self._extract_pos_features(doc)
            dep_features = self._extract_dependency_features(doc)
            complexity_features = self._extract_complexity_features(doc)
            pattern_features = self._extract_pattern_features(doc)
            
            # 合併特徵
            features_dict.update(pos_features)
            features_dict.update(dep_features)
            features_dict.update(complexity_features)
            features_dict.update(pattern_features)
        
        # CoreNLP 分析（如果啟用）
        if self.use_corenlp:
            corenlp_features = self._parse_with_corenlp(text)
            features_dict.update(corenlp_features)
        
        # 轉換為向量
        feature_vector = np.zeros(len(self._feature_names))
        for i, feature_name in enumerate(self._feature_names):
            feature_vector[i] = features_dict.get(feature_name, 0.0)
        
        return feature_vector
    
    def batch_extract_features(self, texts: List[str]) -> np.ndarray:
        """批量從文本中提取特徵
        
        Args:
            texts: 輸入文本列表
            
        Returns:
            np.ndarray: 特徵向量批次，形狀為 [batch_size, feature_dim]
        """
        features = []
        for text in texts:
            feature_vector = self.extract_features(text)
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_features_tensor(self, text: str) -> torch.Tensor:
        """從文本中提取特徵並返回 PyTorch 張量
        
        Args:
            text: 輸入文本
            
        Returns:
            torch.Tensor: 特徵張量
        """
        features = self.extract_features(text)
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def batch_extract_features_tensor(self, texts: List[str]) -> torch.Tensor:
        """批量從文本中提取特徵並返回 PyTorch 張量
        
        Args:
            texts: 輸入文本列表
            
        Returns:
            torch.Tensor: 特徵張量批次，形狀為 [batch_size, feature_dim]
        """
        features = self.batch_extract_features(texts)
        return torch.tensor(features, dtype=torch.float32, device=self.device)