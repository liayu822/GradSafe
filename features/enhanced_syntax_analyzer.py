import re
from typing import Dict, Any, List, Tuple

class EnhancedSyntaxAnalyzer:
    def __init__(self):
        # 初始化分析器
        pass

    def analyze(self, text: str) -> Dict[str, Any]:
        # 分析文本，返回分析結果
        result = {
            "sensitive_verbs_count": 0,
            "imperative_mood": False,
            "complex_sequence": False,
            "syntax_anomaly_score": 0.0,
            # 其他指標...
        }
        
        # 檢查複雜序列模式
        sequence_markers = re.findall(r'\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|next|then|after|before|finally)\b', text, re.I)
        result["complex_sequence"] = len(sequence_markers) >= 2

        # 檢查命令式語氣
        imperative_starters = ["do", "execute", "run", "delete", "remove"]
        result["imperative_mood"] = any(text.lower().strip().startswith(word) for word in imperative_starters) or bool(re.search(r'^[A-Z][a-z]+\s', text))

        # 檢查條件語句和循環
        result["has_conditionals"] = bool(re.search(r'\b(if|else|unless|when|case|switch)\b', text, re.I))
        result["has_loops"] = bool(re.search(r'\b(for|while|foreach|loop|iterate|until|repeat)\b', text, re.I))

        return result

    def _calculate_anomaly_score(self, result: Dict[str, Any]) -> None:
        # 計算語法異常得分
        score = 0.0
        
        # 敏感詞得分
        sensitive_word_score = min(1.0, (result["sensitive_verbs_count"] * 0.15) / 5)
        score += sensitive_word_score * 0.3
        
        # 模式匹配得分
        pattern_score = min(1.0, (result["complex_sequence"] * 0.3) / 3)
        score += pattern_score * 0.4
        
        # 結構特徵得分
        structure_score = min(1.0, ((1 if result["imperative_mood"] else 0) * 0.1 + 
                                   (1 if result["has_conditionals"] else 0) * 0.1 + 
                                   (1 if result["has_loops"] else 0) * 0.1) / 0.75)
        score += structure_score * 0.3
        
        result["syntax_anomaly_score"] = score

    def analyze_contextual_coherence(self, sentences: List[str]) -> float:
        # 分析句子間的語法連貫性
        coherence_score = 0.0
        # 假設我們有一個方法來計算句子間的連貫性
        for i in range(len(sentences) - 1):
            coherence_score += self._calculate_sentence_coherence(sentences[i], sentences[i + 1])
        return coherence_score / max(1, len(sentences) - 1)

    def _calculate_sentence_coherence(self, sentence1: str, sentence2: str) -> float:
        # 假設這是一個計算兩個句子間連貫性的輔助方法
        return 1.0  # 這裡應該是具體的計算邏輯

    def analyze_semantic_syntax_consistency(self, text: str) -> float:
        # 比較語法結構與語義內容之間的一致性
        # 假設我們有一個方法來計算一致性
        return 1.0  # 這裡應該是具體的計算邏輯

    def analyze_query_and_tools(self, query_text: str, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 分析工具調用序列
        tool_analysis = self.analyze_tool_sequence(tool_calls, query_text)
        result = {
            "query_text": query_text,
            "tool_analysis": tool_analysis,
            "combined_risk_score": tool_analysis["risk_score"],
            "is_potentially_harmful": tool_analysis["is_suspicious"]
        }
        # 檢查查詢和工具調用的一致性
        if query_text:
            keywords = self._extract_keywords(query_text)
            query_tool_alignment = self._check_query_tool_alignment(keywords, tool_calls)
            result["query_tool_alignment"] = query_tool_alignment
            if query_tool_alignment < 0.3 and tool_analysis["risk_score"] > 0.3:
                result["combined_risk_score"] += 0.2
                result["is_potentially_harmful"] = True
        return result
