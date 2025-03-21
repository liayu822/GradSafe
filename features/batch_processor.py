#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量處理優化模組 - 適用於GradSafe框架，提供批處理優化和提前退出機制，
以減少LLM代理處理時的延遲，並支持資源自適應調整。
"""

import os
import time
import logging
import threading
import queue
from typing import List, Dict, Any, Optional, Callable, Tuple, Union, TypeVar
import numpy as np
import torch
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 類型變量定義
T = TypeVar('T')  # 輸入類型
R = TypeVar('R')  # 結果類型

@dataclass
class BatchConfig:
    """批處理配置類"""
    
    # 批量大小設置
    max_batch_size: int = 8
    min_batch_size: int = 1
    adaptive_batch_size: bool = True
    
    # 時間相關設置
    max_wait_time: float = 0.1  # 最大等待時間（秒）
    timeout: float = 2.0  # 處理超時時間（秒）
    
    # 提前退出設置
    early_exit_enabled: bool = True
    confidence_threshold: float = 0.95  # 提前退出的置信度閾值
    min_samples_before_exit: int = 3  # 在考慮提前退出前的最小樣本數
    
    # 資源自適應設置
    adapt_to_resources: bool = True
    gpu_memory_threshold: float = 0.8  # GPU記憶體使用量閾值
    cpu_usage_threshold: float = 0.9  # CPU使用量閾值
    
    # 快取設置
    enable_caching: bool = True
    cache_size: int = 1000
    
    # 性能監控設置
    collect_stats: bool = True


class ResourceMonitor:
    """
    資源監控器 - 監控系統資源使用情況，提供自適應調整策略
    """
    
    def __init__(self, config: BatchConfig):
        """初始化資源監視器"""
        self.config = config
        self.cpu_usages = []
        self.gpu_usages = []
        self.memory_usages = []
        self._last_check_time = 0
        self._check_interval = 0.5  # 檢查間隔（秒）
    
    def should_check(self) -> bool:
        """檢查是否應該執行資源檢查"""
        current_time = time.time()
        if current_time - self._last_check_time > self._check_interval:
            self._last_check_time = current_time
            return True
        return False
    
    def check_resources(self) -> Dict[str, float]:
        """檢查當前資源使用情況"""
        if not self.should_check():
            # 使用最後一次檢查的結果
            return {
                "cpu_usage": self.cpu_usages[-1] if self.cpu_usages else 0.0,
                "gpu_usage": self.gpu_usages[-1] if self.gpu_usages else 0.0,
                "memory_usage": self.memory_usages[-1] if self.memory_usages else 0.0,
            }
        
        # 獲取CPU使用率
        try:
            import psutil
            cpu_usage = psutil.cpu_percent() / 100
            self.cpu_usages.append(cpu_usage)
        except ImportError:
            logger.warning("psutil未安裝，無法監控CPU使用率")
            cpu_usage = 0.0
        
        # 獲取GPU使用率和記憶體使用率
        gpu_usage = 0.0
        gpu_memory_usage = 0.0
        
        if torch.cuda.is_available():
            try:
                # 獲取活躍的GPU
                device_idx = torch.cuda.current_device()
                
                # 使用nvidia-smi獲取更精確的使用率
                try:
                    gpu_usage = torch.cuda.utilization(device_idx) / 100
                except:
                    gpu_usage = 0.0
                
                # 獲取GPU記憶體使用率
                gpu_memory_allocated = torch.cuda.memory_allocated(device_idx)
                gpu_memory_reserved = torch.cuda.memory_reserved(device_idx)
                gpu_memory_total = torch.cuda.get_device_properties(device_idx).total_memory
                
                gpu_memory_usage = (gpu_memory_allocated + gpu_memory_reserved) / gpu_memory_total
                
                self.gpu_usages.append(gpu_usage)
                self.memory_usages.append(gpu_memory_usage)
            except Exception as e:
                logger.warning(f"監控GPU資源時出錯: {e}")
        
        result = {
            "cpu_usage": cpu_usage,
            "gpu_usage": gpu_usage,
            "memory_usage": gpu_memory_usage,
        }
        
        return result
    
    def get_recommended_batch_size(self, current_batch_size: int) -> int:
        """
        根據當前資源使用情況，推薦適當的批量大小
        
        Args:
            current_batch_size: 當前的批量大小
        
        Returns:
            推薦的新批量大小
        """
        if not self.config.adapt_to_resources:
            return current_batch_size
        
        resources = self.check_resources()
        
        # 如果GPU記憶體使用率超過閾值，減小批量大小
        if resources["memory_usage"] > self.config.gpu_memory_threshold:
            return max(self.config.min_batch_size, current_batch_size - 1)
        
        # 如果CPU使用率超過閾值，減小批量大小
        if resources["cpu_usage"] > self.config.cpu_usage_threshold:
            return max(self.config.min_batch_size, current_batch_size - 1)
        
        # 如果資源充足，增加批量大小
        if resources["memory_usage"] < 0.5 and resources["cpu_usage"] < 0.5:
            return min(self.config.max_batch_size, current_batch_size + 1)
        
        # 否則保持不變
        return current_batch_size


class ResultCache:
    """
    結果快取 - 用於存儲和快速檢索處理結果，避免重複計算
    """
    
    def __init__(self, capacity: int = 1000):
        """
        初始化結果快取
        
        Args:
            capacity: 快取容量，預設為1000項
        """
        self.capacity = capacity
        self.cache = {}
        self.usage_count = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        從快取中獲取結果
        
        Args:
            key: 快取鍵值
        
        Returns:
            快取的結果，如果不存在則返回None
        """
        with self.lock:
            if key in self.cache:
                self.usage_count[key] += 1
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """
        將結果存入快取
        
        Args:
            key: 快取鍵值
            value: 要快取的結果
        """
        with self.lock:
            # 如果快取已滿，移除最少使用的項目
            if len(self.cache) >= self.capacity:
                min_key = min(self.usage_count, key=self.usage_count.get)
                self.cache.pop(min_key)
                self.usage_count.pop(min_key)
            
            self.cache[key] = value
            self.usage_count[key] = 1
    
    def clear(self) -> None:
        """清空快取"""
        with self.lock:
            self.cache.clear()
            self.usage_count.clear()
    
    def __len__(self) -> int:
        """返回快取中的項目數量"""
        return len(self.cache)


class EarlyExitStrategy:
    """
    提前退出策略 - 根據中間結果決定是否可以提前結束處理
    """
    
    def __init__(self, config: BatchConfig):
        """
        初始化提前退出策略
        
        Args:
            config: 批處理配置
        """
        self.config = config
        self.processed_samples = 0
        self.confident_samples = 0
        self.confidence_scores = []
    
    def should_exit(self, confidence_score: float) -> bool:
        """
        檢查是否應該提前退出處理
        
        Args:
            confidence_score: 當前樣本的置信度分數 (0.0-1.0)
        
        Returns:
            是否應該提前退出
        """
        if not self.config.early_exit_enabled:
            return False
        
        self.processed_samples += 1
        self.confidence_scores.append(confidence_score)
        
        # 如果處理的樣本數少於最小要求，不提前退出
        if self.processed_samples < self.config.min_samples_before_exit:
            return False
        
        # 計算平均置信度
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
        
        # 如果平均置信度超過閾值，提前退出
        return avg_confidence >= self.config.confidence_threshold
    
    def reset(self) -> None:
        """重置策略狀態"""
        self.processed_samples = 0
        self.confident_samples = 0
        self.confidence_scores = []


class BatchProcessor:
    """
    批處理器 - 使用批處理和提前退出策略加速處理
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """
        初始化批處理器
        
        Args:
            config: 批處理配置，如果為None則使用預設配置
        """
        self.config = config or BatchConfig()
        self.resource_monitor = ResourceMonitor(self.config)
        self.result_cache = ResultCache(self.config.cache_size) if self.config.enable_caching else None
        self.early_exit = EarlyExitStrategy(self.config)
        
        # 統計資訊
        self.stats = {
            "total_processed": 0,
            "batch_sizes": [],
            "processing_times": [],
            "early_exits": 0,
            "cache_hits": 0,
            "total_batches": 0,
        }
        
        # 處理佇列
        self.queue = queue.Queue()
        self.is_processing = False
        
        # 當前批量大小
        self.current_batch_size = min(4, self.config.max_batch_size)
    
    def process_batch(
        self,
        items: List[T],
        process_fn: Callable[[List[T]], List[Union[R, Tuple[R, float]]]],
        key_fn: Optional[Callable[[T], str]] = None
    ) -> List[R]:
        """
        批量處理項目
        
        Args:
            items: 要處理的項目列表
            process_fn: 批量處理函數，接受項目列表，返回結果列表
            key_fn: 快取鍵值生成函數，如果啟用了快取則需要提供
        
        Returns:
            處理結果列表
        """
        if not items:
            return []
        
        # 更新統計資訊
        self.stats["total_batches"] += 1
        self.stats["total_processed"] += len(items)
        
        # 更新批量大小建議
        if self.config.adaptive_batch_size:
            self.current_batch_size = self.resource_monitor.get_recommended_batch_size(self.current_batch_size)
        
        # 如果啟用了快取，先檢查快取
        if self.config.enable_caching and key_fn:
            cached_results = []
            items_to_process = []
            keys = []
            
            for item in items:
                key = key_fn(item)
                keys.append(key)
                cached_result = self.result_cache.get(key)
                
                if cached_result is not None:
                    cached_results.append((len(cached_results), cached_result))
                    self.stats["cache_hits"] += 1
                else:
                    items_to_process.append((len(items_to_process), item))
            
            # 處理未快取的項目
            if items_to_process:
                uncached_items = [item for _, item in items_to_process]
                start_time = time.time()
                batch_results = process_fn(uncached_items)
                processing_time = time.time() - start_time
                
                # 更新統計資訊
                self.stats["processing_times"].append(processing_time)
                self.stats["batch_sizes"].append(len(uncached_items))
                
                # 放入快取
                for i, result in enumerate(batch_results):
                    # 檢查結果是否包含置信度
                    if isinstance(result, tuple) and len(result) == 2:
                        actual_result, confidence = result
                        
                        # 檢查是否應該提前退出
                        if self.early_exit.should_exit(confidence):
                            self.stats["early_exits"] += 1
                            # 僅返回已處理的結果
                            processed_indices = [idx for idx, _ in items_to_process[:i+1]]
                            cached_indices = [idx for idx, _ in cached_results]
                            
                            # 合併結果並按原始順序排序
                            all_results = [(idx, res) for idx, res in cached_results]
                            all_results.extend([(items_to_process[j][0], batch_results[j][0] if isinstance(batch_results[j], tuple) else batch_results[j]) for j in range(i+1)])
                            all_results.sort(key=lambda x: x[0])
                            
                            # 優化：只快取已完全處理的結果
                            for j in range(i+1):
                                idx = items_to_process[j][0]
                                key = keys[idx]
                                res = batch_results[j][0] if isinstance(batch_results[j], tuple) else batch_results[j]
                                self.result_cache.put(key, res)
                            
                            return [res for _, res in all_results]
                    else:
                        actual_result = result
                        confidence = 1.0
                    
                    # 更新快取
                    original_idx = items_to_process[i][0]
                    key = keys[original_idx]
                    self.result_cache.put(key, actual_result)
            
            # 合併結果並按原始順序排序
            all_results = [(idx, res) for idx, res in cached_results]
            
            if items_to_process:
                batch_result_list = []
                for i, result in enumerate(batch_results):
                    if isinstance(result, tuple) and len(result) == 2:
                        batch_result_list.append((items_to_process[i][0], result[0]))
                    else:
                        batch_result_list.append((items_to_process[i][0], result))
                
                all_results.extend(batch_result_list)
            
            all_results.sort(key=lambda x: x[0])
            return [res for _, res in all_results]
        
        # 直接處理所有項目
        start_time = time.time()
        results = process_fn(items)
        processing_time = time.time() - start_time
        
        # 更新統計資訊
        self.stats["processing_times"].append(processing_time)
        self.stats["batch_sizes"].append(len(items))
        
        # 檢查提前退出
        for i, result in enumerate(results):
            # 如果結果包含置信度
            if isinstance(result, tuple) and len(result) == 2:
                confidence = result[1]
                
                # 檢查是否應該提前退出
                if self.early_exit.should_exit(confidence):
                    self.stats["early_exits"] += 1
                    # 提取到目前為止處理的所有項的實際結果
                    return [r[0] if isinstance(r, tuple) else r for r in results[:i+1]]
        
        # 提取實際結果
        return [r[0] if isinstance(r, tuple) and len(r) == 2 else r for r in results]
    
    def async_process(
        self,
        items: List[T],
        process_fn: Callable[[List[T]], List[R]],
        callback: Callable[[List[R]], None],
        key_fn: Optional[Callable[[T], str]] = None
    ) -> None:
        """
        非同步批量處理
        
        Args:
            items: 要處理的項目列表
            process_fn: 批量處理函數
            callback: 處理完成後的回調函數
            key_fn: 快取鍵值生成函數
        """
        # 創建任務並放入佇列
        task = (items, process_fn, callback, key_fn)
        self.queue.put(task)
        
        # 如果當前沒有處理線程，啟動一個
        if not self.is_processing:
            self._start_processing_thread()
    
    def _start_processing_thread(self) -> None:
        """啟動處理線程"""
        self.is_processing = True
        
        def process_queue():
            while not self.queue.empty():
                try:
                    # 獲取下一個任務
                    items, process_fn, callback, key_fn = self.queue.get(timeout=0.1)
                    
                    # 處理項目
                    results = self.process_batch(items, process_fn, key_fn)
                    
                    # 調用回調
                    callback(results)
                    
                    # 標記任務完成
                    self.queue.task_done()
                except queue.Empty:
                    pass
                except Exception as e:
                    logger.error(f"處理佇列中的任務時出錯: {e}")
            
            # 處理完所有任務後標記為未處理狀態
            self.is_processing = False
        
        # 啟動處理線程
        threading.Thread(target=process_queue, daemon=True).start()
    
    def adaptive_batch(
        self, 
        items: List[T],
        process_fn: Callable[[List[T]], List[R]],
        key_fn: Optional[Callable[[T], str]] = None
    ) -> List[R]:
        """
        自適應批量處理 - 根據資源使用情況動態調整批量大小
        
        Args:
            items: 要處理的項目列表
            process_fn: 處理函數
            key_fn: 快取鍵值生成函數
        
        Returns:
            處理結果列表
        """
        if not items:
            return []
        
        results = []
        remaining_items = items.copy()
        
        while remaining_items:
            # 獲取當前推薦的批量大小
            batch_size = self.resource_monitor.get_recommended_batch_size(self.current_batch_size)
            self.current_batch_size = batch_size
            
            # 取出下一批項目
            batch = remaining_items[:batch_size]
            remaining_items = remaining_items[batch_size:]
            
            # 處理批次
            batch_results = self.process_batch(batch, process_fn, key_fn)
            results.extend(batch_results)
            
            # 如果啟用了提前退出且達到退出條件，不再處理後續批次
            if self.config.early_exit_enabled and self.early_exit.should_exit(1.0):  # 使用默認置信度
                self.stats["early_exits"] += 1
                break
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        獲取處理統計資訊
        
        Returns:
            統計資訊字典
        """
        stats = self.stats.copy()
        
        # 計算平均批量大小和處理時間
        if stats["batch_sizes"]:
            stats["avg_batch_size"] = sum(stats["batch_sizes"]) / len(stats["batch_sizes"])
        else:
            stats["avg_batch_size"] = 0
        
        if stats["processing_times"]:
            stats["avg_processing_time"] = sum(stats["processing_times"]) / len(stats["processing_times"])
        else:
            stats["avg_processing_time"] = 0
        
        # 計算快取命中率
        if self.config.enable_caching:
            stats["cache_size"] = len(self.result_cache)
            stats["cache_hit_rate"] = stats["cache_hits"] / max(1, stats["total_processed"])
        
        # 計算提前退出率
        stats["early_exit_rate"] = stats["early_exits"] / max(1, stats["total_batches"])
        
        # 新增資源使用情況
        resources = self.resource_monitor.check_resources()
        stats.update(resources)
        
        return stats
    
    def reset_stats(self) -> None:
        """重置統計資訊"""
        self.stats = {
            "total_processed": 0,
            "batch_sizes": [],
            "processing_times": [],
            "early_exits": 0,
            "cache_hits": 0,
            "total_batches": 0,
        }


def test_batch_processor():
    """
    測試批處理器
    """
    import random
    
    def dummy_process_fn(items):
        """模擬處理函數，帶有隨機延遲和置信度分數"""
        time.sleep(0.01 * len(items))  # 模擬處理時間與批量大小成正比
        results = []
        for item in items:
            processed = item * 2
            confidence = random.uniform(0.8, 1.0)
            results.append((processed, confidence))
        return results
    
    def dummy_key_fn(item):
        """生成快取鍵值"""
        return str(item)
    
    # 創建批處理器
    config = BatchConfig(
        max_batch_size=10,
        adaptive_batch_size=True,
        early_exit_enabled=True,
        confidence_threshold=0.95,
        min_samples_before_exit=3,
        enable_caching=True
    )
    
    processor = BatchProcessor(config)
    
    # 準備測試數據
    items = list(range(100))
    
    # 測試無快取處理
    start_time = time.time()
    results1 = processor.process_batch(items[:20], dummy_process_fn)
    time1 = time.time() - start_time
    
    # 測試有快取處理
    start_time = time.time()
    results2 = processor.process_batch(items[:20], dummy_process_fn, dummy_key_fn)
    time2 = time.time() - start_time
    
    # 再次處理相同項目（應該從快取獲取）
    start_time = time.time()
    results3 = processor.process_batch(items[:20], dummy_process_fn, dummy_key_fn)
    time3 = time.time() - start_time
    
    # 測試自適應批量處理
    start_time = time.time()
    results4 = processor.adaptive_batch(items, dummy_process_fn, dummy_key_fn)
    time4 = time.time() - start_time
    
    # 輸出結果
    print(f"無快取處理時間: {time1:.4f}秒")
    print(f"有快取首次處理時間: {time2:.4f}秒")
    print(f"快取命中處理時間: {time3:.4f}秒")
    print(f"自適應批量處理時間: {time4:.4f}秒")
    
    # 輸出統計資訊
    stats = processor.get_stats()
    print("\n統計資訊:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_batch_processor() 