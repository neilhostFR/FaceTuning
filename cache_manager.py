import os
import time
import logging
import gc
import psutil
import threading
from collections import OrderedDict
from functools import lru_cache

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("cache_manager.log"),
                             logging.StreamHandler()])

class CacheManager:
    """
    缓存管理器，用于管理图像缓存和内存使用
    
    功能：
    1. LRU图像缓存：限制内存中保存的图像数量
    2. 定期内存清理：监控内存使用并在需要时进行垃圾回收
    3. 资源监控：跟踪内存和CPU使用情况
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, max_cache_size=100, memory_threshold=80, cleanup_interval=60):
        """获取单例实例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(max_cache_size, memory_threshold, cleanup_interval)
            return cls._instance
    
    def __init__(self, max_cache_size=100, memory_threshold=80, cleanup_interval=60):
        """
        初始化缓存管理器
        
        Args:
            max_cache_size: 最大缓存图像数量
            memory_threshold: 内存使用阈值（百分比），超过此值将触发清理
            cleanup_interval: 清理检查间隔（秒）
        """
        self.max_cache_size = max_cache_size
        self.memory_threshold = memory_threshold
        self.cleanup_interval = cleanup_interval
        
        # 使用OrderedDict实现LRU缓存
        self.image_cache = OrderedDict()
        
        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_cleanup_count = 0
        
        # 启动内存监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        
        logging.info(f"缓存管理器初始化完成，最大缓存大小: {max_cache_size}，内存阈值: {memory_threshold}%")
    
    def cache_image(self, key, image):
        """
        缓存图像
        
        Args:
            key: 缓存键（通常是图像路径）
            image: 图像数据
        """
        # 如果缓存已满，移除最久未使用的项
        if len(self.image_cache) >= self.max_cache_size:
            self.image_cache.popitem(last=False)
            logging.debug("缓存已满，移除最久未使用的图像")
        
        # 添加新图像到缓存
        self.image_cache[key] = image
        logging.debug(f"图像已缓存: {key}")
    
    def get_cached_image(self, key):
        """
        获取缓存的图像
        
        Args:
            key: 缓存键
            
        Returns:
            image: 缓存的图像，如果不存在则返回None
        """
        if key in self.image_cache:
            # 将访问的项移到末尾（最近使用）
            image = self.image_cache.pop(key)
            self.image_cache[key] = image
            self.cache_hits += 1
            logging.debug(f"缓存命中: {key}")
            return image
        else:
            self.cache_misses += 1
            logging.debug(f"缓存未命中: {key}")
            return None
    
    def clear_cache(self):
        """清空缓存"""
        self.image_cache.clear()
        gc.collect()
        logging.info("缓存已清空，执行垃圾回收")
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        return {
            "cache_size": len(self.image_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "total_cleanup_count": self.total_cleanup_count,
            "memory_usage": self.get_memory_usage(),
            "cpu_usage": self.get_cpu_usage()
        }
    
    def _monitor_memory(self):
        """内存监控线程"""
        while True:
            try:
                # 检查内存使用情况
                memory_percent = self.get_memory_usage()
                
                if memory_percent > self.memory_threshold:
                    logging.warning(f"内存使用率 ({memory_percent:.1f}%) 超过阈值 ({self.memory_threshold}%)，执行清理")
                    self._cleanup_memory()
                
                # 等待下一次检查
                time.sleep(self.cleanup_interval)
            
            except Exception as e:
                logging.error(f"内存监控线程出错: {e}")
                time.sleep(self.cleanup_interval)
    
    def _cleanup_memory(self):
        """清理内存"""
        # 清空一半的缓存
        items_to_remove = len(self.image_cache) // 2
        if items_to_remove > 0:
            for _ in range(items_to_remove):
                if self.image_cache:
                    self.image_cache.popitem(last=False)
            
            logging.info(f"已从缓存中移除 {items_to_remove} 个图像")
        
        # 强制垃圾回收
        gc.collect()
        
        self.total_cleanup_count += 1
        logging.info(f"内存清理完成，当前内存使用率: {self.get_memory_usage():.1f}%")
    
    @staticmethod
    def get_memory_usage():
        """获取当前内存使用率（百分比）"""
        return psutil.virtual_memory().percent
    
    @staticmethod
    def get_cpu_usage():
        """获取当前CPU使用率（百分比）"""
        return psutil.cpu_percent(interval=0.1)
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def get_file_size(file_path):
        """获取文件大小（带缓存）"""
        try:
            return os.path.getsize(file_path)
        except (OSError, FileNotFoundError):
            return 0

# 全局函数，用于获取缓存管理器实例
def get_cache_manager():
    return CacheManager.get_instance() 