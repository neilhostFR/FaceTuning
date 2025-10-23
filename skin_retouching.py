import os
import cv2
import logging
import traceback
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import concurrent.futures
import numpy as np
import time
import gc

# 尝试导入缓存管理器
try:
    from cache_manager import get_cache_manager
    has_cache_manager = True
except ImportError:
    has_cache_manager = False
    logging.warning("缓存管理器导入失败，将不使用缓存功能")

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("skin_retouching.log"),
                             logging.StreamHandler()])

class SkinRetoucher:
    # 类变量，用于缓存模型路径
    _model_path_cache = {}
    # 类变量，用于缓存模型实例
    _model_instance_cache = {}
    # 类变量，用于缓存处理结果
    _result_cache = {}
    # 缓存命中计数
    _cache_hits = 0
    # 缓存未命中计数
    _cache_misses = 0
    # 最大缓存大小
    _max_cache_size = 100
    
    def __init__(self, model_id='iic/cv_unet_skin-retouching', offline_mode=True, use_cache=True):
        """
        初始化皮肤美化处理器，使用延迟加载模式
        
        Args:
            model_id: ModelScope模型ID
            offline_mode: 是否使用离线模式
            use_cache: 是否使用缓存
        """
        self.model_id = model_id
        self.offline_mode = offline_mode
        self.use_cache = use_cache and has_cache_manager
        self._skin_retouching = None  # 延迟加载
        self.model_dir = None
        
        # 获取缓存管理器
        if self.use_cache:
            try:
                self.cache_manager = get_cache_manager()
                logging.info("已启用缓存管理器")
            except Exception as e:
                logging.error(f"获取缓存管理器失败: {e}")
                self.use_cache = False
        
        try:
            logging.info(f"初始化皮肤美化处理器: {model_id}, 离线模式: {offline_mode}, 使用缓存: {self.use_cache}")
            
            # 检查模型路径
            if offline_mode:
                self.model_dir = self._get_local_model_path(model_id)
                if not self.model_dir:
                    logging.error(f"本地模型不存在: {model_id}")
                    print(f"警告: 本地模型不存在，请先运行 download_modelscope_model.py 下载模型")
                    self.initialized = False
                else:
                    logging.info(f"找到本地模型: {self.model_dir}")
                    self.initialized = True
            else:
                # 在线模式
            self.initialized = True
                
        except Exception as e:
            logging.error(f"皮肤美化处理器初始化失败: {e}")
            logging.error(traceback.format_exc())
            print(f"警告: 皮肤美化处理器初始化失败: {e}")
            self.initialized = False
    
    @property
    def skin_retouching(self):
        """延迟加载模型，只在第一次使用时加载"""
        # 检查类缓存中是否已有模型实例
        cache_key = f"{self.model_id}_{self.offline_mode}"
        if cache_key in self._model_instance_cache:
            return self._model_instance_cache[cache_key]
        
        # 如果实例变量中已有模型，直接返回
        if self._skin_retouching is not None:
            return self._skin_retouching
        
        try:
            if self.offline_mode and self.model_dir:
                logging.info(f"加载本地模型: {self.model_dir}")
                self._skin_retouching = pipeline('skin-retouching', model=self.model_dir)
            elif not self.offline_mode:
                logging.info(f"从ModelScope加载模型: {self.model_id}")
                self._skin_retouching = pipeline('skin-retouching', model=self.model_id)
            else:
                raise ValueError("无法加载模型：本地模型不存在且不允许在线加载")
            
            logging.info("皮肤美化模型加载成功")
            
            # 缓存模型实例
            self._model_instance_cache[cache_key] = self._skin_retouching
            
            return self._skin_retouching
        except Exception as e:
            logging.error(f"加载皮肤美化模型失败: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def _get_local_model_path(self, model_id):
        """
        获取本地模型路径，使用缓存提高性能
        
        Args:
            model_id: ModelScope模型ID
            
        Returns:
            model_path: 本地模型路径
        """
        # 检查缓存中是否已有此模型路径
        if model_id in self._model_path_cache:
            logging.info(f"从缓存获取模型路径: {self._model_path_cache[model_id]}")
            return self._model_path_cache[model_id]
        
        # 默认模型目录
        models_dir = 'models'
        model_name = model_id.replace('/', '_')
        model_path = os.path.join(models_dir, model_name)
        
        # 检查模型是否存在
        if os.path.exists(model_path) and os.path.isdir(model_path):
            # 缓存结果
            self._model_path_cache[model_id] = model_path
            return model_path
        
        # 如果默认目录不存在，尝试查找其他可能的位置
        possible_paths = [
            os.path.join('.', model_name),
            os.path.join('..', 'models', model_name),
            os.path.join(os.path.expanduser('~'), '.cache', 'modelscope', 'hub', model_id)
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                # 缓存结果
                self._model_path_cache[model_id] = path
                return path
        
        return None
    
    def retouch(self, image_path, output_dir="retouched_faces", max_size=1920):
        """
        对图片进行皮肤美化处理
        
        Args:
            image_path: 图片路径或图像数组
            output_dir: 输出目录
            max_size: 图像处理的最大尺寸，超过此尺寸将被缩放
            
        Returns:
            output_path: 处理后的图片路径
        """
        if not self.initialized:
            logging.error("皮肤美化模型未初始化")
            return None
        
        # 生成缓存键
        cache_key = None
        if self.use_cache and isinstance(image_path, str):
            # 使用文件路径和修改时间作为缓存键
            try:
                mtime = os.path.getmtime(image_path)
                file_size = os.path.getsize(image_path)
                cache_key = f"{image_path}_{mtime}_{file_size}_{max_size}"
                
                # 检查缓存
                cached_result = self.cache_manager.get_cached_image(cache_key)
                if cached_result is not None:
                    logging.info(f"使用缓存结果: {image_path}")
                    return cached_result
            except (OSError, FileNotFoundError) as e:
                logging.warning(f"获取文件信息失败，不使用缓存: {e}")
                cache_key = None
        
        logging.info(f"开始皮肤美化处理: {image_path if isinstance(image_path, str) else '图像数组'}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 处理输入图像
            if isinstance(image_path, str):
                # 输入是图片路径
            if not os.path.exists(image_path):
                logging.error(f"图片不存在: {image_path}")
                return None
            
            # 获取文件名（不含扩展名）
            base_name = os.path.basename(image_path)
            file_name = os.path.splitext(base_name)[0]
                
                # 读取图片
                start_time = time.time()
                img = cv2.imread(image_path)
                logging.debug(f"读取图片耗时: {time.time() - start_time:.3f}秒")
                
                if img is None:
                    logging.error(f"无法读取图片: {image_path}")
                    return None
            else:
                # 输入是图像数组
                img = image_path
                # 使用时间戳作为文件名
                # 避免局部导入导致的命名冲突
                current_timestamp = int(time.time())
                file_name = f"retouched_{current_timestamp}"
            
            # 检查图像尺寸，如果过大则调整大小
            h, w = img.shape[:2]
            original_size = (w, h)
            resized = False
            
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                
                start_time = time.time()
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logging.debug(f"调整图片大小耗时: {time.time() - start_time:.3f}秒")
                
                logging.info(f"图片尺寸过大，已调整为 {new_w}x{new_h}")
                resized = True
            
            # 进行皮肤美化处理
            start_time = time.time()
            result = self.skin_retouching(img)
            logging.info(f"皮肤美化处理耗时: {time.time() - start_time:.3f}秒")
            
            # 如果之前调整了大小，现在需要恢复原始大小
            if resized:
                start_time = time.time()
                result_img = result[OutputKeys.OUTPUT_IMG]
                result_img = cv2.resize(result_img, original_size, interpolation=cv2.INTER_LANCZOS4)
                result[OutputKeys.OUTPUT_IMG] = result_img
                logging.debug(f"恢复原始大小耗时: {time.time() - start_time:.3f}秒")
                logging.info(f"已将处理结果恢复到原始尺寸: {original_size[0]}x{original_size[1]}")
            
            # 保存结果
            output_path = os.path.join(output_dir, f"{file_name}_retouched.jpg")
            
            start_time = time.time()
            cv2.imwrite(output_path, result[OutputKeys.OUTPUT_IMG], 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])  # 控制压缩质量
            logging.debug(f"保存图片耗时: {time.time() - start_time:.3f}秒")
            
            logging.info(f"皮肤美化处理完成，结果保存在: {output_path}")
            
            # 缓存结果
            if self.use_cache and cache_key:
                self.cache_manager.cache_image(cache_key, output_path)
            
            return output_path
        
        except Exception as e:
            logging.error(f"皮肤美化处理失败: {e}")
            logging.error(traceback.format_exc())
            print(f"皮肤美化处理失败: {e}")
            return None
    
    def batch_retouch(self, image_paths, output_dir="retouched_faces", max_workers=None, max_size=1920):
        """
        批量处理图片，使用线程池并行处理提高效率
        
        Args:
            image_paths: 图片路径列表
            output_dir: 输出目录
            max_workers: 最大工作线程数，默认为CPU核心数的2倍
            max_size: 图像处理的最大尺寸，超过此尺寸将被缩放
            
        Returns:
            output_paths: 处理后的图片路径列表
        """
        if not self.initialized:
            logging.error("皮肤美化模型未初始化")
            return []
        
        logging.info(f"开始批量皮肤美化处理，共 {len(image_paths)} 张图片")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 确定线程数，默认为CPU核心数的2倍（主要是IO密集型任务）
        if max_workers is None:
            import multiprocessing
            max_workers = min(32, multiprocessing.cpu_count() * 2)
        
        logging.info(f"使用 {max_workers} 个线程进行并行处理")
        
        output_paths = []
        processed_count = 0
        
        # 使用线程池并行处理
        from concurrent.futures import ThreadPoolExecutor
        
        def process_image(image_path):
            try:
                logging.info(f"处理图片: {image_path}")
                return self.retouch(image_path, output_dir, max_size=max_size)
            except Exception as e:
                logging.error(f"处理图片 {image_path} 失败: {e}")
                logging.error(traceback.format_exc())
                return None
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_path = {executor.submit(process_image, path): path for path in image_paths}
            
            # 收集结果
            for i, future in enumerate(concurrent.futures.as_completed(future_to_path)):
                path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        output_paths.append(result)
                        processed_count += 1
                    
                    # 每处理10张图片，强制进行一次垃圾回收
                    if processed_count > 0 and processed_count % 10 == 0:
                        logging.info(f"已处理 {processed_count} 张图片，执行垃圾回收...")
                        gc.collect()
                    
                    # 计算进度和预估剩余时间
                    progress = (i + 1) / len(image_paths) * 100
                    elapsed_time = time.time() - start_time
                    images_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0
                    remaining_time = (len(image_paths) - i - 1) / images_per_second if images_per_second > 0 else 0
                    
                    logging.info(f"处理进度: {progress:.1f}% ({i+1}/{len(image_paths)}), "
                                f"速度: {images_per_second:.2f}张/秒, "
                                f"预计剩余时间: {remaining_time:.1f}秒")
                    
                except Exception as e:
                    logging.error(f"获取处理结果失败 ({path}): {e}")
        
        # 处理完成后进行一次垃圾回收
        gc.collect()
        
        total_time = time.time() - start_time
        avg_time_per_image = total_time / len(image_paths) if image_paths else 0
        
        logging.info(f"批量处理完成，成功处理 {len(output_paths)}/{len(image_paths)} 张图片")
        logging.info(f"总耗时: {total_time:.1f}秒, 平均每张图片: {avg_time_per_image:.1f}秒")
        
        # 如果使用缓存，输出缓存统计
        if self.use_cache:
            stats = self.cache_manager.get_cache_stats()
            logging.info(f"缓存统计: 大小={stats['cache_size']}/{stats['max_cache_size']}, "
                        f"命中率={stats['hit_ratio']*100:.1f}%, "
                        f"内存使用={stats['memory_usage']:.1f}%")
        
        return output_paths


def main():
    # 创建皮肤美化处理器
    retoucher = SkinRetoucher()
    
    # 指定图片路径
    image_path = input("请输入图片路径: ")
    
    # 进行皮肤美化处理
    try:
        output_path = retoucher.retouch(image_path)
        if output_path:
            print(f"皮肤美化处理成功，结果保存在: {output_path}")
            
            # 显示原图和处理后的图片对比
            try:
                import matplotlib.pyplot as plt
                import cv2
                
                # 读取原图和处理后的图片
                original_img = cv2.imread(image_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                retouched_img = cv2.imread(output_path)
                retouched_img = cv2.cvtColor(retouched_img, cv2.COLOR_BGR2RGB)
                
                # 创建图形
                fig, axes = plt.subplots(1, 2, figsize=(15, 7))
                
                # 显示原图
                axes[0].imshow(original_img)
                axes[0].set_title("原图")
                axes[0].axis("off")
                
                # 显示处理后的图片
                axes[1].imshow(retouched_img)
                axes[1].set_title("皮肤美化后")
                axes[1].axis("off")
                
                plt.tight_layout()
                plt.savefig("retouching_comparison.png")
                plt.show()
            
            except ImportError:
                print("未安装matplotlib，无法显示对比图")
        else:
            print("皮肤美化处理失败")
    
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        logging.error(traceback.format_exc())
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main() 