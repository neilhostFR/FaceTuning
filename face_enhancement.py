import os
import cv2
import numpy as np
import logging
import traceback
import matplotlib.pyplot as plt
from face_detection import FaceDetector
from skin_retouching import SkinRetoucher
import platform
import concurrent.futures
import multiprocessing
import gc
import time

# 尝试导入缓存管理器
try:
    from cache_manager import get_cache_manager
    has_cache_manager = True
except ImportError:
    has_cache_manager = False
    logging.warning("缓存管理器导入失败，将不使用缓存功能")

# 导入ModelScope相关模块
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("face_enhancement.log"),
                             logging.StreamHandler()])

class FaceEnhancer:
    # 类变量，用于缓存实例
    _instances = {}
    # 类变量，用于缓存人像增强模型
    _portrait_enhancement_model = None
    
    @classmethod
    def get_instance(cls, offline_mode=True, max_workers=None, model_id='iic/cv_unet_skin-retouching', use_cache=True):
        """获取单例实例，避免重复创建"""
        key = f"{offline_mode}_{max_workers}_{model_id}_{use_cache}"
        if key not in cls._instances:
            cls._instances[key] = cls(offline_mode, max_workers, model_id, use_cache)
        return cls._instances[key]
    
    @classmethod
    def get_portrait_enhancement_model(cls, offline_mode=True):
        """获取人像增强模型的单例实例"""
        if cls._portrait_enhancement_model is None:
            try:
                logging.info("初始化人像增强修复模型")
                if offline_mode:
                    # 检查本地模型是否存在
                    model_dir = os.path.join('models', 'damo_cv_gpen_image-portrait-enhancement-hires')
                    if os.path.exists(model_dir) and os.path.isdir(model_dir):
                        logging.info(f"使用本地人像增强修复模型: {model_dir}")
                        cls._portrait_enhancement_model = pipeline(Tasks.image_portrait_enhancement, model=model_dir)
                    else:
                        logging.info("本地人像增强修复模型不存在，尝试在线加载")
                        cls._portrait_enhancement_model = pipeline(Tasks.image_portrait_enhancement, model='damo/cv_gpen_image-portrait-enhancement-hires')
                else:
                    logging.info("在线加载人像增强修复模型")
                    cls._portrait_enhancement_model = pipeline(Tasks.image_portrait_enhancement, model='damo/cv_gpen_image-portrait-enhancement-hires')
                
                logging.info("人像增强修复模型初始化成功")
            except Exception as e:
                logging.error(f"初始化人像增强修复模型失败: {e}")
                logging.error(traceback.format_exc())
                return None
        
        return cls._portrait_enhancement_model
    
    def __init__(self, offline_mode=True, max_workers=None, model_id='iic/cv_unet_skin-retouching', use_cache=True):
        """
        初始化人脸处理器
        
        Args:
            offline_mode: 是否使用离线模式
            max_workers: 最大工作线程数，默认为CPU核心数
            model_id: ModelScope模型ID
            use_cache: 是否使用缓存
        """
        self.offline_mode = offline_mode
        self.model_id = model_id
        self.use_cache = use_cache and has_cache_manager
        self._face_detector = None  # 延迟加载
        self._skin_retoucher = None  # 延迟加载
        
        # 获取缓存管理器
        if self.use_cache:
            try:
                self.cache_manager = get_cache_manager()
                logging.info("已启用缓存管理器")
            except Exception as e:
                logging.error(f"获取缓存管理器失败: {e}")
                self.use_cache = False
        
        try:
            # 获取当前操作系统
            self.system = platform.system()
            logging.info(f"初始化人脸处理器，当前操作系统: {self.system}, 使用缓存: {self.use_cache}")
            
            # 设置最大工作线程数
            if max_workers is None:
                self.max_workers = min(32, multiprocessing.cpu_count())
            else:
                self.max_workers = max_workers
            logging.info(f"设置最大工作线程数: {self.max_workers}")
            
            self.initialized = True
        except Exception as e:
            logging.error(f"初始化失败: {e}")
            logging.error(traceback.format_exc())
            print(f"警告: 初始化失败: {e}")
            self.initialized = False
    
    @property
    def face_detector(self):
        """延迟加载人脸检测器"""
        if self._face_detector is None:
            logging.info("初始化人脸检测器")
            self._face_detector = FaceDetector()
            if self._face_detector.has_dlib:
                logging.info("人脸检测器初始化成功")
            else:
                logging.warning("人脸检测器初始化失败，瘦脸功能将不可用")
        return self._face_detector
    
    @property
    def skin_retoucher(self):
        """延迟加载皮肤美化器"""
        if self._skin_retoucher is None:
            logging.info(f"初始化皮肤美化器，模型ID: {self.model_id}, 离线模式: {self.offline_mode}, 使用缓存: {self.use_cache}")
            self._skin_retoucher = SkinRetoucher(model_id=self.model_id, offline_mode=self.offline_mode, use_cache=self.use_cache)
            logging.info(f"皮肤美化器初始化成功")
        return self._skin_retoucher
    
    def enhance_face(self, image_path, slim_intensity=0.5, 
                    crop_dir=None, slim_dir=None, 
                    retouch_dir=None, final_dir=None, overwrite_original=False,
                    max_size=1920, apply_portrait_enhancement=True,
                    use_face_slim=False, use_skin_retouch=True, use_portrait_enhancement=False):
        """对图片进行人脸处理
        
        Args:
            image_path: 图片路径
            slim_intensity: 瘦脸强度，范围0-1
            crop_dir: 裁剪后的人脸保存目录，默认为临时目录
            slim_dir: 瘦脸后的人脸保存目录，默认为临时目录
            retouch_dir: 皮肤美化后的人脸保存目录，默认为临时目录
            final_dir: 最终处理后的图片保存目录，默认为输入目录的output子目录
            overwrite_original: 是否覆盖原图（已废弃，保留参数兼容旧代码）
            max_size: 图像处理的最大尺寸，超过此尺寸将被缩放
            apply_portrait_enhancement: 是否应用人像增强修复（已废弃，保留参数兼容旧代码）
            use_face_slim: 是否使用瘦脸功能
            use_skin_retouch: 是否使用皮肤美化功能
            use_portrait_enhancement: 是否使用人像增强功能
            
        Returns:
            最终处理后的图片路径
        """
        if not self.initialized:
            logging.error("人脸处理器未初始化")
            return None
        
        # 生成缓存键
        cache_key = None
        if self.use_cache:
            # 使用文件路径和修改时间作为缓存键
            try:
                mtime = os.path.getmtime(image_path)
                file_size = os.path.getsize(image_path)
                cache_key = f"{image_path}_{mtime}_{file_size}_{slim_intensity}_{max_size}_{use_face_slim}_{use_skin_retouch}_{use_portrait_enhancement}"
                
                # 检查缓存
                cached_result = self.cache_manager.get_cached_image(cache_key)
                if cached_result is not None:
                    logging.info(f"使用缓存结果: {image_path}")
                    return cached_result
            except (OSError, FileNotFoundError) as e:
                logging.warning(f"获取文件信息失败，不使用缓存: {e}")
                cache_key = None
        
        logging.info(f"开始人脸处理: {image_path}, 使用瘦脸: {use_face_slim}, 使用皮肤美化: {use_skin_retouch}, 使用人像增强: {use_portrait_enhancement}")
        
        # 创建临时目录用于处理过程
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_crop_dir = os.path.join(temp_dir, "crop")
        temp_slim_dir = os.path.join(temp_dir, "slim")
        temp_retouch_dir = os.path.join(temp_dir, "retouch")
        temp_portrait_dir = os.path.join(temp_dir, "portrait")
        
        os.makedirs(temp_crop_dir, exist_ok=True)
        os.makedirs(temp_slim_dir, exist_ok=True)
        os.makedirs(temp_retouch_dir, exist_ok=True)
        os.makedirs(temp_portrait_dir, exist_ok=True)
        
        # 如果需要保存中间结果，创建相应目录
        if crop_dir:
            os.makedirs(crop_dir, exist_ok=True)
        if slim_dir:
            os.makedirs(slim_dir, exist_ok=True)
        if retouch_dir:
            os.makedirs(retouch_dir, exist_ok=True)
        
        # 设置默认输出目录为输入目录的output子目录
        if final_dir is None:
            input_dir = os.path.dirname(image_path)
            final_dir = os.path.join(input_dir, "output")
        
        # 创建输出目录
            os.makedirs(final_dir, exist_ok=True)
        
        try:
            # 读取原始图片
            start_time = time.time()
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"无法读取图片: {image_path}")
                return None
                
            h, w = img.shape[:2]
            original_size = (w, h)
            
            # 创建一个副本用于处理
            intermediate_img = img.copy()
            intermediate_path = os.path.join(temp_dir, "intermediate.jpg")
            cv2.imwrite(intermediate_path, intermediate_img)
            
            # 如果启用瘦脸功能，则进行人脸检测和瘦脸处理
            if use_face_slim:
                # 在原始尺寸图片上进行人脸检测
                logging.info("在原始尺寸图片上进行人脸检测")
                
                # 裁剪并瘦脸（在原始尺寸上进行）
            logging.info("裁剪并瘦脸")
                start_time = time.time()
            result_list = self.face_detector.crop_and_slim_face(
                image_path, 
                output_dir=temp_crop_dir, 
                slim_output_dir=temp_slim_dir, 
                slim_intensity=slim_intensity
            )
                logging.debug(f"裁剪并瘦脸耗时: {time.time() - start_time:.3f}秒")
            
            if not result_list:
                logging.warning(f"未检测到人脸: {image_path}")
                    # 如果未检测到人脸但需要继续处理，不返回None
                    if not use_skin_retouch and not use_portrait_enhancement:
                print(f"未检测到人脸: {image_path}")
                return None
                else:
            logging.info(f"检测到 {len(result_list)} 个人脸")
            
                    # 保存中间结果（如果需要）
                    if crop_dir:
                        for i, crop_info in enumerate(result_list):
                            if "output_path" in crop_info:
                                import shutil
                                crop_path = crop_info["output_path"]
                                dest_path = os.path.join(crop_dir, f"{os.path.basename(image_path).split('.')[0]}_face_{i}.jpg")
                                shutil.copy2(crop_path, dest_path)
                    
                    if slim_dir:
                        for i, crop_info in enumerate(result_list):
                            if "slimmed_path" in crop_info:
                                import shutil
                                slim_path = crop_info["slimmed_path"]
                                dest_path = os.path.join(slim_dir, f"{os.path.basename(image_path).split('.')[0]}_face_{i}_slim.jpg")
                                shutil.copy2(slim_path, dest_path)
            
                    # 将瘦脸后的人脸还原到原图
                    logging.info("将瘦脸后的人脸还原到原图")
            
                    # 读取原图
                    start_time = time.time()
                    original_img = cv2.imread(image_path)
                    if original_img is None:
                        logging.error(f"无法读取原图: {image_path}")
                        return None
                    
                    # 创建一个副本用于合成
                    intermediate_img = original_img.copy()
                    
                    # 合成瘦脸后的人脸到原图
                    for i, crop_info in enumerate(result_list):
                        try:
                            # 使用瘦脸后的图片路径，如果没有则使用裁剪后的图片路径
                            if "slimmed_path" in crop_info:
                                face_path = crop_info["slimmed_path"]
                            elif "output_path" in crop_info:
                                face_path = crop_info["output_path"]
                                logging.warning(f"人脸 {i} 没有瘦脸结果，使用原始裁剪图片")
                            else:
                                logging.warning(f"人脸 {i} 没有有效的图片路径")
                                continue
                            
                            # 获取人脸位置信息
                            if "crop_box" in crop_info:
                                # 从crop_box中提取坐标信息
                                square_x1, square_y1, square_x2, square_y2 = crop_info["crop_box"]
                                x, y = square_x1, square_y1
                                w, h = square_x2 - square_x1, square_y2 - square_y1
                                
                                # 读取瘦脸后的人脸
                                face_img = cv2.imread(face_path)
                                if face_img is None:
                                    logging.error(f"无法读取瘦脸后的人脸图片: {face_path}")
                                    continue
                                
                                # 调整大小以匹配原始人脸区域
                                face_img = cv2.resize(face_img, (w, h))
                                
                                # 直接将瘦脸后的人脸放到原图上，不使用任何融合或羽化
                                logging.info(f"直接将瘦脸后的人脸放到原图位置: x={x}, y={y}, w={w}, h={h}")
                                intermediate_img[y:y+h, x:x+w] = face_img
                            else:
                                logging.warning(f"人脸 {i} 缺少crop_box信息，无法放置到原图")
                        except Exception as e:
                            logging.error(f"合成人脸图像失败: {e}")
                            logging.error(traceback.format_exc())
                    
                    logging.debug(f"合成瘦脸人脸耗时: {time.time() - start_time:.3f}秒")
                    
                    # 保存中间结果到临时文件
                    cv2.imwrite(intermediate_path, intermediate_img)
            
            # 如果启用皮肤美化功能，则进行皮肤美化处理
            if use_skin_retouch:
                # 对整个图像进行皮肤美化处理
                logging.info("对整个图像进行皮肤美化处理")
                
                # 进行皮肤美化前检查图像尺寸，如果过大则调整大小
                if max(h, w) > max_size:
                    logging.info(f"图像尺寸过大，在进行皮肤美化前调整大小")
                    scale = max_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    intermediate_img = cv2.resize(intermediate_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    # 重新保存中间结果
                    cv2.imwrite(intermediate_path, intermediate_img)
                    logging.info(f"已将图像调整为 {new_w}x{new_h} 用于皮肤美化")
                    resized_for_retouching = True
                else:
                    resized_for_retouching = False
            
            # 进行皮肤美化
                start_time = time.time()
                final_path = self.skin_retoucher.retouch(
                    intermediate_path, 
                    output_dir=temp_retouch_dir,
                    max_size=max_size
                )
                logging.debug(f"皮肤美化处理耗时: {time.time() - start_time:.3f}秒")
            
            if not final_path:
                logging.warning("皮肤美化失败，使用中间结果")
                    final_path = intermediate_path
            else:
                # 如果不使用皮肤美化，直接使用中间结果
                final_path = intermediate_path
            
            # 应用人像增强修复
            if use_portrait_enhancement:
                try:
                    # 获取人像增强模型
                    portrait_model = self.get_portrait_enhancement_model(self.offline_mode)
                    
                    if portrait_model:
                        logging.info("开始应用人像增强修复")
                        start_time = time.time()
                        
                        # 读取皮肤美化后的图片
                        portrait_img = cv2.imread(final_path)
                        if portrait_img is None:
                            logging.error(f"无法读取皮肤美化后的图片: {final_path}")
                        else:
                            # 检查图像尺寸，如果太大则调整到1024*768
                            p_h, p_w = portrait_img.shape[:2]
                            portrait_resized = False
                            portrait_original_size = (p_w, p_h)
                            
                            if max(p_w, p_h) > 1024:
                                logging.info("图像尺寸过大，调整到适合人像增强的尺寸")
                                if p_w > p_h:
                                    new_w = 1024
                                    new_h = int(p_h * (1024 / p_w))
                                else:
                                    new_h = 768
                                    new_w = int(p_w * (768 / p_h))
                                
                                portrait_img = cv2.resize(portrait_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                portrait_resized = True
                                logging.info(f"已将图像调整为 {new_w}x{new_h} 用于人像增强")
                            
                            # 应用人像增强
                            try:
                                portrait_result = portrait_model(portrait_img)
                                enhanced_img = portrait_result[OutputKeys.OUTPUT_IMG]
                                
                                # 保存增强后的图片
                                portrait_path = os.path.join(temp_portrait_dir, "portrait_enhanced.jpg")
                                cv2.imwrite(portrait_path, enhanced_img)
                                logging.info(f"人像增强修复完成，保存到: {portrait_path}")
                                
                                # 如果之前调整了尺寸，现在恢复到原始尺寸
                                if portrait_resized:
                                    enhanced_img = cv2.resize(enhanced_img, portrait_original_size, interpolation=cv2.INTER_LANCZOS4)
                                    cv2.imwrite(portrait_path, enhanced_img)
                                    logging.info(f"已将人像增强结果恢复到原始尺寸: {portrait_original_size[0]}x{portrait_original_size[1]}")
                                
                                # 更新最终路径
                                final_path = portrait_path
                                logging.debug(f"人像增强修复处理耗时: {time.time() - start_time:.3f}秒")
                            except Exception as e:
                                logging.error(f"人像增强处理失败: {e}")
                                logging.error(traceback.format_exc())
                                # 继续使用皮肤美化的结果
                    else:
                        logging.warning("人像增强模型未初始化，跳过人像增强步骤")
                except Exception as e:
                    logging.error(f"应用人像增强时出错: {e}")
                    logging.error(traceback.format_exc())
            
            # 如果为了皮肤美化调整了图像大小，现在将结果调整回原始大小
            if use_skin_retouch and 'resized_for_retouching' in locals() and resized_for_retouching:
                final_img = cv2.imread(final_path)
                if final_img is not None:
                    final_img = cv2.resize(final_img, original_size, interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(final_path, final_img)
                    logging.info(f"已将最终结果恢复到原始尺寸: {original_size[0]}x{original_size[1]}")
            
            # 保存最终结果
            # 不再支持覆盖原图，始终保存到指定目录
            # 保持原文件名和后缀名
            base_name = os.path.basename(image_path)
                output_path = os.path.join(final_dir, base_name)
            
            # 复制最终结果到输出路径
            import shutil
            shutil.copy2(final_path, output_path)
            
            # 保存中间结果（如果需要）
            if retouch_dir:
                dest_path = os.path.join(retouch_dir, base_name)
                shutil.copy2(final_path, dest_path)
            
            logging.info(f"人脸处理完成，最终结果保存在: {output_path}")
            
            # 清理临时目录
            try:
            shutil.rmtree(temp_dir)
            except Exception as e:
                logging.error(f"清理临时目录失败: {e}")
            
            # 缓存结果
            if self.use_cache and cache_key:
                self.cache_manager.cache_image(cache_key, output_path)
            
            return output_path
        
        except Exception as e:
            logging.error(f"人脸处理失败: {e}")
            logging.error(traceback.format_exc())
            print(f"人脸处理失败: {e}")
            
            # 清理临时目录
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
            
            return None
    
    def batch_enhance(self, image_paths, slim_intensity=0.5, 
                      output_dir=None, overwrite_original=False,
                      max_size=1920, move_rules=None, apply_portrait_enhancement=True,
                      use_face_slim=False, use_skin_retouch=True, use_portrait_enhancement=False):
        """
        批量处理图片，使用多线程并行处理
        
        Args:
            image_paths: 图片路径列表
            slim_intensity: 瘦脸强度，范围0-1
            output_dir: 输出目录，默认为每个输入图片所在目录的output子目录
            overwrite_original: 是否覆盖原图（已废弃，保留参数兼容旧代码）
            max_size: 图像处理的最大尺寸，超过此尺寸将被缩放
            move_rules: 移动规则列表，格式为[(关键词, 目标目录), ...]
            apply_portrait_enhancement: 是否应用人像增强修复（已废弃，保留参数兼容旧代码）
            use_face_slim: 是否使用瘦脸功能
            use_skin_retouch: 是否使用皮肤美化功能
            use_portrait_enhancement: 是否使用人像增强功能
            
        Returns:
            output_paths: 处理后的图片路径列表
        """
        if not self.initialized:
            logging.error("人脸处理器未初始化")
            return []
        
        # 设置默认输出目录
        if output_dir is None:
            # 默认输出目录将在enhance_face中设置为每个输入图片所在目录的output子目录
            pass
        else:
            # 创建用户指定的输出目录
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"使用指定的默认输出目录: {output_dir}")
        
        logging.info(f"开始批量处理，共 {len(image_paths)} 张图片，使用瘦脸: {use_face_slim}, 使用皮肤美化: {use_skin_retouch}, 使用人像增强: {use_portrait_enhancement}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 记录内存使用情况
        if has_cache_manager:
            initial_memory = get_cache_manager().get_memory_usage()
            logging.info(f"初始内存使用率: {initial_memory:.1f}%")
        
        output_paths = []
        processed_count = 0
        
        # 使用线程池并行处理
        def process_image(image_path):
            try:
                logging.info(f"处理图片: {image_path}")
                result_path = self.enhance_face(
                    image_path, 
                    slim_intensity=slim_intensity,
                    final_dir=output_dir,  # 如果为None，将使用默认的output子目录
                    max_size=max_size,
                    use_face_slim=use_face_slim,
                    use_skin_retouch=use_skin_retouch,
                    use_portrait_enhancement=use_portrait_enhancement
                )
                
                # 如果处理成功且有移动规则，应用移动规则
                if result_path and move_rules:
                    result_path = self._apply_move_rule(result_path, move_rules)
                
                return result_path
            except Exception as e:
                logging.error(f"处理图片 {image_path} 失败: {e}")
                logging.error(traceback.format_exc())
                return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
                        
                        # 记录内存使用情况
                        if has_cache_manager:
                            current_memory = get_cache_manager().get_memory_usage()
                            logging.info(f"当前内存使用率: {current_memory:.1f}%")
                    
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
        
        # 计算总耗时和平均每张图片的处理时间
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

    def _apply_move_rule(self, file_path, move_rules):
        """
        应用移动规则，将文件移动到匹配的目标目录
        
        Args:
            file_path: 文件路径
            move_rules: 移动规则列表，格式为[(关键词, 目标目录), ...]
            
        Returns:
            new_path: 移动后的文件路径，如果没有移动则返回原路径
        """
        if not move_rules:
            return file_path
        
        file_name = os.path.basename(file_path)
        
        # 检查文件名是否匹配任何规则
        for keyword, target_dir in move_rules:
            if keyword.lower() in file_name.lower():
                # 创建目标目录
                os.makedirs(target_dir, exist_ok=True)
                
                # 构建新路径
                new_path = os.path.join(target_dir, file_name)
                
                # 移动文件
                try:
                    import shutil
                    shutil.move(file_path, new_path)
                    logging.info(f"文件 {file_name} 匹配规则 '{keyword}'，已移动到 {target_dir}")
                    return new_path
                except Exception as e:
                    logging.error(f"移动文件 {file_path} 到 {new_path} 失败: {e}")
                    return file_path
        
        # 如果没有匹配任何规则，返回原路径
        return file_path

    def display_results(self, original_path, final_path):
        """
        显示原图和最终处理后的图片对比
        
        Args:
            original_path: 原图路径
            final_path: 最终处理后的图片路径
        """
        try:
            # 读取原图和最终处理后的图片
            original_img = cv2.imread(original_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            final_img = cv2.imread(final_path)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            
            # 创建图形
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            
            # 显示原图
            axes[0].imshow(original_img)
            axes[0].set_title("原图")
            axes[0].axis("off")
            
            # 显示最终处理后的图片
            axes[1].imshow(final_img)
            axes[1].set_title("处理后")
            axes[1].axis("off")
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            logging.error(f"显示结果失败: {e}")
            logging.error(traceback.format_exc())
            print(f"显示结果失败: {e}")


def main():
    # 创建人脸增强器
    enhancer = FaceEnhancer()
    
    # 指定图片路径
    image_path = input("请输入图片路径: ")
    
    # 设置瘦脸强度
    try:
        intensity = float(input("请输入瘦脸强度 (0.0-1.0): "))
        intensity = max(0.0, min(1.0, intensity))  # 限制在0-1范围内
    except ValueError:
        print("无效的瘦脸强度，使用默认值: 0.5")
        intensity = 0.5
    
    # 进行人脸处理
    try:
        final_path = enhancer.enhance_face(
            image_path, 
            slim_intensity=intensity
        )
        
        if final_path:
            print(f"人脸处理成功，结果保存在: {final_path}")
            
            # 显示原图和最终处理后的图片对比
            enhancer.display_results(image_path, final_path)
        else:
            print("人脸处理失败")
    
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        logging.error(traceback.format_exc())
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main() 