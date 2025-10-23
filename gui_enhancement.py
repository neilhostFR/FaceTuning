import os
import sys
import cv2
import logging
import traceback
import time
import threading
import queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFileDialog, QSlider, QCheckBox, 
                            QProgressBar, QMessageBox, QTabWidget, QGroupBox, QRadioButton,
                            QLineEdit, QComboBox, QGridLayout, QSpacerItem, QSizePolicy, QListWidget,
                            QDialog, QDialogButtonBox)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QMutex
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import platform

from face_enhancement import FaceEnhancer
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from download_modelscope_model import download_skin_retouching_model

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("gui_enhancement.log"),
                             logging.StreamHandler()])

# 获取当前操作系统
SYSTEM = platform.system()
logging.info(f"当前操作系统: {SYSTEM}")

class EnhancementWorker(QThread):
    """处理线程，避免界面卡顿"""
    finished = pyqtSignal(str)  # 处理完成信号，返回结果路径
    progress = pyqtSignal(int)  # 进度信号
    error = pyqtSignal(str)     # 错误信号
    
    def __init__(self, image_path, slim_intensity, overwrite_original, max_size=1920,
                 use_face_slim=False, use_skin_retouch=True, use_portrait_enhancement=False):
        super().__init__()
        self.image_path = image_path
        self.slim_intensity = slim_intensity
        self.overwrite_original = overwrite_original
        self.max_size = max_size
        self.use_face_slim = use_face_slim
        self.use_skin_retouch = use_skin_retouch
        self.use_portrait_enhancement = use_portrait_enhancement
        
    def run(self):
        try:
            self.progress.emit(10)
            
            # 使用单例模式获取FaceEnhancer实例
            enhancer = FaceEnhancer.get_instance(offline_mode=True, use_cache=True)
            self.progress.emit(20)
            
            # 进行人脸处理
            final_path = enhancer.enhance_face(
                self.image_path, 
                slim_intensity=self.slim_intensity,
                overwrite_original=self.overwrite_original,
                max_size=self.max_size,
                use_face_slim=self.use_face_slim,
                use_skin_retouch=self.use_skin_retouch,
                use_portrait_enhancement=self.use_portrait_enhancement
            )
            
            self.progress.emit(100)
            
            if final_path:
                self.finished.emit(final_path)
            else:
                self.error.emit("人脸处理失败")
        
        except Exception as e:
            logging.error(f"处理出错: {e}")
            logging.error(traceback.format_exc())
            self.error.emit(f"处理出错: {e}")


class ImageViewer(QWidget):
    """图片查看器组件"""
    def __init__(self, title="图片"):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 标题
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)
        
        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.layout.addWidget(self.image_label)
        
        # 图片路径
        self.path_label = QLabel()
        self.path_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.path_label)
        
        self.pixmap = None
    
    def set_image(self, image_path):
        """设置图片"""
        if not os.path.exists(image_path):
            self.image_label.setText("图片不存在")
            return False
        
        self.pixmap = QPixmap(image_path)
        if self.pixmap.isNull():
            self.image_label.setText("无法加载图片")
            return False
        
        # 调整图片大小以适应标签
        self.pixmap = self.pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(self.pixmap)
        self.path_label.setText(image_path)
        return True
    
    def clear(self):
        """清除图片"""
        self.image_label.clear()
        self.path_label.clear()
        self.pixmap = None


class ComparisonViewer(QWidget):
    """图片对比查看器"""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # 初始化子图
        self.axes = None
    
    def compare_images(self, original_path, result_path):
        """比较两张图片"""
        if not os.path.exists(original_path) or not os.path.exists(result_path):
            return False
        
        # 清除之前的图形
        self.figure.clear()
        
        # 创建两个子图
        self.axes = self.figure.subplots(1, 2)
        
        # 读取图片
        original_img = cv2.imread(original_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        result_img = cv2.imread(result_path)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # 显示图片
        self.axes[0].imshow(original_img)
        self.axes[0].set_title("原图")
        self.axes[0].axis("off")
        
        self.axes[1].imshow(result_img)
        self.axes[1].set_title("处理后")
        self.axes[1].axis("off")
        
        self.figure.tight_layout()
        self.canvas.draw()
        return True
    
    def clear(self):
        """清除图片"""
        self.figure.clear()
        self.canvas.draw()


class FileWatcherThread(QThread):
    """文件监听线程，监听文件夹中的新增图片"""
    file_added = pyqtSignal(str)  # 新增文件信号
    
    def __init__(self, input_dir, file_queue):
        super().__init__()
        self.input_dir = input_dir
        self.file_queue = file_queue
        self.running = True
        self.mutex = QMutex()
        
    def run(self):
        """运行线程"""
        class ImageHandler(FileSystemEventHandler):
            def __init__(self, thread_instance):
                self.thread = thread_instance
                self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                
            def on_created(self, event):
                if not event.is_directory:
                    file_path = event.src_path
                    # 只处理直接位于输入目录下的文件，不处理子目录中的文件
                    if os.path.dirname(file_path) == self.thread.input_dir:
                    _, ext = os.path.splitext(file_path.lower())
                    if ext in self.image_extensions:
                        # 等待文件写入完成
                        time.sleep(0.5)
                        self.thread.mutex.lock()
                        if self.thread.running:
                            self.thread.file_queue.put(file_path)
                            self.thread.file_added.emit(file_path)
                        self.thread.mutex.unlock()
        
        event_handler = ImageHandler(self)
        observer = Observer()
        # 不使用recursive=True，只监听指定目录
        observer.schedule(event_handler, self.input_dir, recursive=False)
        observer.start()
        
        try:
            while self.running:
                time.sleep(1)
        except Exception as e:
            logging.error(f"文件监听线程出错: {e}")
        finally:
            observer.stop()
            observer.join()
    
    def stop(self):
        """停止线程"""
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()


class BatchProcessingWorker(QThread):
    """批量处理线程"""
    progress = pyqtSignal(int)  # 进度信号
    status = pyqtSignal(str)    # 状态信号
    finished = pyqtSignal(int)  # 完成信号，返回处理的图片数量
    memory_usage = pyqtSignal(float)  # 内存使用率信号
    
    def __init__(self, input_dir, slim_intensity, recursive, skip_existing, move_rules, file_queue=None, max_size=1920, default_output_dir=None, use_face_slim=False, use_skin_retouch=True, use_portrait_enhancement=False):
        super().__init__()
        self.input_dir = input_dir
        self.slim_intensity = slim_intensity
        self.recursive = recursive
        self.skip_existing = skip_existing
        self.move_rules = move_rules
        self.running = True
        self.mutex = QMutex()
        self.file_queue = file_queue
        self.processed_count = 0
        self.max_size = max_size
        self.default_output_dir = default_output_dir if default_output_dir else os.path.join(input_dir, "output")
        self.use_face_slim = use_face_slim
        self.use_skin_retouch = use_skin_retouch
        self.use_portrait_enhancement = use_portrait_enhancement
        
        # 尝试导入缓存管理器
        try:
            from cache_manager import get_cache_manager
            self.cache_manager = get_cache_manager()
            self.has_cache_manager = True
        except ImportError:
            self.has_cache_manager = False
        
    def run(self):
        """运行线程"""
        try:
            # 使用单例模式获取FaceEnhancer实例
            enhancer = FaceEnhancer.get_instance(offline_mode=True, use_cache=True)
            
            # 先处理目录中已有的图片
            self.status.emit("正在处理目录中已有的图片...")
            self.progress.emit(10)
            
            # 获取目录中的所有图片
            image_paths = []
            if self.recursive:
                for root, _, files in os.walk(self.input_dir):
                    for file in files:
                        if self._is_image_file(file):
                            file_path = os.path.join(root, file)
                            # 检查是否需要跳过
                            if self.skip_existing and self._is_processed(file_path):
                                logging.info(f"跳过已处理的图片: {file_path}")
                                continue
                            image_paths.append(file_path)
            else:
                for file in os.listdir(self.input_dir):
                    file_path = os.path.join(self.input_dir, file)
                    if os.path.isfile(file_path) and self._is_image_file(file):
                        # 检查是否需要跳过
                        if self.skip_existing and self._is_processed(file_path):
                            logging.info(f"跳过已处理的图片: {file_path}")
                            continue
                        image_paths.append(file_path)
            
            if image_paths:
                self.status.emit(f"找到 {len(image_paths)} 张图片，开始处理...")
                
                # 获取转换后的移动规则
                converted_rules = self.get_move_rules_for_enhancer()
                
                # 使用多线程批量处理
                output_paths = enhancer.batch_enhance(
                    image_paths,
                    slim_intensity=self.slim_intensity,
                    output_dir=self.default_output_dir,  # 使用默认输出目录
                    max_size=self.max_size,      # 最大图像尺寸
                    move_rules=converted_rules,   # 使用转换后的移动规则
                    use_face_slim=self.use_face_slim,
                    use_skin_retouch=self.use_skin_retouch,
                    use_portrait_enhancement=self.use_portrait_enhancement
            )
            
                self.processed_count += len(output_paths)
                
                # 发送内存使用率信号
                if self.has_cache_manager:
                    memory_usage = self.cache_manager.get_memory_usage()
                    self.memory_usage.emit(memory_usage)
            
            self.progress.emit(50)
            
            # 如果有文件队列，则进入监听模式
            if self.file_queue:
                self.status.emit(f"已处理 {self.processed_count} 张图片，正在监听新图片...")
                
                # 处理队列中的图片
                while self.running:
                    try:
                        # 非阻塞方式获取队列中的图片
                        try:
                            file_path = self.file_queue.get(block=True, timeout=1)
                        except queue.Empty:
                            # 发送内存使用率信号
                            if self.has_cache_manager:
                                memory_usage = self.cache_manager.get_memory_usage()
                                self.memory_usage.emit(memory_usage)
                            continue
                        
                        self.status.emit(f"正在处理新图片: {os.path.basename(file_path)}")
                        
                        # 处理图片
                        try:
                            # 获取转换后的移动规则
                            converted_rules = self.get_move_rules_for_enhancer()
                            
                            # 处理单个图片
                            final_path = enhancer.enhance_face(
                                file_path, 
                                slim_intensity=self.slim_intensity,
                                final_dir=self.default_output_dir,  # 使用默认输出目录
                                max_size=self.max_size,
                                use_face_slim=self.use_face_slim,
                                use_skin_retouch=self.use_skin_retouch,
                                use_portrait_enhancement=self.use_portrait_enhancement
                            )
                            
                            # 应用移动规则
                            if final_path and converted_rules:
                                # 使用FaceEnhancer的_apply_move_rule方法
                                final_path = enhancer._apply_move_rule(final_path, converted_rules)
                            
                            if final_path:
                                self.processed_count += 1
                                self.status.emit(f"已处理 {self.processed_count} 张图片，正在监听新图片...")
                            
                            # 发送内存使用率信号
                            if self.has_cache_manager:
                                memory_usage = self.cache_manager.get_memory_usage()
                                self.memory_usage.emit(memory_usage)
                            
                        except Exception as e:
                            logging.error(f"处理图片出错: {file_path}, 错误: {e}")
                            logging.error(traceback.format_exc())
                        
                        # 标记任务完成
                        self.file_queue.task_done()
                        
                    except Exception as e:
                        logging.error(f"处理队列中的图片出错: {e}")
                        logging.error(traceback.format_exc())
            
            self.progress.emit(100)
            self.finished.emit(self.processed_count)
            
        except Exception as e:
            logging.error(f"批量处理线程出错: {e}")
            logging.error(traceback.format_exc())
            self.status.emit(f"处理出错: {e}")
    
    def _is_image_file(self, filename):
        """检查文件是否为图片"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        _, ext = os.path.splitext(filename.lower())
        return ext in image_extensions
    
    def _is_processed(self, file_path):
        """检查图片是否已处理"""
        # 简单检查：如果文件名中包含"_beautified"或"_enhanced"，则认为已处理
        base_name = os.path.basename(file_path).lower()
        return "_beautified" in base_name or "_enhanced" in base_name
    
    def _apply_move_rules(self, file_paths):
        """应用移动规则到多个文件"""
        for file_path in file_paths:
            self._apply_move_rule(file_path)
    
    def _apply_move_rule(self, file_path):
        """应用移动规则到单个文件"""
        # 获取图片文件名（不含路径）
        file_name = os.path.basename(file_path)
        
        # 检查是否匹配任何规则
        for rule in self.move_rules:
            keyword = rule["keyword"]
            target_dir = rule["target_dir"]
            
            if keyword.lower() in file_name.lower():
                # 构建目标路径
                if os.path.isabs(target_dir):
                    # 如果是绝对路径，直接使用
                    dest_dir = target_dir
                else:
                    # 如果是相对路径，基于输入目录
                    dest_dir = os.path.join(self.input_dir, target_dir)
                
                # 确保目标目录存在
                os.makedirs(dest_dir, exist_ok=True)
                
                # 构建目标文件路径
                dest_path = os.path.join(dest_dir, file_name)
                
                # 移动文件
                try:
                    import shutil
                    shutil.move(file_path, dest_path)  # 使用move而不是copy2，因为我们要移动文件
                    logging.info(f"已将图片移动到: {dest_path}")
                    return dest_path  # 返回新路径
                except Exception as e:
                    logging.error(f"移动图片失败: {e}")
                    return file_path  # 如果移动失败，返回原路径
                
                # 找到第一个匹配的规则后停止
                break
        
        return file_path  # 如果没有匹配的规则，返回原路径
    
    def get_move_rules_for_enhancer(self):
        """将GUI中的移动规则转换为FaceEnhancer期望的格式"""
        if not self.move_rules:
            return None
        
        # 转换为(keyword, target_dir)元组列表
        converted_rules = []
        for rule in self.move_rules:
            keyword = rule["keyword"]
            target_dir = rule["target_dir"]
            
            # 处理相对路径
            if not os.path.isabs(target_dir):
                target_dir = os.path.join(self.input_dir, target_dir)
            
            converted_rules.append((keyword, target_dir))
        
        return converted_rules
    
    def stop(self):
        """停止线程"""
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()


class BatchProcessingTab(QWidget):
    """批量处理标签页"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.move_rules = []  # 存储移动规则
        self.file_queue = queue.Queue()  # 文件队列
        self.watcher_thread = None  # 文件监听线程
        self.worker_thread = None  # 批量处理线程
        self.is_monitoring = False  # 是否正在监听
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 输入目录
        input_group = QGroupBox("输入目录")
        input_layout = QHBoxLayout()
        input_group.setLayout(input_layout)
        
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("选择包含图片的目录")
        input_layout.addWidget(self.input_dir_edit)
        
        self.input_browse_btn = QPushButton("浏览...")
        self.input_browse_btn.clicked.connect(self.browse_input_dir)
        input_layout.addWidget(self.input_browse_btn)
        
        layout.addWidget(input_group)
        
        # 输出目录
        output_group = QGroupBox("默认输出目录")
        output_layout = QHBoxLayout()
        output_group.setLayout(output_layout)
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("未匹配规则的图片将移动到此目录（默认为input/output）")
        output_layout.addWidget(self.output_dir_edit)
        
        self.output_browse_btn = QPushButton("浏览...")
        self.output_browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.output_browse_btn)
        
        layout.addWidget(output_group)
        
        # 处理选项
        options_group = QGroupBox("处理选项")
        options_layout = QGridLayout()
        options_group.setLayout(options_layout)
        
        # 瘦脸强度
        options_layout.addWidget(QLabel("瘦脸强度:"), 0, 0)
        self.slim_slider = QSlider(Qt.Horizontal)
        self.slim_slider.setRange(0, 100)
        self.slim_slider.setValue(50)
        options_layout.addWidget(self.slim_slider, 0, 1)
        
        self.slim_value_label = QLabel("0.5")
        options_layout.addWidget(self.slim_value_label, 0, 2)
        
        self.slim_slider.valueChanged.connect(self.update_slim_value)
        
        # 功能选择复选框
        self.use_face_slim_check = QCheckBox("启用瘦脸功能")
        self.use_face_slim_check.setChecked(False)
        options_layout.addWidget(self.use_face_slim_check, 1, 0, 1, 3)
        
        self.use_skin_retouch_check = QCheckBox("启用皮肤美化功能")
        self.use_skin_retouch_check.setChecked(True)
        options_layout.addWidget(self.use_skin_retouch_check, 2, 0, 1, 3)
        
        self.use_portrait_enhancement_check = QCheckBox("启用人像增强功能")
        self.use_portrait_enhancement_check.setChecked(False)
        options_layout.addWidget(self.use_portrait_enhancement_check, 3, 0, 1, 3)
        
        # 递归处理子目录
        self.recursive_check = QCheckBox("递归处理子目录")
        options_layout.addWidget(self.recursive_check, 4, 0, 1, 3)
        
        # 跳过已存在的结果
        self.skip_check = QCheckBox("跳过已处理的图片")
        self.skip_check.setChecked(True)
        options_layout.addWidget(self.skip_check, 5, 0, 1, 3)
        
        layout.addWidget(options_group)
        
        # 图片移动规则
        rules_group = QGroupBox("图片移动规则")
        rules_layout = QVBoxLayout()
        rules_group.setLayout(rules_layout)
        
        # 规则说明
        rules_desc = QLabel("根据图片名称中的特定字符，将处理后的图片移动到指定文件夹")
        rules_desc.setStyleSheet("color: #666; font-size: 11px;")
        rules_layout.addWidget(rules_desc)
        
        # 规则列表
        self.rules_list = QListWidget()
        self.rules_list.setMinimumHeight(100)
        rules_layout.addWidget(self.rules_list)
        
        # 规则操作按钮
        rules_btn_layout = QHBoxLayout()
        
        self.add_rule_btn = QPushButton("添加规则")
        self.add_rule_btn.clicked.connect(self.add_rule)
        rules_btn_layout.addWidget(self.add_rule_btn)
        
        self.edit_rule_btn = QPushButton("编辑规则")
        self.edit_rule_btn.clicked.connect(self.edit_rule)
        rules_btn_layout.addWidget(self.edit_rule_btn)
        
        self.remove_rule_btn = QPushButton("删除规则")
        self.remove_rule_btn.clicked.connect(self.remove_rule)
        rules_btn_layout.addWidget(self.remove_rule_btn)
        
        rules_layout.addLayout(rules_btn_layout)
        
        layout.addWidget(rules_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # 开始处理按钮
        self.start_btn = QPushButton("开始批量处理")
        self.start_btn.clicked.connect(self.start_batch_processing)
        layout.addWidget(self.start_btn)
        
        # 状态标签
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        
        # 添加弹性空间
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
    
    def browse_input_dir(self):
        """浏览输入目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if dir_path:
            self.input_dir_edit.setText(dir_path)
    
    def browse_output_dir(self):
        """浏览默认输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择默认输出目录")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def update_slim_value(self):
        """更新瘦脸强度值显示"""
        value = self.slim_slider.value() / 100.0
        self.slim_value_label.setText(f"{value:.1f}")
    
    def add_rule(self):
        """添加图片移动规则"""
        dialog = RuleDiaglog(self)
        if dialog.exec_() == QDialog.Accepted:
            keyword = dialog.keyword_edit.text().strip()
            target_dir = dialog.target_dir_edit.text().strip()
            
            if keyword and target_dir:
                rule = {"keyword": keyword, "target_dir": target_dir}
                self.move_rules.append(rule)
                self.rules_list.addItem(f"{keyword} -> {target_dir}")
    
    def edit_rule(self):
        """编辑图片移动规则"""
        current_row = self.rules_list.currentRow()
        if current_row >= 0:
            rule = self.move_rules[current_row]
            dialog = RuleDiaglog(self)
            dialog.keyword_edit.setText(rule["keyword"])
            dialog.target_dir_edit.setText(rule["target_dir"])
            
            if dialog.exec_() == QDialog.Accepted:
                keyword = dialog.keyword_edit.text().strip()
                target_dir = dialog.target_dir_edit.text().strip()
                
                if keyword and target_dir:
                    rule["keyword"] = keyword
                    rule["target_dir"] = target_dir
                    self.rules_list.item(current_row).setText(f"{keyword} -> {target_dir}")
    
    def remove_rule(self):
        """删除图片移动规则"""
        current_row = self.rules_list.currentRow()
        if current_row >= 0:
            self.rules_list.takeItem(current_row)
            self.move_rules.pop(current_row)
    
    def start_batch_processing(self):
        """开始/停止批量处理"""
        # 如果正在监听，则停止监听
        if self.is_monitoring:
            self.stop_monitoring()
            return
            
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir:
            QMessageBox.warning(self, "警告", "请选择输入目录")
            return
        
        if not os.path.exists(input_dir):
            QMessageBox.warning(self, "警告", f"输入目录不存在: {input_dir}")
            return
        
        # 获取默认输出目录
        default_output_dir = self.output_dir_edit.text().strip()
        if not default_output_dir:
            # 如果用户未设置，使用input/output作为默认
            default_output_dir = os.path.join(input_dir, "output")
        
        # 确保默认输出目录存在
        os.makedirs(default_output_dir, exist_ok=True)
        
        # 获取处理选项
        slim_intensity = self.slim_slider.value() / 100.0
        use_face_slim = self.use_face_slim_check.isChecked()
        use_skin_retouch = self.use_skin_retouch_check.isChecked()
        use_portrait_enhancement = self.use_portrait_enhancement_check.isChecked()
        
        # 检查至少选择了一个功能
        if not (use_face_slim or use_skin_retouch or use_portrait_enhancement):
            QMessageBox.warning(self, "警告", "请至少选择一个处理功能（瘦脸、皮肤美化或人像增强）")
            return
        
        # 切换到监听模式
        self.is_monitoring = True
        self.start_btn.setText("结束监听")
        self.progress_bar.setValue(0)
        self.status_label.setText("正在初始化...")
        
        # 禁用选项
        self.input_dir_edit.setEnabled(False)
        self.input_browse_btn.setEnabled(False)
        self.output_dir_edit.setEnabled(False)
        self.output_browse_btn.setEnabled(False)
        self.slim_slider.setEnabled(False)
        self.use_face_slim_check.setEnabled(False)
        self.use_skin_retouch_check.setEnabled(False)
        self.use_portrait_enhancement_check.setEnabled(False)
        self.recursive_check.setEnabled(False)
        self.skip_check.setEnabled(False)
        self.add_rule_btn.setEnabled(False)
        self.edit_rule_btn.setEnabled(False)
        self.remove_rule_btn.setEnabled(False)
        
        try:
            # 创建并启动文件监听线程
            self.file_queue = queue.Queue()
            self.watcher_thread = FileWatcherThread(input_dir, self.file_queue)
            self.watcher_thread.file_added.connect(self.on_file_added)
            self.watcher_thread.start()
            
            # 创建并启动批量处理线程
            self.worker_thread = BatchProcessingWorker(
                input_dir,
                slim_intensity,
                self.recursive_check.isChecked(),
                self.skip_check.isChecked(),
                self.move_rules,
                self.file_queue,
                max_size=1920,
                default_output_dir=default_output_dir,  # 传递默认输出目录
                use_face_slim=use_face_slim,
                use_skin_retouch=use_skin_retouch,
                use_portrait_enhancement=use_portrait_enhancement
            )
            self.worker_thread.progress.connect(self.progress_bar.setValue)
            self.worker_thread.status.connect(self.status_label.setText)
            self.worker_thread.finished.connect(self.on_processing_finished)
            self.worker_thread.start()
            
        except Exception as e:
            logging.error(f"启动监听失败: {e}")
            logging.error(traceback.format_exc())
            self.status_label.setText(f"启动监听失败: {e}")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """停止监听"""
        if self.watcher_thread and self.watcher_thread.isRunning():
            self.watcher_thread.stop()
            self.watcher_thread.wait()
            self.watcher_thread = None
        
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()
            self.worker_thread = None
        
        # 恢复UI状态
        self.is_monitoring = False
        self.start_btn.setText("开始批量处理")
        self.input_dir_edit.setEnabled(True)
        self.input_browse_btn.setEnabled(True)
        self.output_dir_edit.setEnabled(True)
        self.output_browse_btn.setEnabled(True)
        self.slim_slider.setEnabled(True)
        self.use_face_slim_check.setEnabled(True)
        self.use_skin_retouch_check.setEnabled(True)
        self.use_portrait_enhancement_check.setEnabled(True)
        self.recursive_check.setEnabled(True)
        self.skip_check.setEnabled(True)
        self.add_rule_btn.setEnabled(True)
        self.edit_rule_btn.setEnabled(True)
        self.remove_rule_btn.setEnabled(True)
    
    def on_file_added(self, file_path):
        """当有新文件添加时的回调"""
        logging.info(f"检测到新图片: {file_path}")
    
    def on_processing_finished(self, processed_count):
        """当处理完成时的回调"""
        if not self.is_monitoring:
            self.status_label.setText(f"处理完成，成功处理 {processed_count} 张图片")
            QMessageBox.information(self, "完成", f"批量处理完成，成功处理 {processed_count} 张图片")


class RuleDiaglog(QDialog):
    """图片移动规则对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图片移动规则")
        self.setMinimumWidth(400)
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 关键字
        keyword_layout = QHBoxLayout()
        keyword_layout.addWidget(QLabel("图片名称包含:"))
        self.keyword_edit = QLineEdit()
        keyword_layout.addWidget(self.keyword_edit)
        layout.addLayout(keyword_layout)
        
        # 目标目录
        target_dir_layout = QHBoxLayout()
        target_dir_layout.addWidget(QLabel("移动到目录:"))
        self.target_dir_edit = QLineEdit()
        target_dir_layout.addWidget(self.target_dir_edit)
        
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_target_dir)
        target_dir_layout.addWidget(self.browse_btn)
        
        layout.addLayout(target_dir_layout)
        
        # 说明
        desc = QLabel("注意: 目标目录可以是相对路径或绝对路径。如果是相对路径，将基于输入目录创建。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def browse_target_dir(self):
        """浏览目标目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择目标目录")
        if dir_path:
            self.target_dir_edit.setText(dir_path)


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸批量处理工具")
        self.setMinimumSize(800, 600)
        
        # 根据操作系统调整窗口样式
        if SYSTEM == "Darwin":  # macOS
            self.setUnifiedTitleAndToolBarOnMac(True)
            # macOS下可能需要额外的样式调整
            self.setStyleSheet("""
                QMainWindow { background-color: #f5f5f7; }
                QTabWidget::pane { border: 1px solid #d1d1d6; }
                QTabBar::tab { background-color: #e5e5ea; padding: 8px 16px; }
                QTabBar::tab:selected { background-color: #ffffff; }
            """)
        elif SYSTEM == "Windows":
            # Windows下的样式调整
            self.setStyleSheet("""
                QMainWindow { background-color: #f0f0f0; }
                QTabWidget::pane { border: 1px solid #cccccc; }
                QTabBar::tab { background-color: #e0e0e0; padding: 6px 12px; }
                QTabBar::tab:selected { background-color: #ffffff; }
            """)
        
        # 创建标签页
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # 创建批处理标签页
        self.batch_tab = BatchProcessingTab(self)
        self.tabs.addTab(self.batch_tab, "批量处理")


def main():
    """启动GUI应用"""
    app = QApplication(sys.argv)
    
    # 检查皮肤美化模型是否存在，如不存在则下载
    try:
        # 创建临时SkinRetoucher实例检查模型是否存在
        temp_enhancer = FaceEnhancer.get_instance(offline_mode=True, use_cache=True)
        model_exists = temp_enhancer.skin_retoucher.model_dir is not None
        
        if not model_exists:
            logging.info("本地皮肤美化模型不存在，准备下载...")
            # 显示下载提示对话框
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("模型下载")
            msg_box.setText("本地皮肤美化模型不存在，需要下载模型才能继续。\n下载可能需要几分钟时间，请耐心等待。")
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            
            if msg_box.exec_() == QMessageBox.Ok:
                # 用户确认下载
                logging.info("开始下载皮肤美化模型...")
                model_path = download_skin_retouching_model()
                
                if model_path:
                    logging.info(f"模型下载成功，保存在: {model_path}")
                    QMessageBox.information(None, "下载成功", f"皮肤美化模型下载成功，保存在: {model_path}")
                else:
                    logging.error("模型下载失败")
                    QMessageBox.critical(None, "下载失败", "皮肤美化模型下载失败，程序可能无法正常工作。")
            else:
                # 用户取消下载
                logging.warning("用户取消下载模型")
                QMessageBox.warning(None, "警告", "未下载模型，皮肤美化功能可能无法正常工作。")
    except Exception as e:
        logging.error(f"检查模型时出错: {e}")
        logging.error(traceback.format_exc())
    
    # 根据操作系统设置应用样式
    if SYSTEM == "Darwin":  # macOS
        app.setStyle("Fusion")  # macOS下使用Fusion样式可能更好
    
    # 设置应用图标
    try:
        app.setWindowIcon(QIcon("icon.png"))
    except:
        pass  # 如果图标不存在，忽略错误
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 