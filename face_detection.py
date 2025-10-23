import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import dlib
import traceback
import logging
import math
import platform
import sys

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("face_detection.log"),
                             logging.StreamHandler()])

class FaceDetector:
    def __init__(self, model_path=None):
        """
        初始化人脸检测器
        
        Args:
            model_path: YOLO模型路径，如果为None则自动下载
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            # 使用预训练的YOLOv8n-face模型
            self.model = YOLO("yolov8n-face.pt")
        
        # 加载dlib的人脸关键点检测器
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        # 如果关键点检测器文件不存在，提示用户下载
        if not os.path.exists(self.predictor_path):
            logging.warning(f"人脸关键点检测器模型不存在: {self.predictor_path}")
            print(f"请从以下地址下载人脸关键点检测器模型：")
            print(f"https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2")
            print(f"下载后解压并放置在当前目录下")
            
            # 提供针对不同操作系统的下载命令提示
            system = platform.system()
            if system == "Darwin":  # macOS
                print("\nmacOS系统下载命令:")
                print("curl -L \"https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2\" -o shape_predictor_68_face_landmarks.dat.bz2")
                print("bunzip2 shape_predictor_68_face_landmarks.dat.bz2")
            elif system == "Windows":
                print("\nWindows系统下载命令:")
                print("curl -L \"https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2\" -o shape_predictor_68_face_landmarks.dat.bz2")
                print("或者使用浏览器下载后，使用7-Zip等工具解压")
            else:  # Linux等其他系统
                print("\nLinux系统下载命令:")
                print("curl -L \"https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2\" -o shape_predictor_68_face_landmarks.dat.bz2")
                print("bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
        
        try:
            # 检测系统类型，针对macOS进行特殊处理
            self.system = platform.system()
            logging.info(f"当前操作系统: {self.system}")
            
            # 初始化dlib检测器
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.predictor_path)
            self.has_dlib = True
            logging.info("dlib初始化成功")
        except Exception as e:
            logging.error(f"dlib初始化失败: {e}")
            print(f"警告: dlib初始化失败，瘦脸功能将不可用: {e}")
            self.has_dlib = False
            
            # 针对不同系统提供安装dlib的建议
            if self.system == "Darwin":  # macOS
                print("\nmacOS系统安装dlib建议:")
                print("1. 确保已安装Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
                print("2. 安装依赖: brew install cmake")
                print("3. 安装dlib: pip install dlib")
            elif self.system == "Windows":
                print("\nWindows系统安装dlib建议:")
                print("1. 安装Visual Studio Build Tools")
                print("2. 安装CMake")
                print("3. 安装dlib: pip install dlib")
            else:
                print("\nLinux系统安装dlib建议:")
                print("1. 安装依赖: sudo apt-get install build-essential cmake")
                print("2. 安装dlib: pip install dlib")
    
    def detect_faces(self, image_path):
        """
        检测图片中的人脸
        
        Args:
            image_path: 图片路径
            
        Returns:
            results: 检测结果
        """
        results = self.model(image_path)
        return results
    
    def crop_face(self, image_path, output_dir="cropped_faces", size=(512, 512)):
        """
        检测并裁剪人脸
        
        Args:
            image_path: 图片路径
            output_dir: 输出目录
            size: 裁剪后的图片大小
            
        Returns:
            crop_info_list: 裁剪信息列表，包含裁剪位置等参数
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # BGR转RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测人脸
        results = self.detect_faces(image_path)
        
        # 获取文件名（不含扩展名）
        base_name = os.path.basename(image_path)
        file_name = os.path.splitext(base_name)[0]
        
        crop_info_list = []
        
        # 处理每个检测到的人脸
        for i, box in enumerate(results[0].boxes):
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 计算人脸中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 计算边界框的宽度和高度
            width = x2 - x1
            height = y2 - y1
            
            # 取较大的尺寸作为正方形边长，并增加一些边距
            side_length = max(width, height)
            side_length = int(side_length * 1.7)  # 增加70%的边距，原来是2.0（增加100%）
            
            # 计算正方形的左上角和右下角坐标
            square_x1 = max(0, center_x - side_length // 2)
            square_y1 = max(0, center_y - side_length // 2)
            square_x2 = min(image.shape[1], center_x + side_length // 2)
            square_y2 = min(image.shape[0], center_y + side_length // 2)
            
            # 裁剪人脸区域
            face_img = image_rgb[square_y1:square_y2, square_x1:square_x2]
            
            # 转换为PIL图像并调整大小
            pil_img = Image.fromarray(face_img)
            pil_img = pil_img.resize(size, Image.LANCZOS)
            
            # 保存裁剪后的图片
            output_path = os.path.join(output_dir, f"{file_name}_face_{i}.jpg")
            pil_img.save(output_path)
            
            # 记录裁剪信息
            crop_info = {
                "original_box": (x1, y1, x2, y2),
                "crop_box": (square_x1, square_y1, square_x2, square_y2),
                "confidence": float(box.conf[0]),
                "output_path": output_path
            }
            crop_info_list.append(crop_info)
            
            print(f"人脸 {i+1} 已保存到 {output_path}")
            print(f"裁剪信息: {crop_info}")
        
        return crop_info_list
    
    def detect_landmarks(self, img):
        """
        检测人脸关键点
        
        Args:
            img: 输入图像
            
        Returns:
            landmarks_list: 关键点列表
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks_list = []
        
        # 检测人脸
        rects = self.detector(img_gray, 0)
        
        for i in range(len(rects)):
            # 检测关键点
            shape = self.predictor(img_gray, rects[i])
            landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
            landmarks_list.append(landmarks)
        
        return landmarks_list
    
    # 双线性插值法 - 从Face-DetectAndThin.py直接复制
    def BilinearInsert(self, src, ux, uy):
        w, h, c = src.shape
        if c == 3:
            x1 = int(ux)
            x2 = x1 + 1
            y1 = int(uy)
            y2 = y1 + 1
            
            # 确保坐标在图像范围内
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 >= w: x2 = w - 1
            if y2 >= h: y2 = h - 1

            part1 = src[y1, x1].astype(np.float32) * (float(x2) - ux) * (float(y2) - uy)
            part2 = src[y1, x2].astype(np.float32) * (ux - float(x1)) * (float(y2) - uy)
            part3 = src[y2, x1].astype(np.float32) * (float(x2) - ux) * (uy - float(y1))
            part4 = src[y2, x2].astype(np.float32) * (ux - float(x1)) * (uy - float(y1))

            insertValue = part1 + part2 + part3 + part4

            return insertValue.astype(np.uint8)
        return None
    
    # 局部平移变形 - 从Face-DetectAndThin.py直接复制
    def localTranslationWarp(self, srcImg, startX, startY, endX, endY, radius):
        ddradius = float(radius * radius)
        copyImg = np.zeros(srcImg.shape, np.uint8)
        copyImg = srcImg.copy()

        # 计算公式中的|m-c|^2
        ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
        H, W, C = srcImg.shape
        for i in range(W):
            for j in range(H):
                # 计算该点是否在形变圆的范围之内
                # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中
                if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                    continue

                distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)

                if (distance < ddradius):
                    # 计算出（i,j）坐标的原坐标
                    # 计算公式中右边平方号里的部分
                    ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                    ratio = ratio * ratio

                    # 映射原位置
                    UX = i - ratio * (endX - startX)
                    UY = j - ratio * (endY - startY)

                    # 根据双线性插值法得到UX，UY的值
                    value = self.BilinearInsert(srcImg, UX, UY)
                    # 改变当前 i ，j的值
                    if value is not None:
                        copyImg[j, i] = value

        return copyImg
    
    def slim_face(self, image_path, intensity=0.5, output_dir="slimmed_faces"):
        """
        对图片中的人脸进行瘦脸处理
        
        Args:
            image_path: 图片路径
            intensity: 瘦脸强度，范围0-1
            output_dir: 输出目录
            
        Returns:
            output_path: 处理后的图片路径
        """
        logging.info(f"开始瘦脸处理: {image_path}, 强度: {intensity}")
        
        if not self.has_dlib:
            logging.error("瘦脸功能需要dlib库和人脸关键点检测器模型")
            raise ValueError("瘦脸功能需要dlib库和人脸关键点检测器模型")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"创建输出目录: {output_dir}")
        
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"无法读取图片: {image_path}")
                raise ValueError(f"无法读取图片: {image_path}")
            
            logging.info(f"图片尺寸: {image.shape}")
            
            # 检测人脸关键点
            landmarks_list = self.detect_landmarks(image)
            logging.info(f"检测到 {len(landmarks_list)} 个人脸")
            
            if len(landmarks_list) == 0:
                logging.warning(f"未检测到人脸: {image_path}")
                print(f"未检测到人脸: {image_path}")
                return None
            
            # 获取文件名（不含扩展名）
            base_name = os.path.basename(image_path)
            file_name = os.path.splitext(base_name)[0]
            
            # 创建输出图像
            result_img = image.copy()
            
            # 处理每个检测到的人脸
            for i, landmarks_node in enumerate(landmarks_list):
                logging.info(f"处理人脸 {i+1}")
                
                try:
                    # 获取关键点 - 从Face-DetectAndThin.py直接复制逻辑
                    
                    left_landmark = landmarks_node[3]  # 左脸颊关键点（第4个点）
                    left_landmark_down = landmarks_node[5]  # 左脸颊下方关键点（第6个点）
                    
                    right_landmark = landmarks_node[13]  # 右脸颊关键点（第14个点）
                    right_landmark_down = landmarks_node[15]  # 右脸颊下方关键点（第16个点）
                    
                    endPt = landmarks_node[30]  # 鼻尖关键点（第31个点）
                    
                    logging.info(f"左脸颊坐标: ({left_landmark[0, 0]}, {left_landmark[0, 1]})")
                    logging.info(f"右脸颊坐标: ({right_landmark[0, 0]}, {right_landmark[0, 1]})")
                    logging.info(f"鼻尖坐标: ({endPt[0, 0]}, {endPt[0, 1]})")
                    
                    # 计算第4个点到第6个点的距离作为瘦脸距离
                    r_left = math.sqrt(
                        (left_landmark[0, 0] - left_landmark_down[0, 0]) * (left_landmark[0, 0] - left_landmark_down[0, 0]) +
                        (left_landmark[0, 1] - left_landmark_down[0, 1]) * (left_landmark[0, 1] - left_landmark_down[0, 1]))
                    
                    # 计算第14个点到第16个点的距离作为瘦脸距离
                    r_right = math.sqrt(
                        (right_landmark[0, 0] - right_landmark_down[0, 0]) * (right_landmark[0, 0] - right_landmark_down[0, 0]) +
                        (right_landmark[0, 1] - right_landmark_down[0, 1]) * (right_landmark[0, 1] - right_landmark_down[0, 1]))
                    
                    # 根据强度调整半径
                    r_left = r_left * intensity * 2.0  # 调整强度
                    r_right = r_right * intensity * 2.0
                    
                    logging.info(f"左脸颊变形半径: {r_left}")
                    logging.info(f"右脸颊变形半径: {r_right}")
                    
                    # 瘦左边脸
                    logging.info("开始瘦左边脸...")
                    result_img = self.localTranslationWarp(
                        result_img,
                        left_landmark[0, 0], left_landmark[0, 1],
                        endPt[0, 0], endPt[0, 1],
                        r_left
                    )
                    
                    # 瘦右边脸
                    logging.info("开始瘦右边脸...")
                    result_img = self.localTranslationWarp(
                        result_img,
                        right_landmark[0, 0], right_landmark[0, 1],
                        endPt[0, 0], endPt[0, 1],
                        r_right
                    )
                    
                    logging.info("瘦脸变形完成")
                    
                except Exception as e:
                    logging.error(f"处理人脸 {i+1} 时出错: {e}")
                    logging.error(traceback.format_exc())
                    continue
            
            # 保存结果
            output_path = os.path.join(output_dir, f"{file_name}_slimmed.jpg")
            logging.info(f"保存瘦脸结果到: {output_path}")
            
            try:
                cv2.imwrite(output_path, result_img)
                logging.info(f"瘦脸处理后的图片已保存到 {output_path}")
                print(f"瘦脸处理后的图片已保存到 {output_path}")
                return output_path
            except Exception as e:
                logging.error(f"保存图片时出错: {e}")
                logging.error(traceback.format_exc())
                return None
            
        except Exception as e:
            logging.error(f"瘦脸处理失败: {e}")
            logging.error(traceback.format_exc())
            print(f"瘦脸处理失败: {e}")
            return None
    
    def crop_and_slim_face(self, image_path, output_dir="cropped_faces", slim_output_dir="slimmed_faces", 
                          size=(512, 512), slim_intensity=0.5):
        """
        检测、裁剪并瘦脸处理
        
        Args:
            image_path: 图片路径
            output_dir: 裁剪输出目录
            slim_output_dir: 瘦脸输出目录
            size: 裁剪后的图片大小
            slim_intensity: 瘦脸强度
            
        Returns:
            result_list: 结果列表，包含裁剪和瘦脸信息
        """
        logging.info(f"开始裁剪并瘦脸处理: {image_path}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(slim_output_dir, exist_ok=True)
        
        # 裁剪人脸
        crop_info_list = self.crop_face(image_path, output_dir, size)
        
        if not crop_info_list:
            logging.warning(f"未检测到人脸: {image_path}")
            print(f"未检测到人脸: {image_path}")
            return []
        
        result_list = []
        
        # 对每个裁剪的人脸进行瘦脸处理
        for i, crop_info in enumerate(crop_info_list):
            cropped_path = crop_info["output_path"]
            logging.info(f"处理裁剪的人脸 {i+1}: {cropped_path}")
            
            try:
                if self.has_dlib:
                    # 瘦脸处理
                    slimmed_path = self.slim_face(cropped_path, slim_intensity, slim_output_dir)
                    
                    # 更新结果信息
                    result = crop_info.copy()
                    if slimmed_path:
                        result["slimmed_path"] = slimmed_path
                        logging.info(f"瘦脸成功: {slimmed_path}")
                    else:
                        logging.warning(f"瘦脸失败，使用原始裁剪图片")
                    result_list.append(result)
                else:
                    logging.warning(f"dlib未初始化，跳过瘦脸处理")
                    print(f"警告: dlib未初始化，跳过瘦脸处理")
                    result_list.append(crop_info)
            except Exception as e:
                logging.error(f"瘦脸处理失败: {e}")
                logging.error(traceback.format_exc())
                print(f"瘦脸处理失败: {e}")
                result_list.append(crop_info)
        
        return result_list
    
    def restore_face_to_original(self, original_image_path, cropped_face_path, crop_info):
        """
        将处理后的人脸还原到原图中
        
        Args:
            original_image_path: 原图路径
            cropped_face_path: 处理后的人脸图片路径
            crop_info: 裁剪信息
            
        Returns:
            还原后的图片（RGB格式）
        """
        try:
            # 读取原图
            original_image = cv2.imread(original_image_path)
            if original_image is None:
                logging.error(f"无法读取原图: {original_image_path}")
                return None
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # 读取处理后的人脸图片
            face_image = cv2.imread(cropped_face_path)
            if face_image is None:
                logging.error(f"无法读取人脸图片: {cropped_face_path}")
                return None
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # 获取裁剪信息
            square_x1, square_y1, square_x2, square_y2 = crop_info["crop_box"]
            
            # 确保坐标在图像范围内
            square_x1 = max(0, square_x1)
            square_y1 = max(0, square_y1)
            square_x2 = min(original_image.shape[1], square_x2)
            square_y2 = min(original_image.shape[0], square_y2)
            
            # 获取裁剪区域的尺寸
            crop_width = square_x2 - square_x1
            crop_height = square_y2 - square_y1
            
            # 调整处理后的人脸大小以匹配原图中的人脸区域
            face_image_resized = cv2.resize(face_image, (crop_width, crop_height))
            
            # 获取原图中的人脸区域
            original_face = original_image[square_y1:square_y2, square_x1:square_x2]
            
            # 增强色彩校正
            # 计算原始人脸区域的均值和标准差
            original_mean = np.mean(original_face, axis=(0, 1))
            original_std = np.std(original_face, axis=(0, 1))
            
            # 计算处理后人脸的均值和标准差
            face_mean = np.mean(face_image_resized, axis=(0, 1))
            face_std = np.std(face_image_resized, axis=(0, 1))
            
            # 应用色彩校正，保持细节的同时调整整体色调
            corrected_face = ((face_image_resized - face_mean) / (face_std + 1e-6)) * (original_std + 1e-6) + original_mean
            corrected_face = np.clip(corrected_face, 0, 255).astype(np.uint8)
            
            # 创建低分辨率的颜色映射，捕捉大尺度的颜色变化
            scale_factor = 0.1
            small_original = cv2.resize(original_face, (0, 0), fx=scale_factor, fy=scale_factor)
            small_face = cv2.resize(face_image_resized, (0, 0), fx=scale_factor, fy=scale_factor)
            small_diff = small_original - small_face
            color_diff = cv2.resize(small_diff, (crop_width, crop_height))
            
            # 平滑颜色差异
            color_diff = cv2.GaussianBlur(color_diff, (15, 15), 5)
            
            # 应用颜色校正
            corrected_face = corrected_face + color_diff * 0.5
            corrected_face = np.clip(corrected_face, 0, 255).astype(np.uint8)
            
            # 创建改进的掩码，使用椭圆形状更好地匹配人脸
            mask = np.zeros((crop_height, crop_width), dtype=np.float32)
            center = (crop_width // 2, crop_height // 2)
            axes = (int(crop_width * 0.5), int(crop_height * 0.5))  # 扩大椭圆以覆盖更多面部区域
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
            
            # 平滑掩码边缘
            mask = cv2.GaussianBlur(mask, (51, 51), 15)  # 增加模糊半径，使过渡更平滑
            
            # 扩展掩码为3通道
            mask_3channel = np.stack([mask] * 3, axis=2)
            
            # 边缘感知融合，保留原图的边缘特征
            edge_weight = 0.15  # 减少边缘权重，使处理后的人脸特征更明显
            edge_mask = np.ones_like(mask_3channel) * edge_weight
            center_mask = np.ones_like(mask_3channel) * 0.95  # 中心区域几乎完全使用处理后的人脸
            
            # 创建最终掩码，中心使用处理后的人脸，边缘平滑过渡到原图
            final_mask = mask_3channel * center_mask + (1 - mask_3channel) * edge_mask
            
            # 应用掩码进行融合
            blended_face = corrected_face * final_mask + original_face * (1 - final_mask)
            blended_face = np.clip(blended_face, 0, 255).astype(np.uint8)
            
            # 将融合后的人脸放回原图
            result = original_image.copy()
            result[square_y1:square_y2, square_x1:square_x2] = blended_face
            
            return result
            
        except Exception as e:
            logging.error(f"还原人脸到原图时出错: {e}")
            logging.error(traceback.format_exc())
            return None
    
    def restore_all_faces_to_original(self, original_image_path, result_list, use_slimmed=True, use_enhanced=False):
        """
        将所有处理后的人脸一次性还原到原图中
        
        Args:
            original_image_path: 原图路径
            result_list: 裁剪和处理结果列表
            use_slimmed: 是否使用瘦脸后的图片
            use_enhanced: 是否使用增强后的图片
            
        Returns:
            result_image: 还原后的图片
        """
        try:
            # 读取原图
            original_image = cv2.imread(original_image_path)
            if original_image is None:
                logging.error(f"无法读取原图: {original_image_path}")
                return None
            
            # 转换为RGB
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # 创建结果图像
            result_image = original_image.copy()
            
            # 处理每个人脸
            for i, result in enumerate(result_list):
                try:
                    # 获取裁剪信息
                    square_x1, square_y1, square_x2, square_y2 = result["crop_box"]
                    
                    # 确定使用哪个处理后的图片
                    if use_enhanced and "enhanced_path" in result:
                        face_path = result["enhanced_path"]
                        logging.info(f"使用增强后的图片: {face_path}")
                    elif use_slimmed and "slimmed_path" in result:
                        face_path = result["slimmed_path"]
                        logging.info(f"使用瘦脸后的图片: {face_path}")
                    else:
                        face_path = result["output_path"]
                        logging.info(f"使用裁剪后的图片: {face_path}")
                    
                    # 读取处理后的人脸图片
                    face_image = cv2.imread(face_path)
                    if face_image is None:
                        logging.error(f"无法读取处理后的人脸图片: {face_path}")
                        continue
                    
                    # 转换为RGB
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    
                    # 获取裁剪区域的尺寸
                    crop_width = square_x2 - square_x1
                    crop_height = square_y2 - square_y1
                    
                    # 调整人脸图像大小以匹配裁剪区域
                    face_image_resized = cv2.resize(face_image, (crop_width, crop_height))
                    
                    # 获取原图中对应的区域
                    original_face_region = original_image[square_y1:square_y2, square_x1:square_x2]
                    
                    # 1. 增强的颜色校正 - 使处理后的人脸图像与原图颜色风格更匹配
                    # 计算原图区域和处理后人脸图像的均值和标准差
                    orig_mean = np.mean(original_face_region, axis=(0, 1))
                    orig_std = np.std(original_face_region, axis=(0, 1))
                    face_mean = np.mean(face_image_resized, axis=(0, 1))
                    face_std = np.std(face_image_resized, axis=(0, 1))
                    
                    # 应用颜色校正
                    # 保持处理后人脸的细节，但调整整体色调以匹配原图
                    corrected_face = ((face_image_resized - face_mean) * (orig_std / face_std)) + orig_mean
                    
                    # 增强的颜色校正：使用局部颜色映射
                    # 创建一个低分辨率的颜色映射，用于捕获大尺度的颜色变化
                    scale_factor = 8  # 缩小因子
                    small_orig = cv2.resize(original_face_region, 
                                           (crop_width // scale_factor, crop_height // scale_factor))
                    small_face = cv2.resize(face_image_resized, 
                                           (crop_width // scale_factor, crop_height // scale_factor))
                    
                    # 计算低分辨率下的颜色差异
                    color_diff = small_orig.astype(np.float32) - small_face.astype(np.float32)
                    
                    # 将差异上采样回原始大小并进行平滑处理
                    color_diff = cv2.resize(color_diff, (crop_width, crop_height))
                    color_diff = cv2.GaussianBlur(color_diff, (15, 15), 0)
                    
                    # 将颜色差异的一部分添加到校正后的人脸上，以更好地匹配背景色调
                    # 使用较小的权重，以保持人脸细节
                    color_correction_weight = 0.3  # 调整这个值以控制背景匹配程度
                    corrected_face = corrected_face + color_diff * color_correction_weight
                    
                    # 确保值在有效范围内
                    corrected_face = np.clip(corrected_face, 0, 255).astype(np.uint8)
                    
                    # 2. 创建改进的掩码
                    # 使用椭圆形掩码，更符合人脸形状
                    mask = np.zeros((crop_height, crop_width), dtype=np.float32)
                    center = (crop_width // 2, crop_height // 2)
                    
                    # 使用更大的椭圆参数，覆盖更多的脸部区域，包括轮廓
                    # 水平轴和垂直轴都增大，以覆盖更多的脸部区域
                    axes = (int(crop_width * 0.55), int(crop_height * 0.65))  # 增大椭圆的大小
                    cv2.ellipse(mask, center, axes, 0, 0, 360, (1, 1, 1), -1)
                    
                    # 对掩码进行高斯模糊，创建平滑过渡区域，使用更大的模糊半径
                    # 这样可以创建更平滑的过渡区域，减少原始轮廓的可见性
                    blur_radius = min(crop_width, crop_height) // 8  # 增大模糊半径
                    blur_radius = max(15, blur_radius)  # 确保至少有15个像素的过渡区域
                    mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), 0)
                    
                    # 调整掩码的强度分布，使中心区域保持更高的权重
                    # 使用更小的幂值，创建更平缓的过渡
                    mask = np.power(mask, 0.5)  # 减小幂值，使过渡更平缓
                    
                    # 3. 边缘感知融合
                    # 计算原图区域的边缘
                    gray_orig = cv2.cvtColor(original_face_region, cv2.COLOR_RGB2GRAY)
                    edges_orig = cv2.Canny(gray_orig, 50, 150)
                    edges_orig = cv2.dilate(edges_orig, np.ones((3, 3), np.uint8), iterations=1)
                    
                    # 在边缘处减小掩码值，但减小的程度更小
                    edge_weight = edges_orig.astype(np.float32) / 255.0
                    edge_weight = cv2.GaussianBlur(edge_weight, (7, 7), 0)  # 增大模糊半径
                    mask = mask * (1 - edge_weight * 0.1)  # 边缘处减小10%的权重，原来是15%
                    
                    # 扩展掩码为3通道
                    mask_3channel = np.stack([mask, mask, mask], axis=2)
                    
                    # 使用掩码进行融合
                    face_region = result_image[square_y1:square_y2, square_x1:square_x2]
                    
                    # 使用校正后的人脸图像和改进的掩码进行融合
                    blended_region = corrected_face * mask_3channel + face_region * (1 - mask_3channel)
                    
                    # 更新结果图像
                    result_image[square_y1:square_y2, square_x1:square_x2] = blended_region
                    
                    logging.info(f"已将人脸 {i+1} 融合到原图")
                
                except Exception as e:
                    logging.error(f"处理人脸 {i+1} 时出错: {e}")
                    logging.error(traceback.format_exc())
                    continue
            
            return result_image
            
        except Exception as e:
            logging.error(f"还原所有人脸到原图失败: {e}")
            logging.error(traceback.format_exc())
            return None


def main():
    # 创建人脸检测器
    detector = FaceDetector()
    
    # 指定图片路径
    image_path = input("请输入图片路径: ")
    
    # 检测并裁剪人脸
    try:
        print("选择操作模式:")
        print("1. 仅裁剪人脸")
        print("2. 裁剪并瘦脸处理")
        mode = input("请选择 (1/2): ")
        
        if mode == "1":
            crop_info_list = detector.crop_face(image_path)
            
            if not crop_info_list:
                print("未检测到人脸")
            else:
                print(f"共检测到 {len(crop_info_list)} 个人脸")
                
                # 一次性将所有裁剪的人脸还原到原图
                print("\n是否要将所有裁剪的人脸还原到原图? (y/n)")
                choice = input().lower()
                
                if choice == 'y':
                    restored_image = detector.restore_all_faces_to_original(
                        image_path, 
                        crop_info_list,
                        use_slimmed=False
                    )
                    
                    # 保存还原后的图片
                    base_name = os.path.basename(image_path)
                    file_name = os.path.splitext(base_name)[0]
                    output_path = f"{file_name}_all_restored.jpg"
                    
                    cv2.imwrite(output_path, cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR))
                    print(f"所有人脸已还原到原图，保存在: {output_path}")
        
        elif mode == "2":
            # 设置瘦脸强度
            intensity = float(input("请输入瘦脸强度 (0.0-1.0): "))
            intensity = max(0.0, min(1.0, intensity))  # 限制在0-1范围内
            
            # 裁剪并瘦脸
            result_list = detector.crop_and_slim_face(image_path, slim_intensity=intensity)
            
            if not result_list:
                print("未检测到人脸")
            else:
                print(f"共处理 {len(result_list)} 个人脸")
                
                # 一次性将所有瘦脸后的人脸还原到原图
                print("\n是否要将所有瘦脸后的人脸还原到原图? (y/n)")
                choice = input().lower()
                
                if choice == 'y':
                    restored_image = detector.restore_all_faces_to_original(
                        image_path, 
                        result_list,
                        use_slimmed=True
                    )
                    
                    # 保存还原后的图片
                    base_name = os.path.basename(image_path)
                    file_name = os.path.splitext(base_name)[0]
                    output_path = f"{file_name}_all_restored_slimmed.jpg"
                    
                    cv2.imwrite(output_path, cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR))
                    print(f"所有瘦脸后的人脸已还原到原图，保存在: {output_path}")
        
        else:
            print("无效的选择")
    
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        logging.error(traceback.format_exc())
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main() 