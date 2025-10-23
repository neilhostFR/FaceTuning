import os
import sys
import cv2
import matplotlib.pyplot as plt
from face_detection import FaceDetector
from download_model import download_model

def test_slim_face(image_path, intensities=None):
    """测试瘦脸功能，可以测试多个强度"""
    if intensities is None:
        intensities = [0.5]  # 默认强度
    
    print(f"测试瘦脸功能，图片: {image_path}, 强度: {intensities}")
    
    # 确保模型文件存在
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("模型文件不存在，尝试下载...")
        if not download_model():
            print("模型下载失败，无法继续测试")
            return False
    
    # 创建人脸检测器
    detector = FaceDetector()
    
    if not detector.has_dlib:
        print("dlib初始化失败，无法使用瘦脸功能")
        return False
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return False
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return False
    
    # 创建输出目录
    output_dir = "test_slimmed_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理多个强度
    slimmed_paths = []
    for intensity in intensities:
        # 创建强度特定的输出目录
        intensity_dir = os.path.join(output_dir, f"intensity_{intensity:.1f}")
        os.makedirs(intensity_dir, exist_ok=True)
        
        # 对原图进行瘦脸处理
        slimmed_path = detector.slim_face(image_path, intensity, intensity_dir)
        
        if slimmed_path:
            print(f"瘦脸处理成功 (强度: {intensity})，结果保存在: {slimmed_path}")
            slimmed_paths.append((intensity, slimmed_path))
        else:
            print(f"瘦脸处理失败 (强度: {intensity})")
    
    if slimmed_paths:
        # 显示原图和不同强度瘦脸后的图片
        n_images = len(slimmed_paths) + 1  # 原图 + 瘦脸图片
        fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
        
        # 显示原图
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_img)
        axes[0].set_title("原图")
        axes[0].axis("off")
        
        # 显示瘦脸后的图片
        for i, (intensity, path) in enumerate(slimmed_paths):
            slimmed_img = cv2.imread(path)
            slimmed_img = cv2.cvtColor(slimmed_img, cv2.COLOR_BGR2RGB)
            axes[i+1].imshow(slimmed_img)
            axes[i+1].set_title(f"瘦脸 (强度: {intensity:.1f})")
            axes[i+1].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison.png"))
        plt.show()
        
        return True
    else:
        print("所有瘦脸处理都失败了")
        return False

def test_multiple_intensities(image_path):
    """测试多个瘦脸强度"""
    intensities = [0.2, 0.4, 0.6, 0.8, 1.0]
    return test_slim_face(image_path, intensities)

def test_with_landmarks(image_path, intensity=0.5):
    """测试瘦脸功能并显示人脸关键点"""
    print(f"测试瘦脸功能并显示人脸关键点，图片: {image_path}, 强度: {intensity}")
    
    # 确保模型文件存在
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("模型文件不存在，尝试下载...")
        if not download_model():
            print("模型下载失败，无法继续测试")
            return False
    
    # 创建人脸检测器
    detector = FaceDetector()
    
    if not detector.has_dlib:
        print("dlib初始化失败，无法使用瘦脸功能")
        return False
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return False
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return False
    
    # 创建输出目录
    output_dir = "test_slimmed_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    # 检测人脸关键点
    landmarks_list = detector.detect_landmarks(image)
    
    if not landmarks_list:
        print("未检测到人脸关键点")
        return False
    
    # 创建带关键点的原图
    image_with_landmarks = image.copy()
    for landmarks in landmarks_list:
        # 绘制所有关键点
        for i in range(landmarks.shape[0]):
            x, y = landmarks[i, 0], landmarks[i, 1]
            cv2.circle(image_with_landmarks, (x, y), 2, (0, 255, 0), -1)
        
        # 特别标记用于瘦脸的关键点
        # 左脸颊关键点（第4个点）
        left_landmark = landmarks[3]
        cv2.circle(image_with_landmarks, (left_landmark[0, 0], left_landmark[0, 1]), 5, (0, 0, 255), -1)
        cv2.putText(image_with_landmarks, "4", (left_landmark[0, 0]+5, left_landmark[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 左脸颊下方关键点（第6个点）
        left_landmark_down = landmarks[5]
        cv2.circle(image_with_landmarks, (left_landmark_down[0, 0], left_landmark_down[0, 1]), 5, (0, 0, 255), -1)
        cv2.putText(image_with_landmarks, "6", (left_landmark_down[0, 0]+5, left_landmark_down[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 右脸颊关键点（第14个点）
        right_landmark = landmarks[13]
        cv2.circle(image_with_landmarks, (right_landmark[0, 0], right_landmark[0, 1]), 5, (0, 0, 255), -1)
        cv2.putText(image_with_landmarks, "14", (right_landmark[0, 0]+5, right_landmark[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 右脸颊下方关键点（第16个点）
        right_landmark_down = landmarks[15]
        cv2.circle(image_with_landmarks, (right_landmark_down[0, 0], right_landmark_down[0, 1]), 5, (0, 0, 255), -1)
        cv2.putText(image_with_landmarks, "16", (right_landmark_down[0, 0]+5, right_landmark_down[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 鼻尖关键点（第31个点）
        nose_tip = landmarks[30]
        cv2.circle(image_with_landmarks, (nose_tip[0, 0], nose_tip[0, 1]), 5, (255, 0, 0), -1)
        cv2.putText(image_with_landmarks, "31", (nose_tip[0, 0]+5, nose_tip[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # 保存带关键点的原图
    landmarks_path = os.path.join(output_dir, "landmarks.jpg")
    cv2.imwrite(landmarks_path, image_with_landmarks)
    
    # 对原图进行瘦脸处理
    slimmed_path = detector.slim_face(image_path, intensity, output_dir)
    
    if slimmed_path:
        print(f"瘦脸处理成功，结果保存在: {slimmed_path}")
        
        # 显示原图、带关键点的原图和瘦脸后的图片
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示原图
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_img)
        axes[0].set_title("原图")
        axes[0].axis("off")
        
        # 显示带关键点的原图
        landmarks_img = cv2.imread(landmarks_path)
        landmarks_img = cv2.cvtColor(landmarks_img, cv2.COLOR_BGR2RGB)
        axes[1].imshow(landmarks_img)
        axes[1].set_title("人脸关键点")
        axes[1].axis("off")
        
        # 显示瘦脸后的图片
        slimmed_img = cv2.imread(slimmed_path)
        slimmed_img = cv2.cvtColor(slimmed_img, cv2.COLOR_BGR2RGB)
        axes[2].imshow(slimmed_img)
        axes[2].set_title(f"瘦脸 (强度: {intensity})")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "landmarks_comparison.png"))
        plt.show()
        
        return True
    else:
        print("瘦脸处理失败")
        return False

def test_before_after(image_path, intensity=0.5):
    """测试瘦脸前后对比，使用半透明叠加效果"""
    print(f"测试瘦脸前后对比，图片: {image_path}, 强度: {intensity}")
    
    # 确保模型文件存在
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("模型文件不存在，尝试下载...")
        if not download_model():
            print("模型下载失败，无法继续测试")
            return False
    
    # 创建人脸检测器
    detector = FaceDetector()
    
    if not detector.has_dlib:
        print("dlib初始化失败，无法使用瘦脸功能")
        return False
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return False
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return False
    
    # 创建输出目录
    output_dir = "test_slimmed_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    # 对原图进行瘦脸处理
    slimmed_path = detector.slim_face(image_path, intensity, output_dir)
    
    if slimmed_path:
        print(f"瘦脸处理成功，结果保存在: {slimmed_path}")
        
        # 读取原图和瘦脸后的图片
        original_img = cv2.imread(image_path)
        slimmed_img = cv2.imread(slimmed_path)
        
        # 创建半透明叠加效果
        overlay = original_img.copy()
        output = slimmed_img.copy()
        
        # 创建交替的条纹效果
        h, w = original_img.shape[:2]
        stripe_width = 20  # 条纹宽度
        
        for i in range(0, w, stripe_width * 2):
            output[:, i:i+stripe_width] = original_img[:, i:i+stripe_width]
        
        # 添加标签
        cv2.putText(output, "原图", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(output, "瘦脸后", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 保存对比图
        comparison_path = os.path.join(output_dir, "before_after_comparison.jpg")
        cv2.imwrite(comparison_path, output)
        
        # 显示对比图
        comparison_img = cv2.imread(comparison_path)
        comparison_img = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(comparison_img)
        plt.title(f"瘦脸前后对比 (强度: {intensity})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        return True
    else:
        print("瘦脸处理失败")
        return False

def main():
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python test_slim_face.py <图片路径> [瘦脸强度(0.0-1.0)/multi/landmarks/compare]")
        return
    
    # 获取图片路径
    image_path = sys.argv[1]
    
    # 检查测试模式
    if len(sys.argv) > 2:
        mode = sys.argv[2].lower()
        
        if mode == "multi":
            # 测试多个强度
            test_multiple_intensities(image_path)
        elif mode == "landmarks":
            # 测试并显示关键点
            intensity = 0.5
            if len(sys.argv) > 3:
                try:
                    intensity = float(sys.argv[3])
                    intensity = max(0.0, min(1.0, intensity))  # 限制在0-1范围内
                except ValueError:
                    print(f"无效的瘦脸强度: {sys.argv[3]}，使用默认值: 0.5")
            test_with_landmarks(image_path, intensity)
        elif mode == "compare":
            # 测试前后对比
            intensity = 0.5
            if len(sys.argv) > 3:
                try:
                    intensity = float(sys.argv[3])
                    intensity = max(0.0, min(1.0, intensity))  # 限制在0-1范围内
                except ValueError:
                    print(f"无效的瘦脸强度: {sys.argv[3]}，使用默认值: 0.5")
            test_before_after(image_path, intensity)
        else:
            # 单一强度测试
            try:
                intensity = float(mode)
                intensity = max(0.0, min(1.0, intensity))  # 限制在0-1范围内
                test_slim_face(image_path, [intensity])
            except ValueError:
                print(f"无效的参数: {mode}，使用默认值: 0.5")
                test_slim_face(image_path, [0.5])
    else:
        # 默认测试
        test_slim_face(image_path, [0.5])

if __name__ == "__main__":
    main() 