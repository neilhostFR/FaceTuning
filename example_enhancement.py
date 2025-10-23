import os
import sys
import cv2
import matplotlib.pyplot as plt
from face_enhancement import FaceEnhancer

def main():
    """
    人脸处理示例脚本
    
    处理流程：
    1. 人脸检测与裁剪
    2. 智能瘦脸
    3. 还原到原图
    4. 皮肤美化
    """
    print("人脸处理示例")
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 默认使用input.jpg
        image_path = "input.jpg"
        if not os.path.exists(image_path):
            print(f"默认图片不存在: {image_path}")
            image_path = input("请输入图片路径: ")
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return
    
    # 设置瘦脸强度
    if len(sys.argv) > 2:
        try:
            intensity = float(sys.argv[2])
            intensity = max(0.0, min(1.0, intensity))  # 限制在0-1范围内
        except ValueError:
            print("无效的瘦脸强度，使用默认值: 0.5")
            intensity = 0.5
    else:
        intensity = 0.5
    
    # 检查是否覆盖原图
    overwrite = "--overwrite" in sys.argv or "-o" in sys.argv
    
    print(f"处理图片: {image_path}, 瘦脸强度: {intensity}, 覆盖原图: {overwrite}")
    
    # 创建人脸增强器
    enhancer = FaceEnhancer()
    
    # 进行人脸处理
    final_path = enhancer.enhance_face(
        image_path, 
        slim_intensity=intensity,
        overwrite_original=overwrite
    )
    
    if final_path:
        print(f"人脸处理成功，最终结果保存在: {final_path}")
        
        # 显示原图和最终处理后的图片对比
        if not overwrite:
            enhancer.display_results(image_path, final_path)
        else:
            print("已覆盖原图，不显示对比图")
    else:
        print("人脸处理失败")


if __name__ == "__main__":
    main() 