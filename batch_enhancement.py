import os
import sys
import cv2
import glob
import logging
import argparse
import traceback
import matplotlib.pyplot as plt
from face_enhancement import FaceEnhancer
import numpy as np
import platform
import time

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("batch_enhancement.log"),
                             logging.StreamHandler()])

def process_images(input_dir, output_dir=None, slim_intensity=0.5, 
                  file_pattern="*.*", recursive=False, 
                  create_comparison=False, skip_existing=True,
                  overwrite_original=False, move_rules=None):
    """
    批量处理图片
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录，如果为None且overwrite_original为False，则使用默认目录
        slim_intensity: 瘦脸强度，0-1之间
        file_pattern: 文件匹配模式，默认为"*.*"
        recursive: 是否递归处理子目录
        create_comparison: 是否创建对比图
        skip_existing: 是否跳过已存在的结果
        overwrite_original: 是否覆盖原图
        move_rules: 移动规则列表，每个规则是一个字典，包含keyword和target_dir
        
    Returns:
        处理成功的图片数量
    """
    # 获取当前操作系统
    system = platform.system()
    logging.info(f"当前操作系统: {system}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        logging.error(f"输入目录不存在: {input_dir}")
        return 0
    
    # 设置输出目录
    if output_dir is None and not overwrite_original:
        # 如果不覆盖原图且没有指定输出目录，使用默认目录
        final_dir = "beautified_faces"
        comparison_dir = "comparisons"
    elif output_dir is not None and not overwrite_original:
        # 如果不覆盖原图且指定了输出目录
        os.makedirs(output_dir, exist_ok=True)
        final_dir = os.path.join(output_dir, "beautified_faces")
        comparison_dir = os.path.join(output_dir, "comparisons")
    else:
        # 如果覆盖原图，不需要输出目录
        final_dir = None
        comparison_dir = None
    
    # 创建输出目录
    if final_dir:
        os.makedirs(final_dir, exist_ok=True)
    if comparison_dir and create_comparison:
        os.makedirs(comparison_dir, exist_ok=True)
    
    # 创建人脸增强器
    enhancer = FaceEnhancer()
    
    # 获取所有图片文件
    image_files = []
    
    # 支持的图片扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    if recursive:
        # 递归处理子目录
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file_pattern == "*.*":
                    # 检查文件扩展名
                    _, ext = os.path.splitext(file.lower())
                    if ext in image_extensions:
                        image_files.append(os.path.join(root, file))
                elif glob.fnmatch.fnmatch(file, file_pattern):
                    image_files.append(os.path.join(root, file))
    else:
        # 只处理当前目录
        for file in os.listdir(input_dir):
            if file_pattern == "*.*":
                # 检查文件扩展名
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    image_files.append(os.path.join(input_dir, file))
            elif glob.fnmatch.fnmatch(file, file_pattern):
                image_files.append(os.path.join(input_dir, file))
    
    if not image_files:
        logging.warning(f"未找到匹配的图片文件: {input_dir}, 模式: {file_pattern}")
        return 0
    
    logging.info(f"找到 {len(image_files)} 个图片文件")
    
    # 处理每个图片
    processed_count = 0
    for i, image_path in enumerate(image_files):
        try:
            # 计算输出路径
            if final_dir:
                rel_path = os.path.relpath(image_path, input_dir)
                output_path = os.path.join(final_dir, rel_path)
                output_dir_path = os.path.dirname(output_path)
                os.makedirs(output_dir_path, exist_ok=True)
            else:
                output_path = None  # 覆盖原图
            
            # 检查是否跳过已存在的结果
            if skip_existing and output_path and os.path.exists(output_path):
                logging.info(f"跳过已存在的结果: {output_path}")
                processed_count += 1
                continue
            
            logging.info(f"处理图片 {i+1}/{len(image_files)}: {image_path}")
            
            # 进行人脸处理
            final_path = enhancer.enhance_face(
                image_path, 
                slim_intensity=slim_intensity,
                overwrite_original=overwrite_original
            )
            
            if final_path:
                logging.info(f"处理成功: {final_path}")
                
                # 创建对比图
                if create_comparison and comparison_dir:
                    comparison_path = os.path.join(
                        comparison_dir, 
                        os.path.splitext(os.path.basename(image_path))[0] + "_comparison.jpg"
                    )
                    create_comparison_image(image_path, final_path, comparison_path)
                
                # 应用移动规则
                if move_rules:
                    # 获取图片文件名（不含路径）
                    file_name = os.path.basename(final_path)
                    
                    # 检查是否匹配任何规则
                    for rule in move_rules:
                        keyword = rule["keyword"]
                        target_dir = rule["target_dir"]
                        
                        if keyword in file_name:
                            # 构建目标路径
                            if os.path.isabs(target_dir):
                                # 如果是绝对路径，直接使用
                                dest_dir = target_dir
                            else:
                                # 如果是相对路径，基于输入目录
                                dest_dir = os.path.join(input_dir, target_dir)
                            
                            # 确保目标目录存在
                            os.makedirs(dest_dir, exist_ok=True)
                            
                            # 构建目标文件路径
                            dest_path = os.path.join(dest_dir, file_name)
                            
                            # 移动文件
                            try:
                                import shutil
                                shutil.copy2(final_path, dest_path)
                                logging.info(f"已将图片移动到: {dest_path}")
                            except Exception as e:
                                logging.error(f"移动图片失败: {e}")
                            
                            # 找到第一个匹配的规则后停止
                            break
                
                processed_count += 1
            else:
                logging.warning(f"处理失败: {image_path}")
        
        except Exception as e:
            logging.error(f"处理图片出错: {image_path}, 错误: {e}")
            logging.error(traceback.format_exc())
    
    logging.info(f"批量处理完成，成功处理 {processed_count}/{len(image_files)} 张图片")
    return processed_count


def create_comparison_image(original_path, final_path, output_path):
    """
    创建对比图
    
    Args:
        original_path: 原图路径
        final_path: 最终处理后的图片路径
        output_path: 输出路径
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
        plt.savefig(output_path)
        plt.close()
        
        logging.info(f"对比图保存成功: {output_path}")
    
    except Exception as e:
        logging.error(f"创建对比图失败: {e}")
        logging.error(traceback.format_exc())


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="批量人脸处理")
    
    parser.add_argument("input_dir", help="输入目录")
    parser.add_argument("--slim-intensity", "-s", type=float, default=0.5, help="瘦脸强度 (0.0-1.0)")
    parser.add_argument("--file-pattern", "-p", default="*.*", help="文件匹配模式，默认处理所有图片")
    parser.add_argument("--recursive", "-r", action="store_true", help="递归处理子目录")
    parser.add_argument("--no-skip", action="store_true", help="不跳过已存在的处理结果")
    
    args = parser.parse_args()
    
    # 限制瘦脸强度在0-1范围内
    slim_intensity = max(0.0, min(1.0, args.slim_intensity))
    
    # 打印参数
    logging.info(f"输入目录: {args.input_dir}")
    logging.info(f"瘦脸强度: {slim_intensity}")
    logging.info(f"文件匹配模式: {args.file_pattern}")
    logging.info(f"递归处理: {args.recursive}")
    logging.info(f"跳过已存在: {not args.no_skip}")
    
    # 处理图片
    processed_count = process_images(
        args.input_dir,
        args.input_dir,  # 输出目录与输入目录相同
        slim_intensity,
        args.file_pattern,
        args.recursive,
        False,  # 不创建对比图
        not args.no_skip,
        True    # 覆盖原图
    )
    
    print(f"批量处理完成，成功处理 {processed_count} 张图片")


if __name__ == "__main__":
    main() 