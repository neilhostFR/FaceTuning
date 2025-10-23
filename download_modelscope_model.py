import os
import logging
import traceback
import sys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("download_modelscope_model.log"),
                             logging.StreamHandler()])

def download_skin_retouching_model(model_id='iic/cv_unet_skin-retouching', save_dir='models'):
    """
    下载ModelScope的皮肤美化模型到本地
    
    Args:
        model_id: ModelScope模型ID
        save_dir: 保存目录
    
    Returns:
        model_path: 模型保存路径
    """
    try:
        logging.info(f"开始下载皮肤美化模型: {model_id}")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        model_save_dir = os.path.join(save_dir, model_id.replace('/', '_'))
        
        # 检查模型是否已下载
        if os.path.exists(model_save_dir) and os.path.isdir(model_save_dir):
            logging.info(f"模型已存在: {model_save_dir}")
            return model_save_dir
        
        # 初始化pipeline，这会自动下载模型
        try:
            skin_retouching = pipeline('skin-retouching', model=model_id)
            logging.info("模型初始化成功")
        except Exception as e:
            logging.error(f"模型初始化失败: {e}")
            logging.error(traceback.format_exc())
            return None
        
        # 获取模型目录
        try:
            from modelscope.utils.hub import snapshot_download
            model_dir = snapshot_download(model_id)
            logging.info(f"模型下载成功，原始路径: {model_dir}")
        except Exception as e:
            logging.error(f"模型下载失败: {e}")
            logging.error(traceback.format_exc())
            return None
        
        # 复制模型文件到指定目录
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            if os.path.exists(model_save_dir):
                shutil.rmtree(model_save_dir)
            shutil.copytree(model_dir, model_save_dir)
            logging.info(f"模型文件已复制到: {model_save_dir}")
            return model_save_dir
        else:
            logging.error(f"模型目录不存在: {model_dir}")
            return None
        
    except Exception as e:
        logging.error(f"下载模型失败: {e}")
        logging.error(traceback.format_exc())
        print(f"下载模型失败: {e}")
        return None

def download_portrait_enhancement_model(model_id='damo/cv_gpen_image-portrait-enhancement-hires', save_dir='models'):
    """
    下载ModelScope的人像增强修复模型到本地
    
    Args:
        model_id: ModelScope模型ID
        save_dir: 保存目录
    
    Returns:
        model_path: 模型保存路径
    """
    try:
        logging.info(f"开始下载人像增强修复模型: {model_id}")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        model_save_dir = os.path.join(save_dir, model_id.replace('/', '_'))
        
        # 检查模型是否已下载
        if os.path.exists(model_save_dir) and os.path.isdir(model_save_dir):
            logging.info(f"模型已存在: {model_save_dir}")
            return model_save_dir
        
        # 初始化pipeline，这会自动下载模型
        try:
            portrait_enhancement = pipeline(Tasks.image_portrait_enhancement, model=model_id)
            logging.info("人像增强修复模型初始化成功")
        except Exception as e:
            logging.error(f"人像增强修复模型初始化失败: {e}")
            logging.error(traceback.format_exc())
            return None
        
        # 获取模型目录
        try:
            from modelscope.utils.hub import snapshot_download
            model_dir = snapshot_download(model_id)
            logging.info(f"人像增强修复模型下载成功，原始路径: {model_dir}")
        except Exception as e:
            logging.error(f"人像增强修复模型下载失败: {e}")
            logging.error(traceback.format_exc())
            return None
        
        # 复制模型文件到指定目录
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            if os.path.exists(model_save_dir):
                shutil.rmtree(model_save_dir)
            shutil.copytree(model_dir, model_save_dir)
            logging.info(f"人像增强修复模型文件已复制到: {model_save_dir}")
            return model_save_dir
        else:
            logging.error(f"人像增强修复模型目录不存在: {model_dir}")
            return None
        
    except Exception as e:
        logging.error(f"下载人像增强修复模型失败: {e}")
        logging.error(traceback.format_exc())
        print(f"下载人像增强修复模型失败: {e}")
        return None

if __name__ == "__main__":
    # 默认下载皮肤美化模型和人像增强修复模型
    skin_model_id = 'iic/cv_unet_skin-retouching'
    portrait_model_id = 'damo/cv_gpen_image-portrait-enhancement-hires'
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        # 如果提供了参数，检查是否是特定的模型类型
        if sys.argv[1] == 'skin':
            # 只下载皮肤美化模型
            if len(sys.argv) > 2:
                skin_model_id = sys.argv[2]
            print(f"将下载皮肤美化模型: {skin_model_id}")
            model_path = download_skin_retouching_model(skin_model_id)
            if model_path:
                print(f"皮肤美化模型下载成功，保存在: {model_path}")
                print("现在您可以在离线环境中使用此模型")
            else:
                print("皮肤美化模型下载失败")
                sys.exit(1)
        elif sys.argv[1] == 'portrait':
            # 只下载人像增强修复模型
            if len(sys.argv) > 2:
                portrait_model_id = sys.argv[2]
            print(f"将下载人像增强修复模型: {portrait_model_id}")
            model_path = download_portrait_enhancement_model(portrait_model_id)
            if model_path:
                print(f"人像增强修复模型下载成功，保存在: {model_path}")
                print("现在您可以在离线环境中使用此模型")
            else:
                print("人像增强修复模型下载失败")
                sys.exit(1)
        else:
            # 将参数作为皮肤美化模型ID
            skin_model_id = sys.argv[1]
            print(f"将下载皮肤美化模型: {skin_model_id}")
            model_path = download_skin_retouching_model(skin_model_id)
            if model_path:
                print(f"皮肤美化模型下载成功，保存在: {model_path}")
                print("现在您可以在离线环境中使用此模型")
            else:
                print("皮肤美化模型下载失败")
                sys.exit(1)
    else:
        # 下载两种模型
        print("将下载皮肤美化模型和人像增强修复模型")
        
        # 下载皮肤美化模型
        skin_model_path = download_skin_retouching_model(skin_model_id)
        if skin_model_path:
            print(f"皮肤美化模型下载成功，保存在: {skin_model_path}")
        else:
            print("皮肤美化模型下载失败")
        
        # 下载人像增强修复模型
        portrait_model_path = download_portrait_enhancement_model(portrait_model_id)
        if portrait_model_path:
            print(f"人像增强修复模型下载成功，保存在: {portrait_model_path}")
        else:
            print("人像增强修复模型下载失败")
        
        if skin_model_path and portrait_model_path:
            print("所有模型下载成功，现在您可以在离线环境中使用这些模型")
        else:
            print("部分模型下载失败")
            sys.exit(1) 