import os
import urllib.request
import bz2
import shutil
import sys
import platform

def download_model():
    """下载dlib的人脸关键点检测器模型"""
    model_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_model_path = "shape_predictor_68_face_landmarks.dat.bz2"
    model_path = "shape_predictor_68_face_landmarks.dat"
    
    # 检查模型是否已存在
    if os.path.exists(model_path):
        print(f"模型文件已存在: {model_path}")
        return True
    
    # 获取当前操作系统
    system = platform.system()
    print(f"当前操作系统: {system}")
    
    print(f"开始下载模型文件: {model_url}")
    
    try:
        # 下载压缩模型文件
        urllib.request.urlretrieve(model_url, compressed_model_path)
        print(f"下载完成: {compressed_model_path}")
        
        # 解压模型文件
        print(f"开始解压模型文件...")
        with bz2.BZ2File(compressed_model_path) as f_in, open(model_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"解压完成: {model_path}")
        
        # 删除压缩文件
        os.remove(compressed_model_path)
        print(f"已删除压缩文件: {compressed_model_path}")
        
        # 针对macOS系统，确保文件权限正确
        if system == "Darwin":
            try:
                os.chmod(model_path, 0o644)  # 设置读写权限
                print(f"已设置文件权限: {model_path}")
            except Exception as e:
                print(f"设置文件权限时出错: {e}")
        
        return True
    
    except Exception as e:
        print(f"下载或解压模型文件时出错: {e}")
        
        # 针对不同系统提供手动下载建议
        if system == "Darwin":  # macOS
            print("\nmacOS系统手动下载命令:")
            print("curl -L \"https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2\" -o shape_predictor_68_face_landmarks.dat.bz2")
            print("bunzip2 shape_predictor_68_face_landmarks.dat.bz2")
        elif system == "Windows":
            print("\nWindows系统手动下载命令:")
            print("curl -L \"https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2\" -o shape_predictor_68_face_landmarks.dat.bz2")
            print("或者使用浏览器下载后，使用7-Zip等工具解压")
        else:  # Linux等其他系统
            print("\nLinux系统手动下载命令:")
            print("curl -L \"https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2\" -o shape_predictor_68_face_landmarks.dat.bz2")
            print("bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
            
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1) 