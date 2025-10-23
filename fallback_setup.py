import sys
import os
from cx_Freeze import setup, Executable

# 应用程序名称
APP_NAME = "人脸批量处理工具"
# 版本号
VERSION = "1.0.0"

# 基本目录
base_dir = os.path.abspath(os.path.dirname(__file__))

# 构建选项
build_options = {
    # 包含所有可能需要的包
    "packages": [
        "os", "sys", "cv2", "logging", "traceback", "time", "threading", "queue",
        "watchdog", "PyQt5", "matplotlib", "platform", "numpy", "dlib",
        "modelscope", "tensorflow", "concurrent", "multiprocessing", "gc", "shutil",
        "torch", "torchvision", "PIL", "yaml", "json", "requests",
        "h5py", "opt_einsum", "astor", "gast", "termcolor", "wrapt", "astunparse",
        "tensorflow.python", "tensorflow.lite", "tensorflow.keras",
        "modelscope.pipelines", "modelscope.utils", "modelscope.outputs",
        "modelscope.models", "modelscope.preprocessors",
        "matplotlib.backends.backend_qt5agg", "matplotlib.figure", "matplotlib.pyplot",
        "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets", "PyQt5.QtPrintSupport", "PyQt5.sip"
    ],
    
    # 包含所有本地模块
    "includes": [
        "face_enhancement", "face_detection", "skin_retouching", 
        "download_modelscope_model", "cache_manager", "download_model"
    ],
    
    # 包含所有必要的文件
    "include_files": [
        ("models", "models"),
        "shape_predictor_68_face_landmarks.dat",
        "icon.png",
        "README.md",
        "OFFLINE_README.md"
    ],
    
    # 其他选项
    "optimize": 1,  # 降低优化级别以减少问题
    "build_exe": os.path.join(base_dir, "build", APP_NAME),
    "include_msvcr": True,  # 包含Visual C++ 运行库
}

# 可执行文件选项
executables = [
    Executable(
        script="gui_enhancement.py",
        base="Win32GUI" if sys.platform == "win32" else None,
        target_name=f"{APP_NAME}.exe" if sys.platform == "win32" else APP_NAME,
        icon="icon.png"
    )
]

setup(
    name=APP_NAME,
    version=VERSION,
    description="人脸批量处理工具 - 支持离线使用的人脸美化程序",
    options={"build_exe": build_options},
    executables=executables
) 