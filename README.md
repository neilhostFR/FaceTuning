# 人脸美化和增强处理工具

一个基于ModelScope模型的人脸美化和增强处理工具，支持瘦脸、皮肤美化和人像增强功能。该工具提供了图形界面和命令行两种使用方式，支持批量处理和实时监听功能。

## 功能特性

### 核心功能
- **瘦脸功能**：通过人脸检测和变形算法，实现面部瘦脸效果
- **皮肤美化**：使用ModelScope皮肤美化模型，平滑皮肤、去除瑕疵，保留皮肤细节
- **人像增强**：使用ModelScope人像增强模型，提高图片清晰度和质量
- **批量处理**：支持批量处理多张图片
- **实时监听**：监听指定目录，自动处理新添加的图片
- **智能分类**：根据文件名关键词自动分类保存处理结果

### 技术特性
- **离线模式**：支持完全离线使用，无需网络连接
- **多线程处理**：使用多线程并行处理，提高处理效率
- **内存优化**：智能内存管理，支持大图片处理
- **缓存机制**：处理结果缓存，避免重复计算
- **跨平台**：支持Windows、Linux、macOS

## 系统要求

- Python 3.7+
- 内存：建议8GB以上
- 存储：至少2GB可用空间（用于模型文件）
- GPU：可选，支持CUDA加速

## 安装说明

### 1. 克隆项目
```bash
git clone <repository-url>
cd face-enhancement-tool
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 下载模型（首次使用）
```bash
# 下载所有模型
python download_modelscope_model.py

# 或分别下载
python download_modelscope_model.py skin    # 只下载皮肤美化模型
python download_modelscope_model.py portrait # 只下载人像增强模型
```

## 使用方法

### 图形界面模式（推荐）

启动图形界面：
```bash
python gui_enhancement.py
```

#### 界面功能说明

1. **输入目录设置**
   - 选择包含待处理图片的目录
   - 支持递归处理子目录

2. **输出目录设置**
   - 设置默认输出目录
   - 未匹配规则的图片将保存到此目录

3. **处理选项**
   - **瘦脸强度**：调整瘦脸效果强度（0-1）
   - **启用瘦脸功能**：勾选后启用人脸检测和瘦脸处理
   - **启用皮肤美化功能**：勾选后启用皮肤美化处理（默认启用）
   - **启用人像增强功能**：勾选后启用人像增强处理

4. **图片移动规则**
   - 根据文件名关键词自动分类保存
   - 支持添加、编辑、删除规则

5. **批量处理**
   - 点击"开始批量处理"开始处理
   - 支持实时监听新添加的图片
   - 显示处理进度和状态

### 命令行模式

#### 单张图片处理
```bash
python face_enhancement.py
```

#### 批量处理
```bash
python -c "
from face_enhancement import FaceEnhancer
enhancer = FaceEnhancer.get_instance(offline_mode=True)
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
output_paths = enhancer.batch_enhance(
    image_paths,
    slim_intensity=0.5,
    use_face_slim=True,
    use_skin_retouch=True,
    use_portrait_enhancement=False
)
print(f'处理完成，共处理 {len(output_paths)} 张图片')
"
```

## 配置说明

### 模型配置
- 皮肤美化模型：`iic/cv_unet_skin-retouching`
- 人像增强模型：`damo/cv_gpen_image-portrait-enhancement-hires`
- 模型存储位置：`models/` 目录

### 处理参数
- `slim_intensity`：瘦脸强度，范围0-1，默认0.5
- `max_size`：最大处理尺寸，默认1920像素
- `use_face_slim`：是否启用瘦脸功能，默认False
- `use_skin_retouch`：是否启用皮肤美化，默认True
- `use_portrait_enhancement`：是否启用人像增强，默认False

## 文件结构

```
face-enhancement-tool/
├── gui_enhancement.py          # 图形界面主程序
├── face_enhancement.py         # 核心处理模块
├── face_detection.py           # 人脸检测模块
├── skin_retouching.py          # 皮肤美化模块
├── download_modelscope_model.py # 模型下载脚本
├── cache_manager.py            # 缓存管理模块
├── models/                     # 模型文件目录
│   ├── iic_cv_unet_skin-retouching/
│   └── damo_cv_gpen_image-portrait-enhancement-hires/
├── requirements.txt            # 依赖包列表
├── README.md                   # 项目说明文档
├── OFFLINE_README.md          # 离线使用说明
└── logs/                      # 日志文件目录
```

## 离线使用

该工具支持完全离线使用。首次使用前需要下载模型文件，之后即可在无网络环境下使用。

详细离线使用说明请参考：[OFFLINE_README.md](OFFLINE_README.md)

## 性能优化

### 内存优化
- 自动垃圾回收
- 智能内存管理
- 大图片自动缩放处理

### 处理速度优化
- 多线程并行处理
- 结果缓存机制
- 模型实例复用

### 建议配置
- CPU：4核心以上
- 内存：8GB以上
- 存储：SSD硬盘（提高模型加载速度）

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件是否完整下载
   - 确认models目录结构正确
   - 查看日志文件获取详细错误信息

2. **内存不足**
   - 减少并发处理数量
   - 降低图片处理尺寸
   - 关闭不必要的功能

3. **处理速度慢**
   - 启用GPU加速（如果支持）
   - 调整max_size参数
   - 使用缓存功能

### 日志文件
- `face_enhancement.log`：主要处理日志
- `skin_retouching.log`：皮肤美化日志
- `gui_enhancement.log`：图形界面日志

## 开发说明

### 项目架构
- 采用模块化设计
- 支持插件式功能扩展
- 使用单例模式管理模型实例

### 扩展开发
如需添加新功能，可以：
1. 继承`FaceEnhancer`类
2. 实现新的处理方法
3. 在GUI中添加相应选项

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 更新日志

### v1.0.0
- 初始版本发布
- 支持瘦脸、皮肤美化、人像增强功能
- 提供图形界面和命令行两种使用方式
- 支持批量处理和实时监听
- 支持离线使用

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件至：[neilhostfr@gmail.com]

---

**注意**：使用本工具时请遵守相关法律法规，不得用于非法用途。处理他人照片时请确保获得适当授权。