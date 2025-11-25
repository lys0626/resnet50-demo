import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def load_json(file_path):
    """一个辅助函数，用于从磁盘加载 JSON 文件。"""
    with open(file_path, 'rt') as f:
        return json.load(f)

class mimic(Dataset): 
    """
    用于加载 /data/mimic_cxr/mimic/ 目录结构 [cite: image_e1cd19.png] 的数据集类。
    """
    task = 'multilabel'
    
    # --- MODIFIED: 使用您提供的 14 个标签列表 ---
    # 假设这个列表的顺序与 train_y.json/test_y.json 的列顺序完全一致
    classes = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]
    num_labels = len(classes) # 应该为 14

    def __init__(self, root='', mode='train', transform=None):
        """
        :param root: 数据集的根目录 (例如: /data/mimic_cxr/mimic) [cite: image_e1cd19.png]。
        :param mode: 'train', 'valid', 或 'test'。
        :param transform: 应用于图像的 torchvision 变换。
        """
        self.root = root
        self.transform = transform
        self.img_folder = 'img_384' # [cite: image_e1cd19.png]

        if mode == 'train':
            x_path = os.path.join(self.root, 'train_x.json')
            y_path = os.path.join(self.root, 'train_y.json')
        elif mode == 'valid':
            # 警告: 您的截图中 [cite: image_e1cd19.png] 没有 'valid' (验证集) 文件。
            # engine.py [cite: zuiran/splicemix/SpliceMix-3ec25fe712cceeecf44580feff3b6796a1b79b26/engine.py] 在训练期间需要验证集。
            # 我们将 'valid' 模式指向 'test' 文件 [cite: image_e1cd19.png] 作为替代。
            print(f"警告: 未找到 'valid' 模式的 JSON 文件。正在使用 'test' JSON 文件 [cite: image_e1cd19.png] 代替。")
            x_path = os.path.join(self.root, 'test_x.json')
            y_path = os.path.join(self.root, 'test_y.json')
        elif mode == 'test':
            x_path = os.path.join(self.root, 'test_x.json')
            y_path = os.path.join(self.root, 'test_y.json')
        else:
            raise ValueError(f"不支持的 mode: {mode}")

        try:
            self.x = load_json(x_path) # 加载图像文件名列表
            self.y = np.array(load_json(y_path), dtype=np.float32) # 加载标签矩阵
        except FileNotFoundError:
            print(f"错误: 无法在 {self.root} 中找到 {os.path.basename(x_path)} 或 {os.path.basename(y_path)}")
            raise
            
        print(f"为模式 '{mode}' 加载了 {len(self.x)} 条记录 (从 JSON [cite: image_e1cd19.png])")
        
        # 验证标签数量是否匹配
        if self.y.shape[1] != self.num_labels:
            print(f"警告: 'self.classes' 中的标签数 ({self.num_labels}) 与 'y.json' 中的列数 ({self.y.shape[1]}) 不匹配。")
            print("请确保 'mimic.py' 中的 'self.classes' 列表与 JSON 文件的列顺序完全一致。")
            # 尽管如此, 我们还是以 JSON 文件为准
            self.num_labels = self.y.shape[1] 

    def get_number_classes(self):
        """返回数据集中的标签总数。"""
        return self.num_labels

    def __len__(self):
        """返回数据集中的样本总数。"""
        return len(self.x)

    def __getitem__(self, idx):
        """
        获取单个数据样本。
        engine.py [cite: zuiran/splicemix/SpliceMix-3ec25fe712cceeecf44580feff3b6796a1b79b26/engine.py] 中的 on_start_batch [cite: zuiran/splicemix/SpliceMix-3ec25fe712cceeecf44580feff3b6796a1b79b26/engine.py] 期望得到一个字典。
        """
        filename = self.x[idx]
        label = self.y[idx]
        #img_path = os.path.join(self.root, self.img_folder, filename)
        img_path = os.path.join(self.root, filename)
        try:
            # 强制转换为 'RGB' 以匹配 ResNet101 [cite: zuiran/splicemix/SpliceMix-3ec25fe712cceeecf44580feff3b6796a1b79b26/models/ResNet_101.py] 的 3 通道输入
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # 返回 engine.py [cite: zuiran/splicemix/SpliceMix-3ec25fe712cceeecf44580feff3b6796a1b79b26/engine.py] 期望的字典格式
            data = {'image': image, 'target': label, 'name': filename}
            return data
        
        # 错误处理
        except FileNotFoundError:
            print(f"错误：图像文件未找到 {img_path}。正在跳过并加载下一个。")
            next_idx = (idx + 1) % len(self.x) if len(self.x) > 0 else 0
            if len(self.x) > 0: return self.__getitem__(next_idx)
            else: raise IndexError("数据集为空或无法加载任何图像")
        except Exception as e:
            print(f"加载图像 {img_path} 时出错：{e}。正在跳过并加载下一个。")
            next_idx = (idx + 1) % len(self.x) if len(self.x) > 0 else 0
            if len(self.x) > 0: return self.__getitem__(next_idx)
            else: raise IndexError("数据集为空或无法加载任何图像")