import os
import numpy as np
import pandas as pd
from utilities.mimic import mimic
from utilities.nih import nihchest
#统计mimic和nih数据集中每个类别的样本数量和比例
def count_class_samples(dataset_name, dataset_obj, classes):
    print(f"\n=== 统计 {dataset_name} 数据集样本分布 ===")
    
    # 获取标签矩阵 (Samples x Classes)
    # mimic 和 nih 类都将标签存储在 self.y 中
    try:
        if hasattr(dataset_obj, 'y'):
            labels = dataset_obj.y
        elif hasattr(dataset_obj, 'targets'): # 兼容性处理
            labels = np.array(dataset_obj.targets)
        else:
            print(f"无法找到 {dataset_name} 的标签数据。")
            return

        # 计算每列的和（即每个类别的正样本数）
        counts = np.sum(labels, axis=0)
        total_samples = labels.shape[0]

        print(f"总样本数: {total_samples}")
        print(f"{'Class Name':<30} | {'Count':<10} | {'Ratio (%)':<10}")
        print("-" * 55)
        
        for i, class_name in enumerate(classes):
            count = int(counts[i])
            ratio = (count / total_samples) * 100
            print(f"{class_name:<30} | {count:<10} | {ratio:.2f}%")
            
    except Exception as e:
        print(f"统计 {dataset_name} 时出错: {e}")

if __name__ == '__main__':
    # 1. 设置路径 (请修改为您服务器上的实际路径)
    # 根据 main.py 的默认参数推测的路径
    nih_root = r'/data/nih-chest-xrays' 
    # mimic.py 中似乎没有默认 root，请填写真实路径，例如 /data/mimic_cxr/PA
    mimic_root = r'/data/mimic_cxr/PA/7_1_2' 

    # 2. 统计 MIMIC
    if os.path.exists(mimic_root):
        # 实例化训练集以统计分布
        # mode='train' 读取 mimic_train_PA224.csv
        mimic_dset = mimic(root=mimic_root, mode='train') 
        count_class_samples('MIMIC (Train)', mimic_dset, mimic_dset.classes)
    else:
        print(f"跳过 MIMIC: 路径不存在 {mimic_root}")

    # 3. 统计 NIH
    # if os.path.exists(nih_root):
    #     # mode='train' 读取 cxr14_train.csv
    #     # nih_dset = nihchest(root=nih_root, mode='train')
    #     # count_class_samples('NIH (Train)', nih_dset, nih_dset.classes)
    #     nih_dset = nihchest(root=nih_root, mode='valid')
    #     count_class_samples('NIH (valid)', nih_dset, nih_dset.classes)
    #     nih_dset = nihchest(root=nih_root, mode='test')
    #     count_class_samples('NIH (test)', nih_dset, nih_dset.classes)
    # else:
    #     print(f"跳过 NIH: 路径不存在 {nih_root}")