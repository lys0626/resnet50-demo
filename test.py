import torch
import os
import torch.utils.data
from torch.cuda.amp import autocast
import numpy as np

# --- 从您的项目中导入 ---
from utilities import utils, utils_ddp, metric
from engine import Engine 
from main import parser
from utilities.utils import get_transform
opj = os.path.join

# (根据需要从您的数据集中导入)
from utilities.coco import COCO2014
from utilities.voc import VOC2007
from utilities.nih import nihchest
# 尝试导入其他数据集
try:
    from utilities.mimic import mimic
    MIMIC_IMPORTED = True
except ImportError:
    MIMIC_IMPORTED = False
try:
    from utilities.chexpert import chexpert
    CHEXPERT_IMPORTED = True
except ImportError:
    CHEXPERT_IMPORTED = False


def run_evaluation(args):
    """
    运行模型评估的主函数
    """
    
    # 1. 检查是否指定了 checkpoint
    if not args.resume:
        if args.rank == 0:
            print("错误: 必须使用 -r /path/to/best_checkpoint.pt 来指定要测试的模型。")
        return

    # 2. 将 'evaluate' 模式设为 0 (纯评估模式)
    args.evaluate = 0
    
    # 3. 创建 Engine 实例
    if args.rank == 0:
        print(f"--- 正在初始化引擎并从 {args.resume} 加载模型 ---")
    
    engine = Engine(args) 
    model = engine.model      # 获取已加载权重并封装 DDP 的模型
    loss_fn = engine.loss_fn  # 获取损失函数

    # 4. --- 手动加载真正的 "测试" 数据集 ---
    if args.rank == 0:
        print(f"--- 正在为 '{args.data_set}' 加载 'test' 模式数据集 ---")
        
    test_transfm = get_transform(args, is_train=False)
    
    # 动态构建 data_dict
    data_dict = {'MS-COCO': COCO2014, 'VOC2007': VOC2007, 'NIH-CHEST': nihchest}
    if MIMIC_IMPORTED: data_dict['MIMIC'] = mimic
    if CHEXPERT_IMPORTED: data_dict['CHEXPERT'] = chexpert
    
    if args.data_set not in data_dict:
         raise ValueError(f"数据集 {args.data_set} 未在 test.py 中配置或无法导入")

    data_dir = args.data_root 
    
    # 数据集加载逻辑
    if args.data_set in ('MS-COCO'):
        data_dir = opj(args.data_root, 'COCO2014')
        real_test_set = data_dict[args.data_set](data_dir, phase='val', transform=test_transfm)
    elif args.data_set in ('VOC2007'):
        data_dir = opj(args.data_root, 'VOC2007')
        real_test_set = data_dict[args.data_set](data_dir, phase='test', transform=test_transfm)
    elif args.data_set in ('NIH-CHEST', 'MIMIC', 'CHEXPERT'):
        real_test_set = data_dict[args.data_set](data_dir, mode='test', transform=test_transfm)
    else:
        raise ValueError(f"数据集 {args.data_set} 的加载逻辑未在 test.py 中配置")

    # 5. --- 创建 "测试" Dataloader ---
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(real_test_set, shuffle=False)
    else:
        test_sampler = None
    
    real_test_loader = torch.utils.data.DataLoader(
        real_test_set, 
        batch_size=args.batch_size_per,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False, 
        sampler=test_sampler,
        shuffle=False
    )
    
    if args.rank == 0:
        print(f"成功加载 {len(real_test_set)} 个测试图像。")

    # 6. --- 手动运行评估循环 ---
    model.eval()
    
    # 初始化 Meter (不需要 difficult_examples 参数，因为新的 metric.py 内部处理了)
    ap_meter = metric.AveragePrecisionMeter(difficult_examples=False)
    loss_meter = metric.AverageMeter('loss_test')

    if args.rank == 0:
        # 尝试使用 tqdm，如果没有安装则回退
        try:
            from tqdm import tqdm
            loader_tqdm = tqdm(real_test_loader, desc="Testing")
        except ImportError:
            loader_tqdm = real_test_loader
    else:
        loader_tqdm = real_test_loader

    for i, data in enumerate(loader_tqdm):
        inputs = data['image'].to(args.rank)
        targets_gt = data['target']
        file_name = data['name']
        targets = targets_gt.clone().to(args.rank)
        targets[targets == -1] = 0

        with torch.no_grad():
            with autocast(enabled=not args.disable_amp):
                # 注意：ResNet50 模型定义中不需要 extra args，传空字典即可
                outputs = model(inputs, args={}) 
                loss = loss_fn(outputs, targets)
        
        # 处理 tuple 输出
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        # 截取有效 batch (防止最后一个 batch 补齐)
        outputs = outputs[:inputs.shape[0]].data

        # DDP 数据收集
        if args.distributed:
            outputs_all = utils_ddp.distributed_concat(outputs.detach(), args.batch_size)
            targets_gt_all = utils_ddp.distributed_concat(targets_gt.detach().to(args.rank), args.batch_size)
            # loss_all = utils_ddp.distributed_concat(loss.detach().unsqueeze(0), args.world_size) # 暂时不用
        else:
            outputs_all = outputs.detach()
            targets_gt_all = targets_gt.detach()
            
        loss_meter.add(loss.cpu())
        
        # 只有主进程记录指标数据
        if args.rank == 0:
            ap_meter.add(outputs_all.detach().cpu(), targets_gt_all.cpu(), file_name)
    
    if args.distributed:
        utils_ddp.barrier() # 等待所有进程完成

    # 7. --- 在主进程中打印最终结果 (使用新指标) ---
    if args.rank == 0:
        loss = loss_meter.average()
        
        print(f"\n--- [ 最终测试结果 ] ---")
        print(f"Checkpoint: {args.resume}")
        print(f"Loss: {loss:.4f}")

        # [关键修改] 调用新的 compute_all_metrics 方法
        # 这需要您已经使用了上一轮提供的 updated utilities/metric.py
        metrics_res = ap_meter.compute_all_metrics()
        
        if not metrics_res:
            print("警告: 未能计算出指标 (可能数据为空)。")
        else:
            print("\n=== 指标 ===")
            print(f"mAUC:     {metrics_res.get('mAUC', -1):.4f}")
            print(f"Micro F1: {metrics_res.get('micro_F1', -1):.4f}")
            print(f"Macro F1: {metrics_res.get('macro_F1', -1):.4f}")
            print("-" * 30)
            print(f"Micro P:  {metrics_res.get('micro_P', -1):.4f}")
            print(f"Micro R:  {metrics_res.get('micro_R', -1):.4f}")
            print(f"Macro P:  {metrics_res.get('macro_P', -1):.4f}")
            print(f"Macro R:  {metrics_res.get('macro_R', -1):.4f}")
            print("=" * 30)

    if args.distributed:
        utils_ddp.cleanup()

if __name__ == "__main__":
    # 1. 解析参数
    args = parser.parse_args()

    # 2. 初始化环境
    args = utils.init(args) 
    
    # 3. 运行评估
    run_evaluation(args)