import torch
import torchvision
import torch.nn as nn
import models.loss_fns as loss_fns
import os

class model(nn.Module):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(model, self).__init__()
        
        # 1. 初始化标准 ResNet50 (默认为 ImageNet 1000类结构)
        M = torchvision.models.resnet50(pretrained=False)
        
        # --- 修改 1: 提前定义分类头 (为了在加载权重时能赋值) ---
        self.num_classes = num_classes
        self.glb_pooling = nn.AdaptiveMaxPool2d((1, 1))
        # ResNet50 layer4 输出通道通常是 2048
        self.cls = nn.Linear(M.layer4[-1].conv3.out_channels, num_classes)
        
        # 2. 加载自定义预训练权重
        if pretrained:
            weight_path = r'/data/dsj/lys/SpliceMix-resnet50/pretrain-weight/pretrain_CXR14.pth'
            if os.path.exists(weight_path):
                print(f"=> loading custom pretrained weights from {weight_path}")
                try:
                    checkpoint = torch.load(weight_path, map_location='cpu')
                    
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        # 去除 DDP 的 module. 前缀
                        name = k[7:] if k.startswith('module.') else k
                        
                        # --- 修改 2: 尝试加载分类头权重 ---
                        # 检查 key 是否包含 fc, classifier, cls 等常见命名
                        if any(x in name for x in ['fc', 'classifier', 'cls']):
                            # 尝试加载 Weight (形状必须完全匹配)
                            if 'weight' in name and v.shape == self.cls.weight.shape:
                                self.cls.weight.data.copy_(v)
                                print(f"=> Loaded classifier weight from: {name}")
                                continue
                            # 尝试加载 Bias
                            elif 'bias' in name and v.shape == self.cls.bias.shape:
                                self.cls.bias.data.copy_(v)
                                print(f"=> Loaded classifier bias from: {name}")
                                continue
                            
                            # 如果形状不匹配，则跳过 (不加入 new_state_dict)
                            if name.startswith('fc.') or name.startswith('classifier.'):
                                continue

                        # 其他层则加入字典，准备加载进 Backbone (M)
                        new_state_dict[name] = v
                            
                    # 加载 Backbone 权重 (strict=False 允许 state_dict 缺少 fc 层参数)
                    msg = M.load_state_dict(new_state_dict, strict=False)
                    print(f"=> Loaded backbone weights.")
                    
                except Exception as e:
                    print(f"Error loading custom weights: {e}")
            else:
                print(f"=> Warning: Custom weight file not found at {weight_path}. Using random initialization.")

        # 3. 构建 Backbone (复用 M 的层，不包含 fc)
        self.backbone = nn.Sequential(
            M.conv1, M.bn1, M.relu, M.maxpool,
            M.layer1, M.layer2, M.layer3, M.layer4
        )
        
        self.num_classes = num_classes

        # 4. 定义新的分类头
        self.glb_pooling = nn.AdaptiveMaxPool2d((1, 1))
        in_channels = M.layer4[-1].conv3.out_channels # 2048
        self.cls = nn.Linear(in_channels, num_classes)

    def forward(self, inputs, args=None):
        fea4 = self.backbone(inputs)
        fea_gmp = self.glb_pooling(fea4).flatten(1)
        output = self.cls(fea_gmp)
        return output

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.backbone.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.backbone.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]

class Loss_fn(loss_fns.BCELoss):
    def __init__(self):
        super(Loss_fn, self).__init__()