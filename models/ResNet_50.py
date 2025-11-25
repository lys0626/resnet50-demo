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
                        
                        # --- 关键修改：过滤掉全连接层 (fc) ---
                        # 无论它是叫 fc.weight, fc.bias, 还是 fc.linear...
                        # 只要 key 中包含 'fc' 且位于顶层名称，就跳过
                        if name.startswith('fc.') or name.startswith('classifier.'):
                            continue
                            
                        new_state_dict[name] = v
                            
                    # 加载权重 (strict=False 允许 state_dict 缺少 fc 层参数)
                    msg = M.load_state_dict(new_state_dict, strict=False)
                    print(f"=> Loaded backbone weights. Keys filtered/missing: {msg.missing_keys}")
                    
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