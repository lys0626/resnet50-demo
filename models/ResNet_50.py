import torch
import torchvision
import torch.nn as nn
import models.loss_fns as loss_fns
import os

class model(nn.Module):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(model, self).__init__()
        
        # 1. 初始化标准 ResNet50 骨架
        M = torchvision.models.resnet50(pretrained=False)
        
        # 2. 提前定义分类头 (关键：必须在加载权重前定义)
        self.num_classes = num_classes
        self.glb_pooling = nn.AdaptiveMaxPool2d((1, 1))
        # ResNet50 layer4 输出通道通常是 2048
        in_channels = M.layer4[-1].conv3.out_channels 
        self.cls = nn.Linear(in_channels, num_classes)

        # 3. 加载自定义预训练权重
        if pretrained:
            # 请确认路径是否正确
            weight_path = r'/data/dsj/lys/SpliceMix-resnet50/pretrain-weight/pretrain_CXR14.pth'
            
            if os.path.exists(weight_path):
                print(f"=> loading custom pretrained weights from {weight_path}")
                try:
                    checkpoint = torch.load(weight_path, map_location='cpu')
                    
                    # --- 权重解包逻辑 ---
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'encoder' in checkpoint:
                        state_dict = checkpoint['encoder']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                    
                    new_state_dict = {}
                    
                    for k, v in state_dict.items():
                        # 清洗 key 名称 (去除前缀)
                        name = k
                        if name.startswith('module.'): name = name[7:]
                        if name.startswith('encoder.'): name = name[8:]
                        
                        # --- 强制加载分类头权重 ---
                        # 只要是 fc 或 classifier 相关的层，就尝试赋值给 self.cls
                        if 'fc' in name or 'classifier' in name or 'cls' in name:
                            if 'weight' in name:
                                print(f"=> Force loading {name} into cls.weight | shape: {v.shape}")
                                try:
                                    self.cls.weight.data.copy_(v)
                                except Exception as e:
                                    print(f"   Failed to copy {name}: {e}")
                                continue
                            elif 'bias' in name:
                                print(f"=> Force loading {name} into cls.bias | shape: {v.shape}")
                                try:
                                    self.cls.bias.data.copy_(v)
                                except Exception as e:
                                    print(f"   Failed to copy {name}: {e}")
                                continue
                        
                        # 其他层加入字典，准备给 Backbone 加载
                        new_state_dict[name] = v
                            
                    # 加载 Backbone 权重
                    # strict=False 是必须的，因为 new_state_dict 中移除了 'fc'
                    msg = M.load_state_dict(new_state_dict, strict=False)
                    print(f"=> Loaded backbone weights.")
                    print(f"   Missing keys (expected, as we handled fc manually): {msg.missing_keys}")
                    
                except Exception as e:
                    print(f"Error loading custom weights: {e}")
                    raise e 
            else:
                print(f"=> Warning: Custom weight file not found at {weight_path}. Using random initialization.")

        # 4. 构建 Backbone (复用 M 的层，不包含 fc)
        self.backbone = nn.Sequential(
            M.conv1, M.bn1, M.relu, M.maxpool,
            M.layer1, M.layer2, M.layer3, M.layer4
        )

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