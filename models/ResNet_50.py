import torch
import torchvision
import torch.nn as nn
import models.loss_fns as loss_fns
class LogitAdjustment(nn.Module):
    def __init__(self, cls_priors, tau=1.0):
        super(LogitAdjustment, self).__init__()
        self.tau = tau
        # 计算 log(p)
        # 加上 1e-12 防止 log(0)
        prior_log = torch.log(torch.tensor(cls_priors) + 1e-12)
        # 注册为 buffer (不会作为参数更新，但会随模型保存)
        self.register_buffer('prior_log', prior_log)

    def forward(self, logits):
        # Logit Adjustment 公式: logits - tau * log(p)
        # 注意: 这里的符号取决于具体推导，但在多标签长尾中，
        # 我们希望 Rare Class (log p 是负大数) 的 Logit 被 '提升'。
        # logits - (负数) = logits + 正数，从而更容易被激活。
        return logits - self.tau * self.prior_log
    
class model(nn.Module):
    def __init__(self, num_classes, pretrained=True, args=None, use_logit_adj=False, cls_priors=None):
        super(model, self).__init__()
        M = torchvision.models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(M.conv1, M.bn1, M.relu, M.maxpool,
                                      M.layer1, M.layer2, M.layer3, M.layer4, )
        self.num_classes = num_classes

        self.glb_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.cls = nn.Linear(M.layer4[-1].conv3.out_channels, num_classes)
        # 2. [新增] 初始化 Logit Adjustment
        self.use_logit_adj = use_logit_adj
        if use_logit_adj and cls_priors is not None:
            # print("Initializing Logit Adjustment Layer...") 
            self.logit_adj_layer = LogitAdjustment(cls_priors, tau=1.0)
        else:
            self.logit_adj_layer = None
    def forward(self, inputs, args=None):  #

        fea4 = self.backbone(inputs)  # bs, C, h, w
        fea_gmp = self.glb_pooling(fea4).flatten(1)  # bs, C
        output = self.cls(fea_gmp)    # bs, nc
        # 3. [新增] 应用 Logit Adjustment
        # 通常建议在训练时应用，推理时可选(这里默认一直应用以消除偏置)
        if self.use_logit_adj and self.logit_adj_layer is not None:
            output = self.logit_adj_layer(output)
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

if __name__ == '__main__':
    inputs = torch.randn((2, 3, 448, 448)).cuda()
    target = torch.zeros((2, 20)).cuda()
    target[:, 1:3] = 1

    loss_fn = Loss_fn()

    model = model(20).cuda()
    output = model(inputs)

    loss = loss_fn(output, target)
    loss.backward()

    a= 'pause'
