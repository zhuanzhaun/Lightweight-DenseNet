import torch
import os
from thop import profile
from defect_instance_evaluation_v2 import create_model
import time

def print_model_info(model, input_size=(1, 3, 608, 608), save_path='tmp.pth', device='cpu'):
    # 参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    # 模型大小
    torch.save(model.state_dict(), save_path)
    model_size = os.path.getsize(save_path) / 1024 / 1024
    os.remove(save_path)
    
    # 创建一个包装类来处理epoch和num_epochs参数
    class ModelWrapper(torch.nn.Module):
        def __init__(self, original_model):
            super(ModelWrapper, self).__init__()
            self.model = original_model
            
        def forward(self, x):
            return self.model(x, epoch=0, num_epochs=1)
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.to(device)
    
    # FLOPs
    input = torch.randn(*input_size).to(device)
    flops, _ = profile(wrapped_model, inputs=(input, ))
    flops = flops / 1e9
    print(f'参数量(M): {total_params:.2f}')
    print(f'模型大小(MB): {model_size:.2f}')
    print(f'FLOPs(G): {flops:.2f}')
    
    # FPS
    model.eval()
    with torch.no_grad():
        images = [torch.randn(1, 3, 608, 608).to(device) for _ in range(10)]
        start = time.time()
        for img in images:
            model(img, epoch=0, num_epochs=1)
        end = time.time()
    fps = len(images) / (end - start)
    print(f'FPS: {fps:.2f}')

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    # 支持densenet_mod6
    model = create_model(num_classes=2, model_name='densenet169')
    model.to(device)
    print_model_info(model, device=device) 