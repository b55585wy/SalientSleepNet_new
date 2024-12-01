import torch
import yaml
from models.sleep_stage_sst import SleepStageSST
from torch.utils.data import DataLoader
from utils.metrics import compute_metrics

def train(config):
    # 初始化模型
    model = SleepStageSST(config)
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate
    )
    
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(config.training.epochs):
        model.train()
        for batch in train_loader:
            signals, labels = batch
            
            # 前向传播
            logits = model(signals)
            loss = criterion(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 验证
        model.eval()
        metrics = evaluate(model, valid_loader)
        print(f"Epoch {epoch}: {metrics}")

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    train(config)