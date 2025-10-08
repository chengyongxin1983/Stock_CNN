#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet股票价格预测训练脚本

使用ResNet模型训练股票价格预测任务，支持多标签分类。
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import time
from datetime import datetime
import json
from tqdm import tqdm
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from roger.cnn.resnet import get_model, count_parameters
from roger.cnn.load_traindata import load_training_data

class StockDataset(Dataset):
    """
    股票数据集类
    """
    def __init__(self, images: np.ndarray, labels_df: pd.DataFrame, label_columns: List[str]):
        """
        初始化数据集
        
        参数:
        images: 图像数组，形状为(N, H, W, C)
        labels_df: 标签DataFrame
        label_columns: 标签列名列表
        """
        self.images = images
        self.labels_df = labels_df
        self.label_columns = label_columns
        
        # 确保图像数据格式正确 (N, C, H, W)
        if len(self.images.shape) == 4 and self.images.shape[-1] == 3:
            # 从 (N, H, W, C) 转换为 (N, C, H, W)
            self.images = np.transpose(self.images, (0, 3, 1, 2))
        
        # 转换为float32并归一化到[0,1]
        self.images = self.images.astype(np.float32) / 255.0
        
        print(f"数据集初始化完成：")
        print(f"  图像形状: {self.images.shape}")
        print(f"  标签数量: {len(self.labels_df)}")
        print(f"  标签列: {self.label_columns}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            image = torch.FloatTensor(self.images[idx])
            
            # 检查图像是否包含NaN或无穷值
            if torch.isnan(image).any() or torch.isinf(image).any():
                print(f"⚠️ 警告: 图像 {idx} 包含NaN或无穷值")
                # 用零填充NaN值
                image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 根据标签列数判断是单标签还是多标签
            if len(self.label_columns) == 7 and all(col.startswith('class_') for col in self.label_columns):
                # 单标签分类（独热码）
                labels = []
                for col in self.label_columns:
                    label_value = self.labels_df.iloc[idx][col]
                    labels.append(float(label_value))
                
                labels = torch.FloatTensor(labels)
                
                # 检查标签是否包含NaN或无穷值
                if torch.isnan(labels).any() or torch.isinf(labels).any():
                    print(f"⚠️ 警告: 标签 {idx} 包含NaN或无穷值")
                    # 用零填充NaN值
                    labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
                
                return image, labels
            else:
                # 多标签分类
                labels = []
                for col in self.label_columns:
                    label_value = self.labels_df.iloc[idx][col]
                    labels.append(float(label_value))
                
                labels = torch.FloatTensor(labels)
                
                # 检查标签是否包含NaN或无穷值
                if torch.isnan(labels).any() or torch.isinf(labels).any():
                    print(f"⚠️ 警告: 标签 {idx} 包含NaN或无穷值")
                    # 用零填充NaN值
                    labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
                
                return image, labels
            
        except Exception as e:
            print(f"获取数据项 {idx} 时出错: {e}")
            print(f"图像形状: {self.images.shape if hasattr(self, 'images') else 'N/A'}")
            print(f"标签DataFrame形状: {self.labels_df.shape if hasattr(self, 'labels_df') else 'N/A'}")
            raise

class MultiLabelCrossEntropyLoss(nn.Module):
    """
    多标签交叉熵损失函数
    """
    def __init__(self, num_labels: int = 14, pos_weight: torch.Tensor = None):
        super(MultiLabelCrossEntropyLoss, self).__init__()
        self.num_labels = num_labels
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算多标签交叉熵损失
        
        参数:
        outputs: 模型输出，形状为(batch_size, num_labels)
        targets: 真实标签，形状为(batch_size, num_labels)
        
        返回:
        损失值
        """
        return self.bce_loss(outputs, targets)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        参数:
        inputs: 模型输出（logits），形状为(batch_size, num_labels)
        targets: 真实标签，形状为(batch_size, num_labels)
        
        返回:
        损失值
        """
        # 限制输入范围防止数值不稳定
        inputs = torch.clamp(inputs, min=-10, max=10)
        
        # 计算BCE损失
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-bce_loss)
        
        # 限制pt的范围防止数值不稳定
        pt = torch.clamp(pt, min=1e-8, max=1-1e-8)
        
        # 计算Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # 检查结果是否包含NaN
        if torch.isnan(focal_loss).any():
            print("⚠️ Focal Loss产生NaN，回退到BCE Loss")
            return bce_loss.mean()
        
        return focal_loss.mean()

def test_data_loader(data_loader: DataLoader, max_batches: int = 3) -> bool:
    """
    测试数据加载器是否正常工作
    
    参数:
    data_loader: 数据加载器
    max_batches: 最大测试批次数
    
    返回:
    是否正常工作
    """
    print(f"测试数据加载器，最大测试批次: {max_batches}")
    
    try:
        for batch_idx, (images, labels) in enumerate(data_loader):
            print(f"  批次 {batch_idx + 1}: 图像形状 {images.shape}, 标签形状 {labels.shape}")
            
            if batch_idx >= max_batches - 1:
                break
        
        print("数据加载器测试成功！")
        return True
        
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        return False

def create_data_loaders(images: np.ndarray, labels_df: pd.DataFrame, 
                       label_columns: List[str], batch_size: int = 32, 
                       train_ratio: float = 0.8, val_ratio: float = 0.1,
                       shuffle_train: bool = True, sequential_split: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    参数:
    images: 图像数组
    labels_df: 标签DataFrame
    label_columns: 标签列名列表
    batch_size: 批次大小
    train_ratio: 训练集比例
    val_ratio: 验证集比例
    
    返回:
    train_loader, val_loader, test_loader
    """
    # 创建数据集
    dataset = StockDataset(images, labels_df, label_columns)
    
    # 计算数据集大小
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    print(f"数据集划分：")
    print(f"  总样本数: {total_size}")
    print(f"  训练集: {train_size} ({train_size/total_size*100:.1f}%)")
    print(f"  验证集: {val_size} ({val_size/total_size*100:.1f}%)")
    print(f"  测试集: {test_size} ({test_size/total_size*100:.1f}%)")
    
    # 划分数据集
    if sequential_split:
        # 按顺序划分数据集（无随机性）
        print("使用顺序划分数据集（无随机性）")
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_size))
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        print(f"顺序划分结果:")
        print(f"  训练集索引: {train_indices[:5]}...{train_indices[-5:] if len(train_indices) > 5 else train_indices}")
        print(f"  验证集索引: {val_indices[:5]}...{val_indices[-5:] if len(val_indices) > 5 else val_indices}")
        print(f"  测试集索引: {test_indices[:5]}...{test_indices[-5:] if len(test_indices) > 5 else test_indices}")
    else:
        # 随机划分数据集
        print("使用随机划分数据集")
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    # 创建数据加载器（设置num_workers=0避免多进程问题）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device, epoch: int = 0) -> Tuple[float, float, float]:
    """
    训练一个epoch
    
    参数:
    model: 模型
    train_loader: 训练数据加载器
    criterion: 损失函数
    optimizer: 优化器
    device: 设备
    
    返回:
    (平均损失, 准确率)
    """
    model.train()
    total_loss = 0.0
    exact_correct = 0
    partial_correct = 0.0
    total = 0
    
    print(f"开始训练epoch，数据加载器长度: {len(train_loader)}")
    progress_bar = tqdm(train_loader, desc="训练中", leave=False)
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        print(f"处理批次 {batch_idx + 1}/{len(train_loader)}")
        
        # 保存前两张图像为PNG文件（仅在前两个批次）
        if batch_idx < 2:
            import cv2
            import os
            
            # 创建输出目录
            debug_dir = "roger/cnn/debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            
            # 保存前两张图像
            for img_idx in range(min(2, images.shape[0])):
                img = images[img_idx]
                label = labels[img_idx]
                
                # 转换图像格式：从 (C, H, W) 到 (H, W, C)
                img_np = img.permute(1, 2, 0).numpy()
                
                # 反归一化：从 [0,1] 到 [0,255]
                img_np = (img_np * 255).astype(np.uint8)
                
                # 生成文件名
                filename = f"batch_{batch_idx + 1}_img_{img_idx + 1}_epoch_{epoch + 1}.png"
                filepath = os.path.join(debug_dir, filename)
                
                # 保存图像
                cv2.imwrite(filepath, img_np)
                
                print(f"  保存图像: {filename}")
                print(f"    图像形状: {img.shape}")
                print(f"    图像数值范围: [{img.min():.4f}, {img.max():.4f}]")
                print(f"    标签值: {label.tolist()}")
        
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        
        # 检查输出是否包含NaN
        if torch.isnan(outputs).any():
            print(f"⚠️ 警告: 模型输出包含NaN (Epoch {epoch+1}, Batch {batch_idx+1})")
            print(f"  输出范围: [{outputs.min():.6f}, {outputs.max():.6f}]")
            print(f"  NaN数量: {torch.isnan(outputs).sum().item()}")
            # 跳过这个批次
            continue
        
        # 诊断：打印前几个批次的输出分布
        if epoch == 0 and batch_idx < 3:
            print(f"  批次 {batch_idx+1} 输出统计:")
            print(f"    输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            print(f"    输出均值: {outputs.mean().item():.4f}")
            print(f"    输出标准差: {outputs.std().item():.4f}")
            # 打印softmax后的概率分布
            probs = torch.softmax(outputs, dim=1)
            print(f"    概率范围: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
            print(f"    概率均值: {probs.mean().item():.4f}")
            # 打印标签分布
            if len(labels.shape) == 2:
                _, true_labels = torch.max(labels, 1)
                print(f"    真实标签分布: {torch.bincount(true_labels, minlength=7).tolist()}")
                print(f"    预测标签分布: {torch.bincount(torch.argmax(outputs, 1), minlength=7).tolist()}")
        
        loss = criterion(outputs, labels)
        
        # 检查损失是否为NaN
        if torch.isnan(loss):
            print(f"⚠️ 警告: 损失为NaN (Epoch {epoch+1}, Batch {batch_idx+1})")
            print(f"  输出范围: [{outputs.min():.6f}, {outputs.max():.6f}]")
            print(f"  标签范围: [{labels.min():.6f}, {labels.max():.6f}]")
            # 跳过这个批次
            continue
        
        # 反向传播
        loss.backward()
        
        # 检查梯度（仅在前几个epoch）
        if epoch < 3 and batch_idx == 0:
            # 计算梯度范数
            total_norm = 0
            param_count = 0
            total_params = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    total_params += p.numel()
            total_norm = total_norm ** (1. / 2)
            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}: 梯度范数={total_norm:.6f}, 参数层数={param_count}, 总参数数={total_params:,}")
        
        # 梯度裁剪防止梯度爆炸 - 更严格的裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        # 检查梯度是否包含NaN
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"⚠️ 警告: 梯度包含NaN (Epoch {epoch+1}, Batch {batch_idx+1})")
            # 跳过这个批次
            continue
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        
        # 计算准确率（多标签）
        with torch.no_grad():
            predictions = torch.sigmoid(outputs) > 0.5
            
            # 调试信息：打印前几个样本的预测和标签
            if batch_idx == 0 and epoch == 0:
                print(f"  调试信息 - 批次 {batch_idx + 1}:")
                for i in range(min(2, predictions.shape[0])):
                    pred = predictions[i].float()
                    label = labels[i]
                    print(f"    样本 {i+1}:")
                    print(f"      预测: {pred.tolist()}")
                    print(f"      标签: {label.tolist()}")
                    print(f"      匹配: {(pred == label).tolist()}")
                    print(f"      全部匹配: {(pred == label).all().item()}")
            
            # 计算准确率
            if len(labels.shape) == 2 and labels.shape[1] == 7:  # 单标签分类（独热码）
                # 对于独热码，使用softmax + argmax
                _, predicted = torch.max(outputs, 1)
                _, true_labels = torch.max(labels, 1)
                correct = (predicted == true_labels).sum().item()
                exact_correct += correct
            else:  # 多标签分类
                predictions = torch.sigmoid(outputs) > 0.5
                exact_match = (predictions == labels).all(dim=1).sum().item()
                exact_correct += exact_match
            
            total += labels.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{exact_correct/total:.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    exact_accuracy = exact_correct / total
    
    return avg_loss, exact_accuracy

def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                   device: torch.device) -> Tuple[float, float]:
    """
    验证一个epoch
    
    参数:
    model: 模型
    val_loader: 验证数据加载器
    criterion: 损失函数
    device: 设备
    
    返回:
    (平均损失, 准确率)
    """
    model.eval()
    total_loss = 0.0
    exact_correct = 0
    partial_correct = 0.0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="验证中", leave=False)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 统计
            total_loss += loss.item()
            
            # 计算准确率
            if len(labels.shape) == 2 and labels.shape[1] == 7:  # 单标签分类（独热码）
                # 对于独热码，使用softmax + argmax
                _, predicted = torch.max(outputs, 1)
                _, true_labels = torch.max(labels, 1)
                correct = (predicted == true_labels).sum().item()
                exact_correct += correct
            else:  # 多标签分类
                predictions = torch.sigmoid(outputs) > 0.5
                exact_match = (predictions == labels).all(dim=1).sum().item()
                exact_correct += exact_match
            
            total += labels.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{exact_correct/total:.4f}'
            })
    
    avg_loss = total_loss / len(val_loader)
    exact_accuracy = exact_correct / total
    
    return avg_loss, exact_accuracy

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 50, learning_rate: float = 0.001, 
                device: torch.device = None, save_dir: str = "roger/cnn/checkpoints",
                lr_schedule: str = "step", loss_function: str = "bce", 
                label_columns: list = None, optimizer_type: str = "adam") -> Dict:
    """
    训练模型
    
    参数:
    model: 模型
    train_loader: 训练数据加载器
    val_loader: 验证数据加载器
    num_epochs: 训练轮数
    learning_rate: 学习率
    device: 设备
    save_dir: 模型保存目录
    
    返回:
    训练历史字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    model = model.to(device)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 损失函数和优化器 - 单标签分类使用Cross-Entropy Loss
    if loss_function == 'singlelabel':
        # 计算类别权重（如果有标签分布信息）
        criterion = nn.CrossEntropyLoss()  # 单标签分类
    elif loss_function == 'bce':
        criterion = MultiLabelCrossEntropyLoss(num_labels=14)  # 多标签分类
    elif loss_function == 'focal':
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    elif loss_function == 'weighted_bce':
        # 计算类别权重
        criterion = nn.CrossEntropyLoss(weight=None)  # 可以添加类别权重
    
    # 根据参数选择优化器
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4, eps=1e-8)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    
    # 学习率调度器
    if lr_schedule == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif lr_schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif lr_schedule == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 打印模型和优化器信息
    print(f"损失函数: {loss_function}")
    print(f"优化器: Adam (lr={learning_rate}, weight_decay=1e-4)")
    print(f"学习率调度: {lr_schedule}")
    
    # 检查模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")
    
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'grad_norms': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"开始训练，共 {num_epochs} 个epoch...")
    print(f"模型参数数量: {count_parameters(model):,}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        if lr_schedule == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        
        # 打印结果
        print(f"Epoch {epoch+1:3d}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # 保存最佳模型（使用准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            # 保存模型
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'history': history
            }
            
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"  → 保存最佳模型 (Val Acc: {val_acc:.4f})")
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'history': history
            }
            torch.save(checkpoint, checkpoint_path)
    
    total_time = time.time() - start_time
    
    print(f"\n训练完成！")
    print(f"总训练时间: {total_time/60:.1f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"模型保存路径: {save_dir}")
    
    # 保存训练历史
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # 转换numpy类型为Python类型
        history_serializable = {}
        for key, values in history.items():
            history_serializable[key] = [float(v) for v in values]
        json.dump(history_serializable, f, indent=2)
    
    return history

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='ResNet股票价格预测训练')
    parser.add_argument('--data_dir', type=str, default='roger/cnn/training_data',
                       help='训练数据目录')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'lightweight'],
                       help='模型类型')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='学习率')
    parser.add_argument('--lr_schedule', type=str, default='step',
                       choices=['step', 'cosine', 'plateau'],
                       help='学习率调度策略')
    parser.add_argument('--save_dir', type=str, default='roger/cnn/checkpoints',
                       help='模型保存目录')
    parser.add_argument('--years', type=str, default='auto',
                       help='要加载的年份，用逗号分隔，或使用auto自动检测')
    parser.add_argument('--no_shuffle', action='store_true',
                       help='不打乱训练数据顺序（用于调试）')
    parser.add_argument('--sequential_split', action='store_true',
                       help='按顺序划分数据集，不使用随机划分（用于调试）')
    parser.add_argument('--loss_function', type=str, default='singlelabel',
                       choices=['bce', 'focal', 'weighted_bce', 'singlelabel'],
                       help='损失函数类型')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout率')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='优化器类型')
    
    args = parser.parse_args()
    
    print("=== ResNet股票价格预测训练 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"参数设置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  模型类型: {args.model}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  保存目录: {args.save_dir}")
    print(f"  年份: {args.years}")
    
    # 解析年份参数
    years = None
    if args.years.lower() != 'auto':
        try:
            years = [int(y.strip()) for y in args.years.split(',')]
        except ValueError:
            print(f"无效的年份参数: {args.years}")
            return
    
    # 加载训练数据
    print(f"\n=== 加载训练数据 ===")
    images, labels_df = load_training_data(args.data_dir, years)
    
    if images is None or labels_df is None:
        print("无法加载训练数据！")
        return
    
    # 根据损失函数类型决定标签处理方式
    if args.loss_function == 'singlelabel':
        # 只使用10k标签，改为单标签分类
        label_columns = [col for col in labels_df.columns if col.startswith('10k_')]
        print(f"10k标签列: {label_columns}")
        
        # 将多标签转换为单标签
        def convert_to_single_label(row):
            """将多标签转换为单标签分类（独热码）"""
            # 找到所有为1的标签
            positive_labels = [i for i, val in enumerate(row) if val == 1]
            
            # 创建7维独热码
            one_hot = [0.0] * 7
            
            if len(positive_labels) == 0:
                # 如果没有正标签，返回全0（表示无变化）
                return one_hot
            elif len(positive_labels) == 1:
                # 如果只有一个正标签，设置对应位置为1
                one_hot[positive_labels[0]] = 1.0
                return one_hot
            else:
                # 如果有多个正标签，选择最大的ATR倍数
                # 10k_4atr=0, 10k_2atr=1, 10k_1atr=2, 10k_0atr=3, 10k_-1atr=4, 10k_-2atr=5, 10k_-4atr=6
                atr_values = [4, 2, 1, 0, -1, -2, -4]
                max_atr_idx = max(positive_labels, key=lambda x: atr_values[x])
                one_hot[max_atr_idx] = 1.0
                return one_hot
        
        # 转换标签
        print("转换多标签为单标签（独热码）...")
        single_labels = []
        for idx, row in labels_df[label_columns].iterrows():
            single_label = convert_to_single_label(row.values)
            single_labels.append(single_label)
        
        # 创建新的标签DataFrame（7列独热码）
        labels_df_single = pd.DataFrame(single_labels, columns=[f'class_{i}' for i in range(7)])
        print(f"单标签分布:")
        total_samples = len(labels_df_single)
        for i in range(7):
            count = int(labels_df_single[f'class_{i}'].sum())
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"  类别{i}: {count} 个样本 ({percentage:.1f}%)")
        
        # 检查是否有类别样本为0
        zero_classes = []
        for i in range(7):
            if labels_df_single[f'class_{i}'].sum() == 0:
                zero_classes.append(i)
        if zero_classes:
            print(f"⚠️ 警告: 类别 {zero_classes} 没有样本！")
        
        # 更新标签列和DataFrame
        label_columns = [f'class_{i}' for i in range(7)]
        labels_df = labels_df_single
    else:
        # 多标签分类，使用所有标签
        label_columns = [col for col in labels_df.columns if col.startswith(('10k_', '5k_'))]
        print(f"多标签列: {label_columns}")
    
    # 分析标签分布
    print(f"\n=== 标签分布分析 ===")
    for col in label_columns:
        positive_count = labels_df[col].sum()
        total_count = len(labels_df)
        percentage = (positive_count / total_count) * 100 if total_count > 0 else 0
        print(f"  {col}: {positive_count}/{total_count} ({percentage:.1f}%)")
    
    # 检查是否有全零标签
    all_zero_count = (labels_df[label_columns].sum(axis=1) == 0).sum()
    print(f"  全零标签样本: {all_zero_count}/{total_count} ({all_zero_count/total_count*100:.1f}%)")
    
    # 检查标签稀疏性
    avg_positive_labels = labels_df[label_columns].sum(axis=1).mean()
    print(f"  平均每个样本的正标签数: {avg_positive_labels:.2f}")
    
    # 创建数据加载器
    print(f"\n=== 创建数据加载器 ===")
    shuffle_train = not args.no_shuffle
    sequential_split = args.sequential_split
    print(f"训练数据打乱: {'是' if shuffle_train else '否'}")
    print(f"顺序划分数据集: {'是' if sequential_split else '否'}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        images, labels_df, label_columns, 
        batch_size=args.batch_size,
        shuffle_train=shuffle_train,
        sequential_split=sequential_split
    )
    
    # 测试数据加载器
    print(f"\n=== 测试数据加载器 ===")
    if not test_data_loader(train_loader, max_batches=2):
        print("训练数据加载器测试失败，请检查数据！")
        return
    
    if not test_data_loader(val_loader, max_batches=1):
        print("验证数据加载器测试失败，请检查数据！")
        return
    
    # 创建模型
    print(f"\n=== 创建模型 ===")
    num_classes = 7 if args.loss_function == 'singlelabel' else len(label_columns)
    if args.model in ['lightweight']:
        from roger.cnn.lightweight_cnn import get_lightweight_model
        model = get_lightweight_model(args.model, num_classes=num_classes, input_channels=3)
    elif args.model in ['simple']:
        from roger.cnn.simple_cnn import get_simple_model
        model = get_simple_model(args.model, num_classes=num_classes, input_channels=3)
    else:
        model = get_model(args.model, num_classes=num_classes, input_channels=3, dropout_rate=args.dropout_rate)
    print(f"模型: {args.model}")
    print(f"参数数量: {count_parameters(model):,}")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 训练模型
    print(f"\n=== 开始训练 ===")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=args.save_dir,
        lr_schedule=args.lr_schedule,
        loss_function=args.loss_function,
        label_columns=label_columns,
        optimizer_type=args.optimizer
    )
    
    print(f"\n=== 训练完成 ===")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"最佳验证准确率: {max(history['val_acc']):.4f}")

if __name__ == "__main__":
    main()
