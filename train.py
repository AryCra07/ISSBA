import pandas as pd
import torch
from tqdm import tqdm


def train(data_loader_train, data_loader_clean, data_loader_poisoned, model, criterion, optimizer, epochs,
          device, args, accumulation_steps=4):
    model.train()
    stats = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(tqdm(data_loader_train, desc=f"Epoch {epoch + 1}/{epochs}")):
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()

            # 梯度累积 - 每 accumulation_steps 次更新一次梯度
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(data_loader_train):
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 清空梯度

            total_loss += loss.item()

        epoch_loss = total_loss / len(data_loader_train)
        print(f"# EPOCH {epoch + 1}: Training Loss: {epoch_loss:.4f}")

        # Evaluate the model after each epoch
        test_stats = eval_attack(data_loader_clean, data_loader_poisoned, model, device)
        print(
            f"# EPOCH {epoch + 1} - Test Clean Accuracy (TCA): {test_stats['clean_acc']:.4f}, Attack Success Rate ("
            f"ASR): {test_stats['asr']:.4f}")

        # Save model checkpoint
        basic_model_path = f"./checkpoints/resnet-epoch{epoch + 1}-{args.dataset}.pth"
        torch.save(model.state_dict(), basic_model_path)

        # Collect stats for each epoch to log later
        log_stats = {
            'epoch': epoch + 1,
            'train_loss': round(epoch_loss, 4),
            'test_clean_acc': round(test_stats['clean_acc'], 4),
            'test_asr': round(test_stats['asr'], 4),
            'test_clean_loss': round(test_stats['clean_loss'], 4),
            'test_asr_loss': round(test_stats['asr_loss'], 4),
        }
        stats.append(log_stats)
        df = pd.DataFrame(stats)
        df.to_csv(f"./logs/{args.dataset}_{args.target_label}.csv", index=False, header=True, encoding='utf-8')

    return stats


def eval_attack(data_loader_clean, data_loader_poisoned, model, device):
    ba = eval(data_loader_clean, model, device, print_perform=True)
    asr = eval(data_loader_poisoned, model, device, print_perform=False)
    return {
        'clean_acc': ba['acc'], 'clean_loss': ba['loss'],
        'asr': asr['acc'], 'asr_loss': asr['loss']
    }


def eval(data_loader, model, device, print_perform=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():  # 避免计算梯度，减少显存占用
        for batch_x, batch_y in tqdm(data_loader, desc="Evaluating"):
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            output = model(batch_x)
            loss = criterion(output, batch_y)

            # 累积损失和准确率
            loss_sum += loss.item()
            predictions = output.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    # 计算平均损失和准确率
    avg_loss = loss_sum / len(data_loader)
    accuracy = correct / total

    if print_perform:
        print(f"Average Loss: {avg_loss:.4f}")

    return {
        'acc': accuracy,
        'loss': avg_loss
    }
