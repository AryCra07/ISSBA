import pandas as pd
import torch
from tqdm import tqdm


def train(data_loader_train, data_loader_clean, data_loader_poisoned, model, criterion, optimizer, epochs,
          device, args):
    model.train()
    stats = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in tqdm(data_loader_train, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
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
    y_true = []
    y_predict = []
    loss_sum = []
    for batch_x, batch_y in tqdm(data_loader, desc="Evaluating"):
        batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
        output = model(batch_x)
        loss = criterion(output, batch_y)
        y_true.append(batch_y)
        y_predict.append(output)
        loss_sum.append(loss.item())

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)
    loss_sum = sum(loss_sum) / len(data_loader)

    if print_perform:
        print(f"Average Loss: {loss_sum:.4f}")

    return {
        'acc': (y_true == y_predict.argmax(dim=1)).float().mean().item(),
        'loss': loss_sum
    }
