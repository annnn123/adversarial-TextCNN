import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(model, dev_dataloader, device):

    model.eval()
    criterion = nn.NLLLoss()

    true_labels = []
    pred_labels = []
    total_loss = 0
    data_size = 0
    with torch.no_grad():
        for idx, batch in enumerate(dev_dataloader):
            input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            data_size += input_ids.size(0)

            pred = list(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels += list(labels.cpu().numpy())
            pred_labels += pred

    loss = total_loss / data_size

    acc = accuracy_score(y_pred=pred_labels, y_true=true_labels)
    precision = precision_score(y_pred=pred_labels, y_true=true_labels)
    recall = recall_score(y_pred=pred_labels, y_true=true_labels)
    f1 = f1_score(y_pred=pred_labels, y_true=true_labels)

    return loss, acc, precision, recall, f1