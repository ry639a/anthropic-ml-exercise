import math
import torch
from tqdm import tqdm
from torch.nn import BCELoss

""" Analysis of model behavior on edge cases """
"""
 * Separate validation and test sets 
 * Multiple evaluation metrics : accuracy, perplexity, loss
 * Error analysis with examples 
 * Ablation study (test at least one architectural or training choice) 
"""

def evaluate(encoder, test_loader, device, criterion = BCELoss()):
    encoder.eval()

    total_loss = 0.
    total_correct = 0
    total_samples = 0
    predictions = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = (input_ids == 0)
            labels = batch["label"].to(device).long()

            logits, _ = encoder(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            predicted = logits.argmax(dim=1)
            predictions.append(predicted)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    perplexity = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": perplexity
    }