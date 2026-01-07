import math
import torch
from tqdm import tqdm

""" Analysis of model behavior on edge cases """
"""
 * Separate validation and test sets 
 * Multiple evaluation metrics : accuracy, perplexity, loss
 * Error analysis with examples 
 * Ablation study (test at least one architectural or training choice) 
"""

def evaluate(encoder, test_loader, device, criterion, logger):
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
    text = batch['tokens']
    with open("logs\\predictions.csv", "w") as f:
        f.write("Reviews | predicted label | Actual label \n\n")
        for i in range(len(text)):
            f.write(text[i] + " | " + str(predictions[i]) + " | " + str(
                labels[i]) + "\n")
        f.close()

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": perplexity
    }