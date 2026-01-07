import math
import torch
from tqdm import tqdm
import csv

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
    predictions, labels_list, wrong_preds = [], [], []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = (input_ids == 0)
            labels = batch["label"].to(device).long()
            #texts.append(batch['tokens'])

            logits, _ = encoder(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            predicted = logits.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            preds_list = predicted.cpu().tolist()
            labels_list = labels.cpu().tolist()

            predictions.extend(list(zip(preds_list, labels_list)))

        with open("logs\\predictions.csv", "w") as f:
            f.write("Reviews | predicted label | Actual label \n")
            for row in predictions:
                f.write("" + " | " + str(row[0]) + " | " + str(row[1]) + "\n")
            f.close()

        wrong_preds = [row for row in predictions if row[0] != row[1]]

        with open("logs\\wrong_predictions.csv", "w") as f:
            f.write("Reviews | predicted label | Actual label \n")
            for row in wrong_preds:
                f.write("" + " | " + str(row[0]) + " | " + str(row[1]) + "\n")
            f.close()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    perplexity = math.exp(avg_loss)

    print("Test Loss:", avg_loss)
    print("Test Accuracy:", accuracy)
    print("Test perplexity:", perplexity)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": perplexity
    }