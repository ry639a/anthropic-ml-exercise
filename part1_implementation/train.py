import matplotlib.pyplot as plt
import torch

from part1_implementation.main import logger


# Attention plotting
def plot_attention(attn_weights, tokens, layer=0, head=0):
    attn = attn_weights[layer][0, head].detach().cpu().numpy()

    seq_len = min(len(tokens), attn.shape[-1])
    attn = attn[:seq_len, :seq_len]
    tokens = tokens[:seq_len]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(attn, cmap="viridis")

    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)

    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")
    ax.set_title(f"Layer {layer+1}, Head {head+1}")
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()


# Training & Validation loops
def train_model(model, train_loader, val_loader,
    optimizer, criterion, num_epochs=10, patience=3,
    device="cpu", scheduler=None, logger=logger):
    model.to(device)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device).long()

            # PAD = 0 → True means padding
            attention_mask = (input_ids == 0)
            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with (torch.no_grad()):
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device).long()
                attention_mask = (input_ids == 0)
                logits, _ = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)

                predicted = torch.argmax(logits, dim=1)
                logger.debug(f"labels: {labels}")
                logger.debug(f"predicted: {predicted}")
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / total
        val_acc = correct / total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    return train_losses, val_losses, val_accuracies


# Setup function
def train_setup(model_instance, train_dataloader, val_dataloader,
                config, device, logger):
    training_cfg = config["training"]
    optimizer_cfg = training_cfg["optimizer"]

    num_epochs = int(training_cfg.get("num_epochs", 10))
    lr = float(training_cfg["lr"])

    # Optimizer
    if optimizer_cfg["name"].lower() == "adam":
        optimizer = torch.optim.Adam(
            model_instance.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_cfg['name']}")

    criterion = torch.nn.CrossEntropyLoss()

    # Scheduler (optional)
    scheduler = None
    scheduler_cfg = training_cfg.get("scheduler")
    if scheduler_cfg:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 3),
            gamma=scheduler_cfg.get("gamma", 0.1),
        )

    return train_model(
        model=model_instance,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device,
        scheduler=scheduler,
        logger=logger
    )