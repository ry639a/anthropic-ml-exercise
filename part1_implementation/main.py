import logging
import yaml
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import train
import evaluate
import model


# Logging setup
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/training.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)


# Dataset
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode(self.texts[idx])

        input_ids = encoding.ids[: self.max_len]
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }

# Load & split data
def load_and_split_data():
    data = load_dataset("imdb")

    train_data = data["train"].select(range(18000))
    val_data = data["train"].select(range(18000, len(data["train"])))
    test_data = data["test"]

    imdb_text = (
        list(train_data["text"])
        + list(val_data["text"])
        + list(test_data["text"])
        + list(data["unsupervised"]["text"])
    )

    logger.info("Dataset split completed")

    return (
        imdb_text,
        list(train_data["text"]),
        list(val_data["text"]),
        list(test_data["text"]),
        list(train_data["label"]),
        list(val_data["label"]),
        list(test_data["label"]),
    )

# Tokenizer
def train_tokenizer(imdb_text):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        imdb_text,
        vocab_size=30_000,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    )

    os.makedirs("imdb_bpe", exist_ok=True)
    tokenizer.save_model("imdb_bpe")

    return tokenizer

def get_tokenizer(imdb_text):
    if os.path.exists("imdb_bpe/vocab.json"):
        return ByteLevelBPETokenizer(
            "imdb_bpe/vocab.json",
            "imdb_bpe/merges.txt",
        )
    return train_tokenizer(imdb_text)

# DataLoaders
def create_dataloaders(batch_size=64, max_len=128):
    (
        imdb_text,
        train_text,
        val_text,
        test_text,
        train_labels,
        val_labels,
        test_labels,
    ) = load_and_split_data()

    tokenizer = get_tokenizer(imdb_text)
    logger.info("Tokenizer loaded")

    train_dataset = IMDBDataset(train_text, train_labels, tokenizer, max_len)
    val_dataset = IMDBDataset(val_text, val_labels, tokenizer, max_len)
    test_dataset = IMDBDataset(test_text, test_labels, tokenizer, max_len)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    logger.info("DataLoaders created")

    return train_dataloader, val_dataloader, test_dataloader


# Config
def setup():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# Main
def main():
    config = setup()

    batch_size = config["training"]["batch_size"]
    max_len = config["model"]["max_len"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_dl, val_dl, test_dl = create_dataloaders(batch_size, max_len)

    model_instance = model.create_model_instance(config, device)

    train.train_setup(model_instance, train_dl, val_dl, config, device)

    # Call evaluate
    results = evaluate.evaluate(
        encoder=model_instance,
        test_loader=test_dl,
        criterion=CrossEntropyLoss()
    )

    print("Test Loss:", results["loss"])
    print("Test Accuracy:", results["accuracy"])


if __name__ == "__main__":
    main()
