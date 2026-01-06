import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead,
                 num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.fc(x)
            return x


def train_model(model, data, epochs=10):
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch in data:
            inputs, targets = batch
    outputs = model(inputs)
    loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")


# Training
vocab_size = 10000
model = TransformerModel(vocab_size, 512, 8, 6)
train_model(model, train_data)
