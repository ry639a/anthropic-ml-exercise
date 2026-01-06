import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead,
                 num_layers, max_len=512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=False
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.embedding(x) + self.pos_embedding(positions)
        x = x.transpose(0, 1)

        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x = self.transformer(x, mask)
        x = x.transpose(0, 1)

        x = self.fc(x)
        return x

def train_model(model, dataloader, epochs=10, device='cpu'):
    model.to(device)
    # Turn on training mode
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Loop over all epochs
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)),
                           targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f"Epoch {epoch + 1} | Loss: "
                  f"{total_loss / len(dataloader):.4f}")