import os

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from binance.client import Client

def generate_trigonometric_data(sequence_length, num_samples_per_class):
    # Initialize lists to collect samples
    sin_samples = []
    cos_samples = []
    tan_samples = []

    for _ in range(num_samples_per_class):
        # Time steps with unique random offset for each sample
        t = np.linspace(0, 1, sequence_length) + np.random.uniform(-0.1, 0.1)

        # Unique random scaling factor for each sample
        scaling_factor = np.random.uniform(0.5, 1.5)

        # Generate sin, cos, tan data with unique offsets and scaling for each sample
        sin_data = (np.sin(2 * np.pi * t) + np.random.uniform(-0.1, 0.1)) * scaling_factor
        cos_data = (np.cos(2 * np.pi * t) + np.random.uniform(-0.1, 0.1)) * scaling_factor
        tan_data = (np.tan(2 * np.pi * t) + np.random.uniform(-0.1, 0.1)) * scaling_factor
        tan_data = np.clip(tan_data, -10, 10)  # Clipping tan values to avoid extreme values

        sin_samples.append(sin_data)
        cos_samples.append(cos_data)
        tan_samples.append(tan_data)

    # Concatenate samples from each class
    data = np.concatenate([sin_samples, cos_samples, tan_samples], axis=0)

    # Labels: 0 for sin, 1 for cos, 2 for tan
    labels = np.array([0] * num_samples_per_class + [1] * num_samples_per_class + [2] * num_samples_per_class)

    # Shuffle the dataset
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    data_shuffled = data[indices]
    labels_shuffled = labels[indices]

    # Convert to PyTorch tensors and add a channel dimension
    data_tensor = torch.tensor(data_shuffled, dtype=torch.float32).unsqueeze(-1)
    labels_tensor = torch.tensor(labels_shuffled, dtype=torch.long)

    return data_tensor, labels_tensor


class SequenceTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, dim_model=512, num_heads=8, num_encoder_layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(dim_model, num_classes)
        self.dim_model = dim_model

    def forward(self, src):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.dim_model, dtype=torch.float32)).to(src.device)  # Adjustment for scaling factor
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Adjust pooling direction since batch is now first
        logits = self.output_layer(output)
        return logits


class FloatSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train():
    n_classes = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequences, labels = generate_trigonometric_data(10, 1024)
    sequences, labels = sequences.to(device), labels.to(device)  # Moving data to the correct device early
    input_dim = sequences.shape[-1]

    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42)
    train_dataset = FloatSequenceDataset(train_sequences, train_labels)
    val_dataset = FloatSequenceDataset(val_sequences, val_labels)


    batch_size = 64  # Example batch size, adjust as needed

    # Create DataLoader instances with the new batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, criterion, optimizer, and scheduler creation follows...
    model = SequenceTransformer(input_dim, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    # Early stopping and model checkpointing
    best_val_loss = float('inf')
    patience, trials = 100000, 0  # patience: number of epochs to wait before early stopping if no improvement
    save_path = 'model_best.pth'  # Path to save the best model

    # Initialize SummaryWriter
    writer = SummaryWriter()

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training")
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_loader_tqdm.set_postfix({"Training Loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # Log training loss using tensorboard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        model.eval()
        total_val_loss = 0
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation")
        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                val_loader_tqdm.set_postfix({"Validation Loss": f"{loss.item():.4f}"})

        avg_val_loss = total_val_loss / len(val_loader)

        # Log validation loss using tensorboard
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # Checkpoint model if improvement in validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch + 1}: Validation loss improved, saving model in {save_path}")
            trials = 0  # Reset trials
        else:
            trials += 1
            if trials >= patience:
                print(f"Early stopping on epoch {epoch + 1}")
                break

        scheduler.step()  # Step the scheduler after each epoch

    writer.close()

if __name__ == '__main__':
    api_key = 'VOfm4JsL1JukjzmW2UFuEdVBU9skY1I5oHiLbdQdiaCr2iqbFA2845JXALGjFVM0'
    api_secret = 'Qkw2YyLqPzUInYslHk1G8zFPob5Q7rim3mkLCAZE2H1eQBQjYlgPRt5htOgiIO2Q'
    client = Client(api_key, api_secret)
    interval = '1h'
    timestamp = client._get_earliest_valid_timestamp('BTCUSDT', interval)
    print(timestamp)
    # valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    bars = client.get_historical_klines('BTCUSDT', interval, timestamp, limit=1000)
    btc_df = pd.DataFrame(bars, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    length_of_dataframe = len(btc_df)
    print("Length of DataFrame:", length_of_dataframe)