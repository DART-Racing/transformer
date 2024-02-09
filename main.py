import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(dim_model, num_classes)
        self.dim_model = dim_model

    def forward(self, src):
        src = self.embedding(src) * np.sqrt(self.dim_model)
        # Note: No need to permute src to (seq_len, batch, input) as we are using batch_first=True
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
    sequences, labels = generate_trigonometric_data(10, 10)
    input_dim = sequences.shape[-1]
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42)
    train_dataset = FloatSequenceDataset(train_sequences, train_labels)
    val_dataset = FloatSequenceDataset(val_sequences, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SequenceTransformer(input_dim, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Initialize SummaryWriter
    writer = SummaryWriter()

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Log training loss using tensorboard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Log validation loss using tensorboard
        writer.add_scalar('Loss/val', avg_val_loss, epoch)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    writer.close()  # Make sure to close the writer object


train()