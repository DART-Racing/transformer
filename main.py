import os
import pickle
import time
import matplotlib.pyplot as plt
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import itertools

np.random.seed(42)


def generate_trigonometric_sample(sequence_length, trig_function, num_samples, offset_range=(-0.1, 0.1),
                                  scale_range=(0.5, 1.5)):
    t = np.linspace(0, 1, sequence_length) + np.random.uniform(*offset_range, size=(num_samples, 1))
    scaling_factor = np.random.uniform(*scale_range, size=(num_samples, 1))
    noise = np.random.uniform(-0.1, 0.1, size=(num_samples, sequence_length))

    data = trig_function(4 * np.pi * t) + noise
    data *= scaling_factor
    if trig_function is np.tan:
        data = np.clip(data, -10, 10)  # Clipping tan values to avoid extreme values

    return data


def generate_trigonometric_data(sequence_length, num_samples):
    num_samples_per_class = round(num_samples / 3)

    # Generate samples for each trigonometric function
    sin_samples = generate_trigonometric_sample(sequence_length, np.sin, num_samples_per_class)
    cos_samples = generate_trigonometric_sample(sequence_length, np.cos, num_samples_per_class)
    tan_samples = generate_trigonometric_sample(sequence_length, np.tan, num_samples_per_class)

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


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=2, dropout=0.1, num_conv_layers=2,
                 kernel_size=16):
        super(ConvLSTM, self).__init__()

        # Define initial convolutional layer
        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                                                    kernel_size=kernel_size, padding=kernel_size // 2)])

        # Add more convolutional layers, if required
        for _ in range(1, num_conv_layers):
            self.conv_layers.append(nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size,
                                              padding=kernel_size // 2))

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, src):
        # Assume src is of shape: (batch_size, sequence_length, input_dim)
        # Conv1d expects: (batch_size, channels, sequence_length)
        src = src.transpose(1, 2)  # Now shape: (batch_size, input_dim, sequence_length)

        # Apply convolutional layers
        for conv in self.conv_layers:
            src = torch.relu(conv(src))

        # Adjust the shape to fit LSTM layer
        src = src.transpose(1, 2)  # Back to: (batch_size, sequence_length, hidden_dim)

        # LSTM layers
        lstm_out, (hn, cn) = self.lstm(src)

        # Use the last hidden state
        output = hn[-1]

        # Fully connected layer to get the logits
        logits = self.output_layer(output)

        return logits


class SequenceLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=2, dropout=0.1):
        super(SequenceLSTM, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, src):
        src = self.embedding(src)
        lstm_out, (hn, cn) = self.lstm(src)
        output = hn[-1]
        logits = self.output_layer(output)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class SequenceTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, dim_model=64, num_heads=8, num_encoder_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(dim_model, num_classes)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
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


def save_dataset(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_dataset(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def data_exists(filename):
    return os.path.isfile(filename)


def generate_binance_data(steps, sequence_length, label_length, threshold_list, data_save_path='binance_data.pkl'):
    if data_exists(data_save_path):
        print("Loading raw data from file...")
        btc_df = load_dataset(data_save_path)
    else:
        print("Fetching new data from API...")
        api_key = 'VOfm4JsL1JukjzmW2UFuEdVBU9skY1I5oHiLbdQdiaCr2iqbFA2845JXALGjFVM0'
        api_secret = 'Qkw2YyLqPzUInYslHk1G8zFPob5Q7rim3mkLCAZE2H1eQBQjYlgPRt5htOgiIO2Q'
        client = Client(api_key, api_secret)
        interval = '1m'
        time_reference = 1707519600
        timestamp = 1000 * (time_reference - 60 * steps * sequence_length)
        bars = client.get_historical_klines(
            'BTCUSDT', interval, timestamp, time_reference * 1000, limit=1000)

        # Data preprocessing
        btc_df = pd.DataFrame(bars,
                              columns=['open_time', 'open', 'high', 'low', 'close',
                                       'volume', 'close_time', 'quote_asset_volume',
                                       'number_of_trades', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'ignore'])

        # Save the raw data only
        save_dataset(data_save_path, btc_df)

    features = btc_df[['open']].astype('float')
    samples, labels = process_features(features, sequence_length, label_length)
    discrete_labels = categorize_labels(labels, threshold_list)

    # Conversion to PyTorch tensors
    samples_tensor = torch.tensor(samples, dtype=torch.float32)
    labels_tensor = torch.tensor(discrete_labels, dtype=torch.long)

    return samples_tensor, labels_tensor


def process_features(features, sequence_length, label_length):
    num_samples = len(features) - sequence_length - label_length + 1
    samples = []
    labels = []

    for i in range(0, num_samples, sequence_length):
        last_entry = features.iloc[i + sequence_length - 1]
        normalized_seq = (features.iloc[i:i + sequence_length + label_length] / last_entry - 1) * 100

        sample = normalized_seq.iloc[:sequence_length].values
        label = normalized_seq.iloc[sequence_length:, 0].max()

        samples.append(sample)
        labels.append(label)

    return np.array(samples), np.array(labels)


def categorize_labels(labels, threshold_list):
    return np.array([sum(label >= np.array(threshold_list)) for label in labels])


def calculate_extended_metrics(predictions, true_labels, device):
    """
    Calculate metrics using tensors to avoid unnecessary data movement from GPU to CPU.
    """
    predictions = predictions.to(device)
    true_labels = true_labels.to(device)

    conf_matrix = confusion_matrix(true_labels.cpu().numpy(), predictions.cpu().numpy())
    class_report = classification_report(true_labels.cpu().numpy(), predictions.cpu().numpy(), output_dict=True,
                                         zero_division=0)

    # To calculate accuracy, precision, recall, and F1 score, the data must be transferred to CPU
    accuracy = accuracy_score(true_labels.cpu().numpy(), predictions.cpu().numpy())
    precision = precision_score(true_labels.cpu().numpy(), predictions.cpu().numpy(), average='macro', zero_division=0)
    recall = recall_score(true_labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')
    f1 = f1_score(true_labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')

    return accuracy, precision, recall, f1, conf_matrix, class_report


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix with enhanced text for readability and better visual representation.
    """
    figure = plt.figure(figsize=(12, 12))  # Slightly adjusted for optimal fitting
    cmap = plt.get_cmap('Blues')  # Gets a clearer colormap
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix", fontsize=24)  # Adjust font size appropriately
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=18)  # Adjust font size for readability
    plt.yticks(tick_marks, class_names, fontsize=18)

    # Normalize the confusion matrix for better clarity on the proportions
    normalized_cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Adjust text color for optimal contrast
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cm_value = normalized_cm[i, j]
        if cm_value > 0.5:  # If cell is predominantly occupied, use a different color
            color = "white"
        else:
            color = "black"

        plt.text(j, i, cm_value, horizontalalignment="center", color=color,
                 fontsize=40)  # Slightly reduced font for subtle emphasis

    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    return figure


def log_metrics(writer, phase, metrics, epoch, class_names=['bad', 'neutral', 'good']):
    loss, accuracy, precision, recall, f1, conf_matrix, class_report = metrics
    writer.add_scalar(f'Loss/{phase}', loss, epoch)
    writer.add_scalar(f'Accuracy/{phase}', accuracy, epoch)
    writer.add_scalar(f'Precision/{phase}', precision, epoch)
    writer.add_scalar(f'Recall/{phase}', recall, epoch)
    writer.add_scalar(f'F1_Score/{phase}', f1, epoch)

    # for class_name, class_metrics in class_report.items():
    #     if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
    #         writer.add_scalars(f'{class_name}/{phase}', class_metrics, epoch)

    # Log Confusion Matrix
    fig = plot_confusion_matrix(conf_matrix, class_names)
    writer.add_figure(f'Confusion Matrix/{phase}', fig, epoch)


def perform_epoch(phase, dataloader, model, criterion, device, optimizer=None):
    assert phase in ['train', 'val'], "Phase must be 'train' or 'val'"
    if phase == 'train':
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_labels = []
    all_predictions = []

    with torch.set_grad_enabled(phase == 'train'):
        for inputs, labels in tqdm(dataloader, desc=f"\n{phase.title()}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            all_labels.append(labels)
            all_predictions.append(predicted)

            total_loss += loss.item() * inputs.size(0)

    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    accuracy, precision, recall, f1, conf_matrix, class_report = calculate_extended_metrics(all_predictions, all_labels,
                                                                                            device)

    avg_loss = total_loss / len(dataloader.dataset)
    avg_metrics = (avg_loss, accuracy, precision, recall, f1)

    return avg_metrics, conf_matrix, class_report


def train(num_epochs=1000, sequence_length=64, sequences=2 ** 11, batch_size=64, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # sequences, labels = generate_trigonometric_data(sequence_length, sequences)
    sequences, labels = generate_binance_data(sequences, sequence_length, 32, [0.05, 0.15])
    sequences, labels = sequences.to(device), labels.to(device)
    plot_labels(labels)

    input_dim = sequences.shape[-1]
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(sequences, labels, test_size=0.2,
                                                                                random_state=42)
    train_dataset = FloatSequenceDataset(train_sequences, train_labels)
    val_dataset = FloatSequenceDataset(val_sequences, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = ConvLSTM(input_dim, len(torch.unique(val_labels))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=100000, gamma=0.5)

    writer = SummaryWriter()

    best_val_loss = float('inf')
    patience, trials = 100000, 0

    for epoch in range(num_epochs):
        train_metrics, train_conf_matrix, train_class_report = perform_epoch('train', train_loader, model, criterion,
                                                                             device, optimizer)
        val_metrics, val_conf_matrix, val_class_report = perform_epoch('val', val_loader, model, criterion, device)

        log_metrics(writer, 'train', train_metrics + (train_conf_matrix, train_class_report), epoch)
        log_metrics(writer, 'val', val_metrics + (val_conf_matrix, val_class_report), epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        val_loss = val_metrics[0]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model_best.pth')
            print(f"\nEpoch {epoch + 1}: Validation loss improved, saving model.")
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                print(f"\nEarly stopping on epoch {epoch + 1}")
                break

        scheduler.step()

    writer.close()


def plot_labels(labels):
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()  # Convert to NumPy array

    weights = np.ones_like(labels) / len(labels)
    labels_unique = np.unique(labels)

    for i, label in enumerate(labels_unique):
        plt.hist(labels[labels == label], bins=np.arange(len(labels_unique) + 1) - 0.5,
                 weights=weights[labels == label], rwidth=0.8,
                 label=f'Label {label}')

    plt.legend()
    plt.xlabel('Labels')
    plt.ylabel('Probability')
    plt.title('Probability Distribution of Labels')
    plt.xticks(range(len(labels_unique)))  # Set x-ticks to correspond to labels

    plt.show()


if __name__ == '__main__':
    train(sequences=6400)
