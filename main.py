import io
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from binance.client import Client
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    labels_tensor = torch.tensor(labels_shuffled, dtype=torch.float32)

    return data_tensor, labels_tensor


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1, num_conv_layers=0, kernel_size=3):
        super(ConvLSTM, self).__init__()
        self.conv_layers = nn.ModuleList([])
        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2))
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, src):
        # Assume src is of shape: (batch_size, sequence_length, input_dim)
        # Conv1d expects: (batch_size, channels, sequence_length)
        src = src.transpose(1, 2)
        for conv in self.conv_layers:
            src = torch.relu(conv(src))
        src = src.transpose(1, 2)
        lstm_out, (hn, cn) = self.lstm(src)
        output = hn[-1]
        prediction = self.output_layer(output)
        return prediction


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


def generate_binance_data(steps, sequence_length, label_length, data_save_path='binance_data.pkl'):
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

    features = btc_df[['open', 'volume']].astype('float')
    samples, labels = process_features(features, sequence_length, label_length)

    # Conversion to PyTorch tensors
    samples_tensor = torch.tensor(samples, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    return samples_tensor, labels_tensor


def process_features(features, sequence_length, label_length):
    num_samples = len(features) - sequence_length - label_length + 1
    samples = []
    labels = []

    for i in range(0, num_samples, sequence_length):
        last_entry = features.iloc[i + sequence_length - 1]
        normalized_seq = (features.iloc[i:i + sequence_length + label_length] / last_entry - 1) * 100

        sample = normalized_seq.iloc[:sequence_length].values
        label = normalized_seq.iloc[sequence_length:, 0].mean()

        samples.append(sample)
        labels.append(label)

    return np.array(samples), np.array(labels)


def calculate_regression_metrics(predictions, true_labels, device='cpu'):
    """
    Calculate regression metrics.
    """
    # Ensure tensors are on the right device
    predictions = predictions.to(device)
    true_labels = true_labels.to(device)

    # Move data back to CPU for sklearn compatibility
    predictions_np = predictions.cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()

    # Calculate common regression metrics
    mse = mean_squared_error(true_labels_np, predictions_np)
    rmse = mean_squared_error(true_labels_np, predictions_np, squared=False)
    mae = mean_absolute_error(true_labels_np, predictions_np)
    r2 = r2_score(true_labels_np, predictions_np)

    return mse, rmse, mae, r2


def log_regression_metrics(writer, phase, results, epoch):
    metrics, all_labels, all_predictions = results
    avg_loss, mse, rmse, mae, r2 = metrics
    writer.add_scalar(f'Loss/{phase}', mse, epoch)
    writer.add_scalar(f'MSE/{phase}', mse, epoch)
    writer.add_scalar(f'RMSE/{phase}', rmse, epoch)
    writer.add_scalar(f'MAE/{phase}', mae, epoch)
    writer.add_scalar(f'R2/{phase}', r2, epoch)
    log_predictions_vs_actuals(all_predictions, all_labels, phase, writer, epoch)


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
        for inputs, labels in tqdm(dataloader, desc=f"\n{phase.title()} Epoch"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_labels.append(labels)
            all_predictions.append(outputs.detach())  # Store raw outputs

            total_loss += loss.item() * inputs.size(0)

    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    # Calculate regression metrics
    mse, rmse, mae, r2 = calculate_regression_metrics(all_predictions, all_labels, device)

    avg_loss = total_loss / len(dataloader.dataset)
    avg_metrics = (avg_loss, mse, rmse, mae, r2)

    return avg_metrics, all_labels, all_predictions  # No need to return confusion matrix or classification report


def train(num_epochs=1000, sequence_length=64, sequences=2 ** 11, batch_size=64, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #sequences, labels = generate_trigonometric_data(sequence_length, sequences)
    sequences, labels = generate_binance_data(sequences, sequence_length, 32)
    sequences, labels = sequences.to(device), labels.to(device)
    plot_value_distribution(labels)

    input_dim = sequences.shape[-1]
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    train_dataset = FloatSequenceDataset(train_sequences, train_labels)
    val_dataset = FloatSequenceDataset(val_sequences, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = ConvLSTM(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=100000, gamma=0.5)

    writer = SummaryWriter()

    best_val_loss = float('inf')
    patience, trials = 100000, 0

    for epoch in range(num_epochs):
        train_results = perform_epoch('train', train_loader, model, criterion, device, optimizer)
        val_results = perform_epoch('val', val_loader, model, criterion, device)

        log_regression_metrics(writer, 'train', train_results, epoch)
        log_regression_metrics(writer, 'val', val_results, epoch)

        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        val_loss = val_results[0][0]
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


def plot_value_distribution(values, plot_type='density'):
    """
    Plot the distribution of float values (e.g., target variable in regression tasks).

    Parameters:
    - values: a tensor or numpy array containing the float values.
    - plot_type: 'histogram' for a histogram plot or 'density' for a density plot.
    """
    if torch.is_tensor(values):
        values = values.cpu().numpy()  # Convert to NumPy array if input is a tensor

    plt.figure(figsize=(10, 6))

    if plot_type == 'histogram':
        plt.hist(values, bins=30, edgecolor='k', alpha=0.7)
        plt.ylabel('Frequency')
    elif plot_type == 'density':
        sns.kdeplot(values, bw_adjust=0.5)
        plt.ylabel('Density')
    else:
        raise ValueError("plot_type must be either 'histogram' or 'density'")

    plt.xlabel('Value')
    plt.title('Distribution of Values')
    plt.grid(True)
    plt.show()


def log_predictions_vs_actuals(predictions, actuals, phase, writer, epoch):
    # Create a figure for plotting
    fig = plt.figure()
    plt.scatter(actuals.cpu().numpy(), predictions.cpu().numpy())
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs. Actual Values')

    # Plot the diagonal line for reference
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=4)

    # Instead of plt.show(), we save figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Log the plot to TensorBoard
    writer.add_image(f'Predictions/{phase}', plt.imread(buf), epoch, dataformats='HWC')

    # Close the figure to prevent it from being displayed in the notebook/IDE output
    plt.close(fig)


if __name__ == '__main__':
    train(sequences=6400, learning_rate=0.001)
