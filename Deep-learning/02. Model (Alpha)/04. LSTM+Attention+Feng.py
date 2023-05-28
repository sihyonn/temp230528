import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_layer = nn.Linear(hidden_size, 1)

    def forward(self, outputs):
        attn_weights = self.attention_layer(outputs)
        attn_weights = torch.softmax(attn_weights, dim=1)
        return torch.sum(outputs * attn_weights, dim=1)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.attention(out)
        out = self.fc(out)
        return out


class DamDataset(Dataset):
    def __init__(self, excel_file, scales):
        # Load the data
        self.data = pd.read_excel(excel_file)

        # Create the additional features
        self.data = create_features(self.data, scales)

        # Normalize the data (except year and month columns)
        for column in self.data.columns[2:]:
            self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()

        # Extract the labels
        self.labels = self.data['Dam_Level'].values

        # Drop the labels from the input data
        self.data = self.data.drop('Dam_Level', axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Convert to tensors
        sample = torch.Tensor(self.data.iloc[idx].values)
        label = torch.Tensor([self.labels[idx]])

        return sample, label


def create_features(data, scales):
    # Initialize the lists for each scale
    rainfall_scale = {scale: [0] * (scale + 1) for scale in scales}
    dam_level_scale = {scale: [0] * (scale + 1) for scale in scales}

    # Iterate over the data
    for month, row in data.iterrows():
        # Update the lists for each scale
        for scale in scales:
            # Calculate the index for the circular buffer
            index = (month % scale) + 1

            # Update the list
            rainfall_scale[scale][index] = row['Rainfall']
            dam_level_scale[scale][index] = row['Dam_Level']

            # Update the index
            rainfall_scale[scale][0] = index
            dam_level_scale[scale][0] = index

        # Only add the features if we have enough data
        if month >= max(scales):
            for scale in scales:
                # Calculate the total rainfall and average dam level for the current scale
                total_rainfall = sum(rainfall_scale[scale][1:])
                avg_dam_level = sum(dam_level_scale[scale][1:]) / scale

                # Add the new features
                data.loc[month, f'Total_Rainfall_{scale}'] = total_rainfall
                data.loc[month, f'Avg_Dam_Level_{scale}'] = avg_dam_level

    return data

def get_loader(excel_file, batch_size, scales):
    dataset = DamDataset(excel_file, scales)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train(model, device, train_loader, optimizer, epoch, loss_months):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Compute the loss for the current month
        loss = nn.MSELoss()(output, target)
        loss_months[epoch % 12] = loss.item()

        # Compute the average loss over the last 12 months
        avg_loss = sum(loss_months) / len(loss_months)
        print(f'Epoch: {epoch}, Loss: {avg_loss}')

        loss.backward()
        optimizer.step()

def main():
    # Training settings
    input_size = 6  # input dimension
    hidden_size = 32  # hidden layer dimension
    num_layers = 2  # number of hidden layers
    output_size = 1  # output dimension
    learning_rate = 0.001  # initial learning rate
    epochs = 1000  # number of epochs
    batch_size = 64  # batch size
    scales = [12, 24, 36]  # scales for feature engineering

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = get_loader('your_excel_file.xlsx', batch_size, scales)
    loss_months = [0]*12

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss_months)

if __name__ == '__main__':
    main()
