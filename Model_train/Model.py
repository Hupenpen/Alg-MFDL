import torch.nn as nn
import torch


class CNN_Model(nn.Module):
    def __init__(self, input_channels):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.act = nn.ReLU()
        self.bigrucell = nn.GRU(1024, 64, num_layers=2, bidirectional=True)
        self.attention = nn.TransformerEncoderLayer(d_model=1024, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.attention, num_layers=4)
        self.transformer_linear = nn.Linear(1024, 128)
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x1):
        x1 = x1.permute(0, 2, 1)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.act(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.act(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = self.act(x1)
        x1 = x1.permute(0, 2, 1)
        final_out = x1
        final_out = self.mlp(final_out)  # MLP分类器
        final_out = torch.squeeze(final_out)
        return final_out


class BiGRU_Model(nn.Module):
    def __init__(self, input_channels):
        super(BiGRU_Model, self).__init__()
        self.bigrucell = nn.GRU(input_channels, 32, num_layers=2, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x2):
        x2, _ = self.bigrucell(x2)
        final_out = x2
        final_out = self.mlp(final_out)
        final_out = torch.squeeze(final_out)
        return final_out, x2


class AttTransform_Model(nn.Module):
    def __init__(self, input_channels):
        super(AttTransform_Model, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.act = nn.ReLU()
        self.bigrucell = nn.GRU(1024, 64, num_layers=2, bidirectional=True)
        self.attention = nn.TransformerEncoderLayer(d_model=1280, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.attention, num_layers=4)
        self.transformer_linear = nn.Linear(1280, 128)
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.transformer_linear(x)

        final_out = x
        final_out = self.mlp(final_out)
        final_out = torch.squeeze(final_out)
        return final_out


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values.astype(np.float32))
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.features[idx]
        label = self.labels[idx]
        return data, label
