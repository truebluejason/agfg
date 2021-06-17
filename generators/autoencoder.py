import pandas as pd
from pandas import DataFrame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple

from autogluon.features.generators import AbstractFeatureGenerator
from generators.dataset import CSVDataset
from generators.util import StandardScaler


class AbstractAutoEncoderFeatureGenerator(AbstractFeatureGenerator):
    """
    Parent class for feature generators that generate autoencoder embedding
    as extra features
    """

    def __init__(self, e_dim, n_epochs, **kwargs):
        super().__init__(**kwargs)
        # defer initializing torch modules until input dimension is known
        self.model = None
        self.e_dim = e_dim
        self.n_epochs = n_epochs
        self.scaler = StandardScaler()

    def _fit_transform(self, X: DataFrame, **kwargs) -> Tuple[DataFrame, dict]:
        if self.model is None:
            self._initialize_model(input_dim=X.shape[1])
        self._train(X, n_epochs=self.n_epochs,
                    lr=5e-4, batch_size=128)
        X_out = self._transform(X)
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        if self.model is None:
            raise Exception("_transform() called w/o initializing the model")
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled.to_numpy()).to(torch.float32)
        new_features = self.model.forward_test(X_tensor).to(torch.float64)
        new_features_df = DataFrame(new_features.numpy())
        new_features_df.columns = new_features_df.columns.map(str)
        # NOTE: For quick experiment, uncomment the line below
        # return new_features_df
        X.reset_index(drop=True, inplace=True)
        X = pd.concat([X, new_features_df], axis=1)
        return X

    def _train(self, X: DataFrame, n_epochs, lr, batch_size):
        X_scaled = self.scaler.fit_transform(X)
        dataloader = DataLoader(dataset=CSVDataset(df=X_scaled),
                                batch_size=batch_size,
                                shuffle=True)
        optimizer = optim.Adam(self.model.parameters(),
                               lr=lr, weight_decay=0.001)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(n_epochs):
            losses = self._train_loop(dataloader, optimizer, criterion)
            print(
                f"epoch {epoch} loss: {round(sum(losses)/len(dataloader),4)}")
        self.model.eval()

    def _train_loop(self, dataloader, optimizer, criterion) -> list:
        raise NotImplementedError()

    def _initialize_model(self, input_dim):
        raise NotImplementedError()

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()


class AutoEncoder(nn.Module):
    """
    Two layer autoencoder with ReLU + BatchNorm
    """

    def __init__(self, input_dim, e_dim):
        super().__init__()

        self.encoder = []
        encoder_shape = [input_dim]+[int(e_dim+(input_dim-e_dim)/4)]+[e_dim]
        for index in range(len(encoder_shape)-2):
            layer = [nn.Linear(encoder_shape[index], encoder_shape[index+1]),
                     nn.ReLU(), nn.BatchNorm1d(encoder_shape[index+1])]
            self.encoder.extend(layer)
        self.encoder.append(nn.Linear(encoder_shape[-2], encoder_shape[-1]))
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        decoder_shape = [e_dim]+[int(e_dim+(input_dim-e_dim)/4)]+[input_dim]
        for index in range(len(decoder_shape)-2):
            layer = [nn.Linear(decoder_shape[index], decoder_shape[index+1]),
                     nn.ReLU(), nn.BatchNorm1d(decoder_shape[index+1])]
            self.decoder.extend(layer)
        self.decoder.append(nn.Linear(decoder_shape[-2], decoder_shape[-1]))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(X)
        out = self.decoder(hidden)
        return out

    def forward_test(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(X)


class AutoEncoderFeatureGenerator(AbstractAutoEncoderFeatureGenerator):
    """
    Generates autoencoder embedding as extra features
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize_model(self, input_dim):
        self.model = AutoEncoder(input_dim=input_dim, e_dim=self.e_dim)

    def _train_loop(self, dataloader, optimizer, criterion) -> list:
        losses = []
        for input in dataloader:
            optimizer.zero_grad()
            output = self.model.forward(input)
            train_loss = criterion(output, input)
            train_loss.backward()
            optimizer.step()
            losses.append(train_loss.item())
        return losses


class DenoisingAutoEncoderFeatureGenerator(AutoEncoderFeatureGenerator):
    """
    Generates denoising autoencoder embedding as extra features
    """

    def __init__(self, noise_stdev=0.1, **kwargs):
        self.noise_stdev = noise_stdev
        super().__init__(**kwargs)

    def _train_loop(self, dataloader, optimizer, criterion) -> list:
        losses = []
        for input in dataloader:
            optimizer.zero_grad()
            noised_input = self._noise(input, self.noise_stdev)
            output = self.model.forward(noised_input)
            train_loss = criterion(output, input)
            train_loss.backward()
            optimizer.step()
            losses.append(train_loss.item())
        return losses

    def _noise(self, input: torch.Tensor, noise: float) -> torch.Tensor:
        """
        Apply Gaussian noise to input
        TODO
        1. Only apply noise to numeric columns
        2. Apply swap noise to categorical columns
        """
        with torch.no_grad():
            numeric_cols = input
            output = numeric_cols + torch.randn(numeric_cols.shape) * noise
            return output
