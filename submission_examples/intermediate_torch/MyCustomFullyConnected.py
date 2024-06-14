"""
Torch fully connected model
"""
import os
import pathlib
from typing import Union
import json

import numpy as np
import numpy.typing as npt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from lips.dataset import DataSet
from lips.dataset.scaler import Scaler
from lips.logger import CustomLogger
from lips.config import ConfigManager
from lips.utils import NpEncoder
from lips.augmented_simulators.torch_models.utils import LOSSES

class MyCustomFullyConnected(nn.Module):
    def __init__(self,
                 sim_config_path: Union[pathlib.Path, str],
                 bench_config_path: Union[str, pathlib.Path],
                 sim_config_name: Union[str, None]=None,
                 bench_config_name: Union[str, None]=None,
                 bench_kwargs: dict={},
                 name: Union[str, None]=None,
                 scaler: Union[Scaler, None]=None,
                 log_path: Union[None, pathlib.Path, str]=None,
                 **kwargs):
        super().__init__()
        if not os.path.exists(sim_config_path):
            raise RuntimeError("Configuration path for the simulator not found!")
        if not str(sim_config_path).endswith(".ini"):
            raise RuntimeError("The configuration file should have `.ini` extension!")
        sim_config_name = sim_config_name if sim_config_name is not None else "DEFAULT"
        self.sim_config = ConfigManager(section_name=sim_config_name, path=sim_config_path)
        self.bench_config = ConfigManager(section_name=bench_config_name, path=bench_config_path)
        self.bench_config.set_options_from_dict(**bench_kwargs)
        self.name = name if name is not None else self.sim_config.get_option("name")
        # scaler
        self.scaler = scaler
        # Logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # model parameters
        self.params = self.sim_config.get_options_dict()
        self.params.update(kwargs)

        self.activation = {
            "relu": F.relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh
        }

        self.input_size = None if kwargs.get("input_size") is None else kwargs["input_size"]
        self.output_size = None if kwargs.get("output_size") is None else kwargs["output_size"]

        self.input_layer = None
        self.input_dropout = None
        self.fc_layers = None
        self.dropout_layers = None
        self.output_layer = None

        self._data = None
        self._target = None

    def build_model(self):
        """Build the model architecture
        """
        linear_sizes = list(self.params["layers"])

        self.input_layer = nn.Linear(self.input_size, linear_sizes[0])
        self.input_dropout = nn.Dropout(p=self.params["input_dropout"])

        self.fc_layers = nn.ModuleList([nn.Linear(in_f, out_f) \
            for in_f, out_f in zip(linear_sizes[:-1], linear_sizes[1:])])

        self.dropout_layers = nn.ModuleList([nn.Dropout(p=self.params["dropout"]) \
            for _ in range(len(self.fc_layers))])

        self.output_layer = nn.Linear(linear_sizes[-1], self.output_size)

    def forward(self, data):
        """The forward pass of the model
        """
        out = self.input_layer(data)
        out = self.input_dropout(out)
        for _, (fc_, dropout) in enumerate(zip(self.fc_layers, self.dropout_layers)):
            out = fc_(out)
            out = self.activation[self.params["activation"]](out)
            out = dropout(out)
        out = self.output_layer(out)
        return out

    def process_dataset(self, dataset: DataSet, training: bool, **kwargs):
        """process the datasets for training and evaluation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Parameters
        ----------
        dataset : DataSet
            A dataset that should be processed
        training : bool, optional
            indicate if we are in training phase or not, by default False

        Returns
        -------
        DataLoader
            _description_
        """
        dtype = kwargs.get("dtype", torch.float32)
        if training:
            self._infer_size(dataset)
            batch_size = self.params["train_batch_size"]
            extract_x, extract_y = dataset.extract_data()
            if self.scaler is not None:
                extract_x, extract_y = self.scaler.fit_transform(extract_x, extract_y)
        else:
            batch_size = self.params["eval_batch_size"]
            extract_x, extract_y = dataset.extract_data()
            if self.scaler is not None:
                extract_x, extract_y = self.scaler.transform(extract_x, extract_y)

        torch_dataset = TensorDataset(torch.tensor(extract_x, dtype=dtype), torch.tensor(extract_y, dtype=dtype))
        data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=self.params["shuffle"])
        return data_loader

    def _post_process(self, data):
        """
        This function is used to inverse the predictions of the model to their original state, before scaling
        to be able to compare them with ground truth data
        """
        if self.scaler is not None:
            try:
                processed = self.scaler.inverse_transform(data)
            except TypeError:
                processed = self.scaler.inverse_transform(data.cpu())
        else:
            processed = data
        return processed

    def _reconstruct_output(self, dataset: DataSet, data: npt.NDArray[np.float64]) -> dict:
        """Reconstruct the outputs to obtain the desired shape for evaluation

        In the simplest form, this function is implemented in DataSet class. It supposes that the predictions 
        obtained by the augmented simulator are exactly the same as the one indicated in the configuration file

        However, if some transformations required by each specific model, the extra operations to obtained the
        desired output shape should be done in this function.

        Parameters
        ----------
        dataset : DataSet
            An object of the `DataSet` class 
        data : npt.NDArray[np.float64]
            the data which should be reconstructed to the desired form
        """
        data_rec = dataset.reconstruct_output(data)
        return data_rec
    
    def _infer_size(self, dataset: DataSet):
        """Infer the size of the input and ouput variables
        """
        *dim_inputs, self.output_size = dataset.get_sizes()
        self.input_size = np.sum(dim_inputs)

    def get_metadata(self):
        res_json = {}
        res_json["input_size"] = self.input_size
        res_json["output_size"] = self.output_size
        return res_json

    def _save_metadata(self, path: str):
        res_json = {}
        res_json["input_size"] = self.input_size
        res_json["output_size"] = self.output_size
        with open((path / "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def _load_metadata(self, path: str):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        with open((path / "metadata.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self.input_size = res_json["input_size"]
        self.output_size = res_json["output_size"]

    def _do_forward(self, batch, **kwargs):
        """Do the forward step through a batch of data

        This step could be very specific to each augmented simulator as each architecture
        takes various inputs during the learning procedure. 

        Parameters
        ----------
        batch : _type_
            A batch of data including various information required by an architecture
        device : _type_
            the device on which the data should be processed

        Returns
        -------
        ``tuple``
            returns the predictions made by the augmented simulator and also the real targets
            on which the loss function should be computed
        """
        non_blocking = kwargs.get("non_blocking", True)
        device = self.params.get("device", "cpu")
        self._data, self._target = batch
        self._data = self._data.to(device, non_blocking=non_blocking)
        self._target = self._target.to(device, non_blocking=non_blocking)

        predictions = self.forward(self._data)
        
        return self._data, predictions, self._target

    def get_loss_func(self, loss_name: str, **kwargs) -> torch.Tensor:
        """
        Helper to get loss. It is specific to each architecture
        """
        loss_func = LOSSES[loss_name](**kwargs)
        
        return loss_func