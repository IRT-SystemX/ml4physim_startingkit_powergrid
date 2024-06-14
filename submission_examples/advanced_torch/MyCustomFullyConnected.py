import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from lips.augmented_simulators import AugmentedSimulator
from lips.benchmark.powergridBenchmark import PowerGridBenchmark

class TorchSimulator(AugmentedSimulator):
    def __init__(self,
                 benchmark,
                 device,
                 **kwargs):
        
        ## the dataset is provided through the benchmark object during ingestion

        ## Paramaters can be passsed through "simulator_extra_parameters" in the paraùeters.json file
        params = kwargs
        

        input_size, output_size = infer_input_output_size(benchmark.train_dataset)


        ## initialisation of the model
        model = MyCustomModel(input_size=input_size,
                               output_size=output_size,
                               hidden_sizes=(50,100,50),
                               activation=F.relu
                               )


        super().__init__(model)
        self.model = model
        self.device = device

    def build_model(self):
        self.model.build_model()

    def train(self, train_dataset, val_dataset, **kwargs):
        ## training and validation set are passed during training
        train_loader = process_dataset(train_dataset, training=True)
        val_loader = process_dataset(val_dataset)

        ##training parameters are passed through parameters.json
        params = kwargs
        self.build_model()
        self.model.to(self.device)
        train_losses = []
        val_losses = []
        # select your optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])
        # select your loss function
        loss_function = nn.MSELoss()
        for epoch in range(params["epochs"]):
            # set your model for training
            self.model.train()
            total_loss = 0
            # iterate over the batches of data
            for batch in train_loader:
                data, target = batch
                # transfer your data on proper device. The model and your data should be on the same device
                data = data.to(self.device)
                target = target.to(self.device)
                # reset the gradient
                optimizer.zero_grad()
                # predict using your model on the current batch of data
                prediction = self.model(data)
                # compute the loss between prediction and real target
                loss = loss_function(prediction, target)
                # compute the gradient (backward pass of back propagation algorithm)
                loss.backward()
                # update the parameters of your model
                optimizer.step()
                total_loss += loss.item() * len(data)
            # the validation step is optional
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)
            mean_loss = total_loss / len(train_loader.dataset)
            print(f"Train Epoch: {epoch}   Avg_Loss: {mean_loss:.5f}")
            train_losses.append(mean_loss)
        return train_losses, val_losses

    def validate(self, val_loader):
        # set the model for evaluation (no update of the parameters)
        self.model.eval()
        total_loss = 0
        loss_function = nn.MSELoss()
        with torch.no_grad():
            for batch in val_loader:
                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device)
                prediction = self.model(data)
                loss = loss_function(prediction, target)
                total_loss += loss.item()*len(data)
            mean_loss = total_loss / len(val_loader.dataset)
            print(f"Eval:   Avg_Loss: {mean_loss:.5f}")
        return mean_loss

    def predict(self, dataset, eval_batch_size=128, shuffle=False, env=None, **kwargs):
        # set the model for the evaluation
        self.model.eval()
        predictions = []
        observations = []
        test_loader = process_dataset(dataset, batch_size=eval_batch_size, training=False, shuffle=shuffle)
        # we dont require the computation of the gradient
        with torch.no_grad():
            for batch in test_loader:
                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device)
                prediction = self.model(data)
                
                if self.device == torch.device("cpu"):
                    predictions.append(prediction.numpy())
                    observations.append(target.numpy())
                else:
                    predictions.append(prediction.cpu().data.numpy())
                    observations.append(target.cpu().data.numpy())
        # reconstruct the prediction in the proper required shape of target variables
        predictions = np.concatenate(predictions)
        predictions = dataset.reconstruct_output(predictions)
        # Do the same for the real observations
        observations = np.concatenate(observations)
        observations = dataset.reconstruct_output(observations)

        return predictions


def process_dataset(dataset, batch_size: int=128, training: bool=False, shuffle: bool=False, dtype=torch.float32):
    if training:
        batch_size = batch_size
        extract_x, extract_y = dataset.extract_data()
    else:
        batch_size = batch_size
        extract_x, extract_y = dataset.extract_data()

    torch_dataset = TensorDataset(torch.tensor(extract_x, dtype=dtype), torch.tensor(extract_y, dtype=dtype))
    data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def infer_input_output_size(dataset):
    *dim_inputs, output_size = dataset.get_sizes()
    input_size = np.sum(dim_inputs)
    return input_size, output_size


class MyCustomModel(nn.Module):
    def __init__(self,
                 name: str="MyCustomFC",
                 input_size: int=None,
                 output_size: int=None,
                 hidden_sizes: tuple=(100,100,),
                 activation=F.relu
                ):
        super().__init__()
        self.name = name
        
        self.activation = activation

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

    def build_model(self):
        # model architecture
        self.input_layer = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.fc_layers = nn.ModuleList([nn.Linear(in_f, out_f) \
                                        for in_f, out_f in zip(self.hidden_sizes[:-1], self.hidden_sizes[1:])])
        self.output_layer = nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, data):
        """The forward pass of the model
        """
        out = self.input_layer(data)
        for _, fc_ in enumerate(self.fc_layers):
            out = fc_(out)
            out = self.activation(out)
        out = self.output_layer(out)
        return out
    

