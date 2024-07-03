import numpy as np
from lips.augmented_simulators import AugmentedSimulator
from tensorflow import keras

class TfSimulator(AugmentedSimulator):
    """Simulator class that allows to train and evalute your custom model

        Parameters
        ----------
        benchmark : PowerGridBenchmark
            A benchmark object passed inside the ingestion program
        config : ConfigManager
            A lips ConfigManager object allowing the access to all the parameters in `config.ini`
        scaler : StandardScaler
            A scaler class already implemented in LIPS and selected in parameters.json file
        device : torch.device
            not to use for tensorflow based implementations
        **kwargs
            The set of supplementary parameters passed through the parameters.json file and `simulator_extra_parameters` key
        
        """        
    def __init__(self,
                 benchmark,
                 config,
                 scaler,
                 device=None,
                 **kwargs):
        
        ## You can use this function to infer the inputs and outputs size or giving directly the sizes
        self.input_size, self.output_size = infer_input_output_size(benchmark.train_dataset)
        
        self.params = config.get_options_dict() # Load parameters from config.ini file
        ## Paramaters can be passsed through "simulator_extra_parameters" in the parameters.json file
        self.params.update(kwargs) # update parameters with user defined `simulator_extra_parameters` parameters
        self.name = self.params["name"]
        self.hidden_sizes = self.params["layers"] 
        
        if scaler is not None:
            self.scaler = scaler()
        else:
            self.scaler = None
        self._model = None

    def build_model(self):
        input_ = keras.layers.Input(shape=(self.input_size,), name="input")
        x = input_
        for layer_id, layer_size in enumerate(self.hidden_sizes):
            x = keras.layers.Dense(layer_size, name=f"layer_{layer_id}")(x)
            x = keras.layers.Activation("relu", name=f"activation_{layer_id}")(x)
        output_ = keras.layers.Dense(self.output_size)(x)
        self._model = keras.Model(inputs=input_,
                                  outputs=output_,
                                  name=f"{self.name}_model")     

    def train(self,
              train_dataset,
              val_dataset,
              **kwargs
              ):
        processed_x_train, processed_y_train = process_dataset(train_dataset, training=True, scaler=self.scaler)
        processed_x_val, processed_y_val = process_dataset(val_dataset, training=False, scaler=self.scaler)
        validation_data = (processed_x_val, processed_y_val)
        
        params = kwargs 
        
        # init the model
        self.build_model()
        optimizer = keras.optimizers.Adam(learning_rate=params["lr"])
        self._model.compile(optimizer=optimizer,
                            loss="mse",
                            metrics=["mae"])        

        history_callback = self._model.fit(x=processed_x_train,
                                           y=processed_y_train,
                                           validation_data=validation_data,
                                           epochs=params["epochs"],
                                           batch_size=params["batch_size"],
                                           shuffle=True)
        return history_callback

    def predict(self, dataset, eval_batch_size=128) -> dict:
        processed_x, _ = process_dataset(dataset, training=False, scaler=self.scaler)

        # make the predictions
        predictions = self._model.predict(processed_x, batch_size=eval_batch_size)

        predictions = post_process(dataset, predictions)

        return predictions

def process_dataset(dataset, training: bool=False, scaler=None) -> tuple:
    if training:
        inputs, outputs = dataset.extract_data(concat=True)
        if scaler is not None:
            inputs, outputs = scaler.fit_transform(inputs, outputs)
    else:
        inputs, outputs = dataset.extract_data(concat=True)
        if scaler is not None:
            inputs, outputs = scaler.transform(inputs, outputs)

    return inputs, outputs

def infer_input_output_size(dataset):
    *dim_inputs, output_size = dataset.get_sizes()
    input_size = np.sum(dim_inputs)
    return input_size, output_size

def post_process(dataset, predictions, scaler=None):
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
    predictions = dataset.reconstruct_output(predictions)
    return predictions
