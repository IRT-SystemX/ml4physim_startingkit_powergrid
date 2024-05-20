import copy
import math
import itertools
import warnings

import numpy.typing as npt
import numpy as np
from tqdm.auto import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing


import grid2op
from grid2op import Observation
from grid2op.Parameters import Parameters
from lightsim2grid.lightSimBackend import LightSimBackend

from lips.dataset import DataSet

def get_obs(benchmark):
    params = Parameters()
    params.ENV_DC = True
    env = grid2op.make(benchmark.env_name, param=params, backend=LightSimBackend())
    obs = env.reset()
    return env, obs

def get_features_per_sub(obs: Observation, 
                         dataset: DataSet, 
                         index: int=0) -> npt.NDArray[np.float32]:
    """It returns the features at substation level without aggregation

    Parameters
    ----------
    obs : ``Observation``
        Grid2op observation
    dataset : ``DataSet``
        _description_
    index : ``int``, optional
        A specific index of dataset for which the features should be extracted, by default 0

    Returns
    -------
    npt.NDArray[np.float32]
        the feature array for one observation of dataset
    """
    feature_matrix = np.zeros((obs.n_sub, 2), dtype=np.float32)
    for sub_ in range(obs.n_sub):
        objects = obs.get_obj_connect_to(substation_id=sub_)

        if len(objects["generators_id"]) > 0:
            feature_matrix[sub_, 0] += np.sum(dataset.get("prod_p")[index, objects["generators_id"]])
        if len(objects["loads_id"]) > 0:
            feature_matrix[sub_,1] += np.sum(dataset.get("load_p")[index, objects["loads_id"]])
        
    return feature_matrix

def get_all_features_per_sub(obs: Observation, dataset: DataSet) -> torch.Tensor:
    """Get all the features from dataset without their aggregation

    Parameters
    ----------
    obs : Observation
        Grid2op observation used for some functionalities
    dataset : DataSet
        LIPS dataset from which the features should be extracted

    Returns
    -------
    torch.Tensor
        Torch tensor including the features that should be used as the inputs for a model
    """
    features = torch.zeros((len(dataset["prod_p"]), obs.n_sub, 2))
    for i in range(len(features)):
        features[i, :, :] = torch.tensor(get_features_per_sub(obs, dataset, index=i))
    return features.float()

def get_theta_node(obs, sub_id, bus):
    obj_to_sub = obs.get_obj_connect_to(substation_id=sub_id)

    lines_or_to_sub_bus = [i for i in obj_to_sub['lines_or_id'] if obs.line_or_bus[i] == bus]
    lines_ex_to_sub_bus = [i for i in obj_to_sub['lines_ex_id'] if obs.line_ex_bus[i] == bus]

    thetas_node = np.append(obs.theta_or[lines_or_to_sub_bus], obs.theta_ex[lines_ex_to_sub_bus])
    thetas_node = thetas_node[thetas_node != 0]

    theta_node = 0.
    if len(thetas_node) != 0:
        theta_node = np.max(thetas_node)

    return theta_node

def create_fake_obs(obs, data, idx = 0):
    """Create a fake observation from the env by copying data values

    Args:
        obs (_type_): _description_
        data (_type_): _description_
        idx (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    obs.line_status = data["line_status"][idx]
    obs.topo_vect = data["topo_vect"][idx]
    return obs

def get_theta_bus(dataset, obs):
    """
    Function to compute complex voltages at bus
    """
    Ybus = dataset["YBus"]
    bus_theta = np.zeros((Ybus.shape[0], obs.n_sub), dtype=complex)
    for idx in range(Ybus.shape[0]):
        obs = create_fake_obs(obs, dataset, idx)
        obs.theta_or = dataset["theta_or"][idx]
        obs.theta_ex = dataset["theta_ex"][idx]
        for sub_ in range(obs.n_sub):
            bus_theta[idx, sub_] = get_theta_node(obs, sub_id=sub_, bus=1)
    return bus_theta

def get_target_variables_per_sub(obs: Observation, dataset: DataSet) -> torch.Tensor:
    """Gets the target variables which should be predicted by a model

    Parameters
    ----------
    obs : Observation
        Grid2op observation
    dataset : DataSet
        LIPS dataset from which the target variable(s) could be extracted

    Returns
    -------
    torch.Tensor
        Tensor including the target(s)
    """
    targets = torch.tensor(get_theta_bus(dataset, obs).real).unsqueeze(dim=2)
    return targets.float()

def get_edge_index_from_ybus(ybus_matrix, add_loops=True) -> list:
    """Get all the edge_indices from Ybus matrix

    Parameters
    ----------
    ybus_matrix : _type_
        Ybus matrix as input (NxMxM)
        with N number of observations
        and M number of nodes in the graph

    Returns
    -------
    ``list``
        a list of edge indices
    """
    ybus_mat = copy.deepcopy(ybus_matrix)
    edge_indices = []
    for ybus in ybus_mat:
        if not(add_loops):
            np.fill_diagonal(ybus, val=0.)
        bus_or, bus_ex = np.where(ybus)
        edge_index = np.column_stack((bus_or, bus_ex)).T
        edge_indices.append(edge_index)
    return edge_indices

def get_edge_weights_from_ybus(ybus_matrix, edge_indices) -> list:
    """Get edge weights corresponding to each edge index

    Parameters
    ----------
    ybus_matrix : _type_
        _description_
    edge_indices : _type_
        edge indices returned by the get_edge_index_from_ybus function

    Returns
    -------
    ``list``
        a list of edge weights
    """
    edge_weights = []
    for edge_index, ybus in zip(edge_indices, ybus_matrix):
        edge_weight = []
        for i in range(edge_index.shape[1]):
            edge_weight.append(ybus[edge_index[0][i], edge_index[1][i]])
        edge_weight = np.array(edge_weight)
        edge_weights.append(edge_weight)
    return edge_weights

def get_batches_pyg(edge_indices,
                    edge_indices_no_diag,
                    features,
                    targets,
                    ybuses,
                    batch_size=128,
                    device="cpu",
                    edge_weights=None,
                    edge_weights_no_diag=None):
    torchDataset = []
    for i, feature in enumerate(features):
        if edge_weights is not None:
            edge_weight = torch.tensor(edge_weights[i], dtype=feature.dtype)
            edge_weight_no_diag = torch.tensor(edge_weights_no_diag[i], dtype=feature.dtype)
        else:
            edge_weight=None
        sample_data = Data(x=feature,
                           y=targets[i],
                           edge_index=torch.tensor(edge_indices[i]),
                           edge_index_no_diag=torch.tensor(edge_indices_no_diag[i]),
                           edge_attr=edge_weight,
                           edge_attr_no_diag=edge_weight_no_diag,
                           ybus=torch.tensor(ybuses[i][:14,:14].real))
        sample_data.to(device)
        torchDataset.append(sample_data)
    loader = DataLoader(torchDataset, batch_size=batch_size)

    return loader

def prepare_dataset(benchmark, batch_size=128, device="cpu"):
    warnings.filterwarnings("ignore")
    params = Parameters()
    params.ENV_DC = True
    env = grid2op.make(benchmark.env_name, param=params, backend=LightSimBackend())
    obs = env.reset()
    
    # Train dataset
    print("*******Train dataset*******")
    print(f"Train data size: {benchmark.train_dataset.size}")
    pbar = tqdm(range(4))
    pbar.set_description("Get Features")
    train_features = get_all_features_per_sub(obs, benchmark.train_dataset.data) # dim [100000, 14, 2]
    pbar.update(1)
    pbar.set_description("Get Targets")
    train_targets = get_target_variables_per_sub(obs, benchmark.train_dataset.data)
    pbar.update(1)
    pbar.set_description("Get edge_index info")
    train_edge_indices = get_edge_index_from_ybus(benchmark.train_dataset.data["YBus"])
    train_edge_weights = get_edge_weights_from_ybus(benchmark.train_dataset.data["YBus"], train_edge_indices)
    train_edge_indices_no_diag = get_edge_index_from_ybus(benchmark.train_dataset.data["YBus"], add_loops=False)
    train_edge_weights_no_diag = get_edge_weights_from_ybus(benchmark.train_dataset.data["YBus"], train_edge_indices_no_diag)
    pbar.update(1)
    pbar.set_description("Create loader")
    train_loader = get_batches_pyg(edge_indices=train_edge_indices,
                                   edge_indices_no_diag=train_edge_indices_no_diag,
                                   features=train_features,
                                   targets=train_targets,
                                   ybuses=benchmark.train_dataset.data["YBus"],
                                   edge_weights=train_edge_weights,
                                   edge_weights_no_diag=train_edge_weights_no_diag,
                                   batch_size=batch_size,
                                   device=device)
    pbar.update(1)
    pbar.close()
    # Val dataset
    print("*******Validation dataset*******")
    print(f"Validation data size: {benchmark.val_dataset.size}")
    pbar = tqdm(range(4))
    pbar.set_description("Get Features")
    val_features = get_all_features_per_sub(obs, benchmark.val_dataset.data) # dim [10000, 14, 2]
    pbar.update(1)
    pbar.set_description("Get Targets")
    val_targets = get_target_variables_per_sub(obs, benchmark.val_dataset.data)
    pbar.update(1)
    pbar.set_description("Get edge_index info")
    val_edge_indices = get_edge_index_from_ybus(benchmark.val_dataset.data["YBus"])
    val_edge_weights = get_edge_weights_from_ybus(benchmark.val_dataset.data["YBus"], val_edge_indices)
    val_edge_indices_no_diag = get_edge_index_from_ybus(benchmark.val_dataset.data["YBus"], add_loops=False)
    val_edge_weights_no_diag = get_edge_weights_from_ybus(benchmark.val_dataset.data["YBus"], val_edge_indices_no_diag)
    pbar.update(1)
    pbar.set_description("Create loader")
    val_loader = get_batches_pyg(edge_indices=val_edge_indices,
                                 edge_indices_no_diag=val_edge_indices_no_diag,
                                 features=val_features,
                                 targets=val_targets,
                                 ybuses=benchmark.val_dataset.data["YBus"],
                                 edge_weights=val_edge_weights,
                                 edge_weights_no_diag=val_edge_weights_no_diag,
                                 batch_size=batch_size,
                                 device=device)
    pbar.update(1)
    pbar.close()
    
    # Test dataset
    print("*******Test dataset*******")
    print(f"Test data size : {benchmark._test_dataset.size}")
    pbar = tqdm(range(4))
    pbar.set_description("Get Features")
    test_features = get_all_features_per_sub(obs, benchmark._test_dataset.data) # dim [10000, 14, 2]
    pbar.update(1)
    pbar.set_description("Get Targets")
    test_targets = get_target_variables_per_sub(obs, benchmark._test_dataset.data)
    pbar.update(1)
    pbar.set_description("Get edge_index info")
    test_edge_indices = get_edge_index_from_ybus(benchmark._test_dataset.data["YBus"])
    test_edge_weights = get_edge_weights_from_ybus(benchmark._test_dataset.data["YBus"], test_edge_indices)
    test_edge_indices_no_diag = get_edge_index_from_ybus(benchmark._test_dataset.data["YBus"], add_loops=False)
    test_edge_weights_no_diag = get_edge_weights_from_ybus(benchmark._test_dataset.data["YBus"], test_edge_indices_no_diag)
    pbar.update(1)
    pbar.set_description("Create loader")
    test_loader = get_batches_pyg(edge_indices=test_edge_indices,
                                  edge_indices_no_diag=test_edge_indices_no_diag,
                                  features=test_features,
                                  targets=test_targets,
                                  ybuses=benchmark._test_dataset.data["YBus"],
                                  edge_weights=test_edge_weights,
                                  edge_weights_no_diag=test_edge_weights_no_diag,
                                  batch_size=batch_size,
                                  device=device)
    pbar.update(1)
    pbar.close()
    # OOD dataset
    print("*******OOD dataset*******")
    print(f"OOD data size: {benchmark._test_ood_topo_dataset.size}")
    pbar = tqdm(range(4))
    pbar.set_description("Get Features")
    test_ood_features = get_all_features_per_sub(obs, benchmark._test_ood_topo_dataset.data)
    pbar.update(1)
    pbar.set_description("Get Targets")
    test_ood_targets = get_target_variables_per_sub(obs, benchmark._test_ood_topo_dataset.data)
    pbar.update(1)
    pbar.set_description("Get edge_index info")
    test_ood_edge_indices = get_edge_index_from_ybus(benchmark._test_ood_topo_dataset.data["YBus"])
    test_ood_edge_weights = get_edge_weights_from_ybus(benchmark._test_ood_topo_dataset.data["YBus"], test_ood_edge_indices)
    test_ood_edge_indices_no_diag = get_edge_index_from_ybus(benchmark._test_ood_topo_dataset.data["YBus"], add_loops=False)
    test_ood_edge_weights_no_diag = get_edge_weights_from_ybus(benchmark._test_ood_topo_dataset.data["YBus"], test_ood_edge_indices_no_diag)
    pbar.update(1)
    pbar.set_description("Create loader")
    test_ood_loader = get_batches_pyg(edge_indices=test_ood_edge_indices,
                                            edge_indices_no_diag=test_ood_edge_indices_no_diag,
                                            features=test_ood_features,
                                            targets=test_ood_targets,
                                            ybuses=benchmark._test_ood_topo_dataset.data["YBus"],
                                            edge_weights=test_ood_edge_weights,
                                            edge_weights_no_diag=test_ood_edge_weights_no_diag,
                                            batch_size=batch_size,
                                            device=device)
    pbar.update(1)
    pbar.close()

    return train_loader, val_loader, test_loader, test_ood_loader

class GPGinput_without_NN(MessagePassing):
    def __init__(self,
                 device="cpu"
                 ):
        super().__init__(aggr="add")
        self.theta = None
        self.device = device

    def forward(self, batch):

        self.theta = torch.zeros_like(batch.y, dtype=batch.y.dtype)

        aggr_msg = self.propagate(batch.edge_index_no_diag,
                                  y=self.theta,
                                  edge_weights=batch.edge_attr_no_diag * 100.0
                                 )
        
        # keep only the diagonal elements of the ybus 3D tensors
        ybus = batch.ybus.view(-1, 14, 14) * 100.0
        ybus = ybus * torch.eye(*ybus.shape[-2:], device=self.device).repeat(ybus.shape[0], 1, 1)
        denominator = ybus[ybus.nonzero(as_tuple=True)].view(-1,1)
        
        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        out = (input_node_power - aggr_msg) / denominator

        #we impose that node 0 has theta=0
        out = out.view(-1, 14, 1) - out.view(-1,14,1)[:,0].repeat_interleave(14, 1).view(-1, 14, 1)
        out = out.flatten().view(-1,1)
        
        return out, aggr_msg
    
    def message(self, y_j, edge_weights):
        tmp = y_j * edge_weights.view(-1,1)
        return tmp
    
    def update(self, aggr_out):
        return aggr_out
    
class GPGintermediate(MessagePassing):
    def __init__(self,
                 device="cpu"):
        super().__init__(aggr="add")
        self.theta = None
        self.device = device
        
    
    def forward(self, batch, theta):
        self.theta = theta
        
        aggr_msg = self.propagate(batch.edge_index_no_diag,
                                  y=self.theta,
                                  edge_weights=batch.edge_attr_no_diag * 100.0
                                 )

        # keep only the diagonal elements of the ybus 3D tensors for denominator part
        ybus = batch.ybus.view(-1, 14, 14) * 100.0
        ybus = ybus * torch.eye(*ybus.shape[-2:], device=self.device).repeat(ybus.shape[0], 1, 1)
        denominator = ybus[ybus.nonzero(as_tuple=True)].view(-1,1)

        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        out = (input_node_power - aggr_msg) / denominator

        #we impose that node 0 has theta=0
        out = out.view(-1, 14, 1) - out.view(-1,14,1)[:,0].repeat_interleave(14, 1).view(-1, 14, 1)
        out = out.flatten().view(-1,1)
        
        return out, aggr_msg

    def message(self, y_i, y_j, edge_weights):
        tmp = y_j * edge_weights.view(-1,1)
        return tmp

    def update(self, aggr_out):
        return aggr_out

class LocalConservationLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")
        self.thetas = None
        
    def forward(self, batch, thetas=None):
        self.thetas = thetas

        aggr_message = self.propagate(batch.edge_index,
                                      y=self.thetas,
                                      edge_weights=batch.edge_attr * 100)

        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        nodal_error = input_node_power - aggr_message

        return nodal_error

    def message(self, y_i, y_j, edge_weights):
        tmp = y_j * edge_weights.view(-1,1)
        return tmp

class GPGmodel_without_NN(torch.nn.Module):
    def __init__(self,
                 num_gnn_layers=10,
                 device="cpu"):
        super().__init__()
        self.num_gnn_layers = num_gnn_layers
        self.device = device

        self.input_layer = None
        self.lc_layer = None
        self.inter_layers = None

        self.build_model()

    def build_model(self):
        self.input_layer = GPGinput_without_NN(device=self.device)
        self.lc_layer = LocalConservationLayer()
        self.inter_layers = torch.nn.ModuleList([GPGintermediate(device=self.device) for _ in range(self.num_gnn_layers)])

    def forward(self, batch):
        errors = []
        out, _ = self.input_layer(batch)
        nodal_error = self.lc_layer(batch, out)
        errors.append(abs(nodal_error).sum())
        
        for gnn_layer, lc_layer_ in zip(self.inter_layers, itertools.repeat(self.lc_layer)):
            out, _ = gnn_layer(batch, out)
            nodal_error = lc_layer_(batch, out)
            errors.append(abs(nodal_error).sum())

        return out, errors

def get_active_power(dataset, obs, theta, index):
    """Computes the active power (flows) from thetas (subs) for an index

    Parameters
    ----------
    dataset : _type_
        _description_
    obs : _type_
        _description_
    theta : _type_
        _description_
    index : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    lor_bus, lor_conn = obs._get_bus_id(obs.line_or_pos_topo_vect, obs.line_or_to_subid)
    lex_bus, lex_conn = obs._get_bus_id(obs.line_ex_pos_topo_vect, obs.line_ex_to_subid)
    index_array = np.vstack((np.arange(obs.n_line), lor_bus, lex_bus)).T 
    # Create the adjacency matrix (MxN) M: branches and N: Nodes
    A_or = np.zeros((obs.n_line, obs.n_sub))
    A_ex = np.zeros((obs.n_line, obs.n_sub))

    for line in index_array[:,0]:
        if index_array[line,1] != -1:
            A_or[line, index_array[line,1]] = 1
            A_or[line, index_array[line,2]] = -1
            A_ex[line, index_array[line,1]] = -1
            A_ex[line, index_array[line,2]] = 1        
    
    # Create the diagonal matrix D (MxM)
    Ybus = dataset["YBus"][index][:obs.n_sub,:obs.n_sub]
    D = np.zeros((obs.n_line, obs.n_line), dtype=complex)
    for line in index_array[:, 0]:
        bus_from = index_array[line, 1]
        bus_to = index_array[line, 2]
        D[line,line] = Ybus[bus_from, bus_to] * (-1)

    # Create the theta vector ((M-1)x1)
    theta = 1j*((theta[index,1:]*math.pi)/180)
    p_or = (D.dot(A_or[:,1:])).dot(theta.reshape(-1,1))
    p_ex = (D.dot(A_ex[:,1:])).dot(theta.reshape(-1,1))

    #return p_or, p_ex
    return p_or.imag * 100 , p_ex.imag * 100

def get_all_active_powers(dataset, obs, theta_bus):
    """Computes all the active powers for all the observations from theta at bus

    Parameters
    ----------
    dataset : _type_
        _description_
    obs : _type_
        _description_
    theta_bus : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    data_size = len(dataset["p_or"])
    p_or = np.zeros_like(dataset["p_or"])
    p_ex = np.zeros_like(dataset["p_ex"])

    #theta_bus = get_theta_bus(dataset, obs)
    for ind in tqdm(range(data_size)):
        obs = create_fake_obs(obs, dataset, ind)
        p_or_computed, p_ex_computed = get_active_power(dataset, obs, theta_bus, index=ind)
        p_or[ind, :] = p_or_computed.flatten()
        p_ex[ind, :] = p_ex_computed.flatten()
    
    return p_or, p_ex
