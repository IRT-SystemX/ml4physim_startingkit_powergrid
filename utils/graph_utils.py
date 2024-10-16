import copy
import math
import itertools
import warnings

import numpy.typing as npt
from scipy.sparse import csr_matrix
import numpy as np
from tqdm.auto import tqdm

import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing


import grid2op
from grid2op import Observation
from grid2op.Parameters import Parameters
from lightsim2grid.lightSimBackend import LightSimBackend

from lips.dataset import DataSet

def visualize_graph(edge_index, 
                    node_color="orange", 
                    node_size=1e3,
                    edge_labels=None,
                    font_size=8,
                    edge_width=0.7,
                    edge_alpha=0.8,
                    figsize=(18,12),
                    seed=2,
                    save_fig=False
                    ):
    """Visualize the graph using networkx

    Parameters
    ----------
    edge_index : ``list``
        list containing the pair of nodes indicating the presence of edges
    node_color : str, optional
        _description_, by default "orange"
    node_size : _type_, optional
        _description_, by default 1e3
    font_size : int, optional
        _description_, by default 8
    edge_width : float, optional
        _description_, by default 0.7
    edge_alpha : float, optional
        _description_, by default 0.8
    figsize : tuple, optional
        _description_, by default (18,12)
    seed : int, optional
        _description_, by default 2
    save_fig : bool, optional
        _description_, by default False
    """
    g = nx.Graph()
    g.add_edges_from(edge_index)
    labels = {i:f"{i}" for i in g.nodes}

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(g, seed=seed)
    nx.draw_networkx_nodes(g, pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=font_size)
    nx.draw_networkx_edges(g, pos, width=edge_width, alpha=edge_alpha)
    if edge_labels is not None:
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=font_size)
    if save_fig:
        pass
    else:
        plt.show()

    return g

def visualize_batch(batch):
    """Visualize a batch of graphs at once

    Parameters
    ----------
    batch : _type_
        A batch of torch-geometric `DataLoader`
    """
    g = pyg.utils.to_networkx(batch, to_undirected=True)
    labels = {i:f"{i}" for i, _ in enumerate(batch.x)}

    plt.figure(figsize=(18,12))
    pos = nx.spring_layout(g, seed=2)
    nx.draw_networkx_nodes(g, pos, node_color="orange", node_size=1e2)
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)
    nx.draw_networkx_edges(g, pos, width=0.7, alpha=0.8)
    #nx.draw_networkx_edge_labels(g, pos)
    plt.show()
    
def get_obs(benchmark):
    """Get an observation of a Grid2op environment contained in a benchmark object

    Args:
        benchmark (_type_): LIPS benchmark

    Returns:
        _type_: Grid2op environment and an observation
    """
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
    """Get the voltage angles for a specific substation

    Args:
        obs (_type_): Grid2op observation
        sub_id (int): the idenitifer of a substation
        bus (int): the bus number

    Returns:
        float: returns the voltage angle at the node
    """
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

def get_edge_index_from_ybus(ybus_matrix, obs, add_loops=True) -> list:
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
    # ybus_mat = copy.deepcopy(ybus_matrix)
    edge_indices = []
    #for ybus in ybus_mat:
    for ybus in ybus_matrix:
        ybus_cpy = copy.deepcopy(ybus)
        if isinstance(ybus_cpy, csr_matrix):
            ybus_cpy = np.squeeze(np.asarray(ybus_cpy.todense()))
            ybus_cpy = ybus_cpy.reshape(obs.n_sub*2, obs.n_sub*2)
        if not(add_loops):
            np.fill_diagonal(ybus_cpy, val=0.)
        bus_or, bus_ex = np.where(ybus_cpy)
        edge_index = np.column_stack((bus_or, bus_ex)).T
        edge_indices.append(edge_index)
    return edge_indices

def get_edge_weights_from_ybus(ybus_matrix, edge_indices, obs) -> list:
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
        if isinstance(ybus, csr_matrix):
            ybus = np.squeeze(np.asarray(ybus.todense()))
            ybus = ybus.reshape(obs.n_sub*2, obs.n_sub*2)
        for i in range(edge_index.shape[1]):
            edge_weight.append(ybus[edge_index[0][i], edge_index[1][i]])
        edge_weight = np.array(edge_weight)
        edge_weights.append(edge_weight)
    return edge_weights

def get_batches_pyg(obs,
                    edge_indices,
                    edge_indices_no_diag,
                    features,
                    targets,
                    ybuses,
                    batch_size=128,
                    device="cpu",
                    edge_weights=None,
                    edge_weights_no_diag=None):
    """Create Pytorch Geometric based data loaders
    This function gets the features and targets at nodes and create batches of structured data

    Args:
        edge_indices (list): list of edge indices 
        edge_indices_no_diag (_type_): list of edge indices without the self loops (without diagonal elements of adjacency matrix)
        features (_type_): list of features (injections)
        targets (_type_): list of targets (theta)
        ybuses (_type_): admittance matrix
        batch_size (int, optional): _description_. Defaults to 128.
        device (str, optional): _description_. Defaults to "cpu".
        edge_weights (_type_, optional): edge weight which is the admittance matrix element between two nodes. Defaults to None.
        edge_weights_no_diag (_type_, optional): edge weight without values for diagonal elements of adjacency matrix. Defaults to None.

    Returns:
        _type_: _description_
    """
    torchDataset = []
    for i, feature in enumerate(features):
        if edge_weights is not None:
            edge_weight = torch.tensor(edge_weights[i], dtype=feature.dtype)
            edge_weight_no_diag = torch.tensor(edge_weights_no_diag[i], dtype=feature.dtype)
        else:
            edge_weight=None
            
        if isinstance(ybuses, csr_matrix):
            ybus = np.squeeze(np.asarray(ybuses[i].todense()))
            ybus = ybus.reshape(obs.n_sub*2, obs.n_sub*2)
        else:
            ybus = ybuses[i]
            
        sample_data = Data(x=feature,
                           y=targets[i],
                           edge_index=torch.tensor(edge_indices[i]),
                           edge_index_no_diag=torch.tensor(edge_indices_no_diag[i]),
                           edge_attr=edge_weight,
                           edge_attr_no_diag=edge_weight_no_diag,
                           ybus=torch.tensor(ybus[:obs.n_sub,:obs.n_sub].real))
        sample_data.to(device)
        torchDataset.append(sample_data)
    loader = DataLoader(torchDataset, batch_size=batch_size)

    return loader

def prepare_dataset(benchmark, batch_size=128, device="cpu"):
    """Prepare the dataset for GNN based model

    Args:
        benchmark (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 128.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """
    warnings.filterwarnings("ignore")
    params = Parameters()
    params.ENV_DC = True
    env = grid2op.make(benchmark.env_name, param=params, backend=LightSimBackend())
    obs = env.reset()
    
    # Train dataset
    print("*******Train dataset*******")
    print(f"Train data size: {benchmark.train_dataset.size}")
    train_loader = get_loader(obs, benchmark.train_dataset.data, batch_size, device)
    # Val dataset
    print("*******Validation dataset*******")
    print(f"Validation data size: {benchmark.val_dataset.size}")
    val_loader = get_loader(obs, benchmark.val_dataset.data, batch_size, device)
    
    # Test dataset
    print("*******Test dataset*******")
    print(f"Test data size : {benchmark._test_dataset.size}")
    test_loader = get_loader(obs, benchmark._test_dataset.data, batch_size, device)

    # OOD dataset
    print("*******OOD dataset*******")
    print(f"OOD data size: {benchmark._test_ood_topo_dataset.size}")
    test_ood_loader = get_loader(obs, benchmark._test_ood_topo_dataset.data, batch_size, device)

    return train_loader, val_loader, test_loader, test_ood_loader

def get_loader(obs, data, batch_size, device):
    """
    This function structures the features, targets, edge_indices and edge weights through a GNN
    point of view and create a data loader for a given dataset.

    Args:
        obs (_type_): an observation of environment
        data (dict): the dataset
        batch_size (int): the batch size considered for data loader
        device (str): the device on which the computation should be performed

    Returns:
        _type_: _description_
    """
    pbar = tqdm(range(4))
    pbar.set_description("Get Features")
    features = get_all_features_per_sub(obs, data)
    pbar.update(1)
    pbar.set_description("Get Targets")
    targets = get_target_variables_per_sub(obs, data)
    pbar.update(1)
    pbar.set_description("Get edge_index info")
    edge_indices = get_edge_index_from_ybus(data["YBus"], obs, add_loops=True)
    edge_weights = get_edge_weights_from_ybus(data["YBus"], edge_indices, obs)
    edge_indices_no_diag = get_edge_index_from_ybus(data["YBus"], obs, add_loops=False)
    edge_weights_no_diag = get_edge_weights_from_ybus(data["YBus"], edge_indices_no_diag, obs)
    pbar.update(1)
    pbar.set_description("Create loader")
    loader = get_batches_pyg(obs,
                             edge_indices=edge_indices,
                             edge_indices_no_diag=edge_indices_no_diag,
                             features=features,
                             targets=targets,
                             ybuses=data["YBus"],
                             edge_weights=edge_weights,
                             edge_weights_no_diag=edge_weights_no_diag,
                             batch_size=batch_size,
                             device=device)
    pbar.update(1)
    pbar.close()

    return loader

class GPGinput_without_NN(MessagePassing):
    """Graph Power Grid Input layer

    This is the input layer of GNN initialize the theta (voltage angles) with zeros and
    updates them through power flow equation

    """
    def __init__(self,
                 ref_node,
                 device="cpu",
                 ):
        super().__init__(aggr="add")
        self.theta = None
        self.device = device
        self.ref_node=ref_node

    def forward(self, batch):
        
        # Initialize the voltage angles (theta) with zeros
        self.theta = torch.zeros_like(batch.y, dtype=batch.y.dtype)

        # Compute a message and propagate it to each node, it does 3 steps
        # 1) It computes a message (Look at the message function below)
        # 2) It propagates the message using an aggregation (sum here)
        # 3) It calls the update function which could be Neural Network
        aggr_msg = self.propagate(batch.edge_index_no_diag,
                                  y=self.theta,
                                  edge_weights=batch.edge_attr_no_diag * 100.0
                                 )
        n_bus = batch.ybus.size()[1]
        # keep only the diagonal elements of the ybus 3D tensors
        ybus = batch.ybus.view(-1, n_bus, n_bus) * 100.0
        ybus = ybus * torch.eye(*ybus.shape[-2:], device=self.device).repeat(ybus.shape[0], 1, 1)
        denominator = ybus[ybus.nonzero(as_tuple=True)].view(-1,1)
        
        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        out = (input_node_power - aggr_msg) / denominator

        #we impose that reference node has theta=0
        out = out.view(-1, n_bus, 1) - out.view(-1,n_bus,1)[:,self.ref_node].repeat_interleave(n_bus, 1).view(-1, n_bus, 1)
        out = out.flatten().view(-1,1)
        
        return out, aggr_msg
    
    def message(self, y_j, edge_weights):
        """Compute the message that should be propagated
        
        This function compute the message (which is the multiplication of theta and 
        admittance matrix elements connecting node i to j)

        Args:
            y_j (_type_): the theta (voltage angle) value at a neighboring node j
            edge_weights (_type_): corresponding edge_weight (admittance matrix element)

        Returns:
            _type_: active powers for each neighboring node
        """
        tmp = y_j * edge_weights.view(-1,1)
        return tmp
    
    def update(self, aggr_out):
        """update function of message passing layers

        We output directly the aggreated message (sum)

        Args:
            aggr_out (_type_): the aggregated message

        Returns:
            _type_: the aggregated message
        """
        return aggr_out
    
class GPGintermediate(MessagePassing):
    """Graph Power Grid intermediate layer

    This is the intermediate layer of GNN that gets the theta from the previous layer and
    updates them through power flow equation
    """
    def __init__(self,
                 ref_node,
                 device="cpu"):
        super().__init__(aggr="add")
        self.ref_node = ref_node
        self.theta = None
        self.device = device
        
    def forward(self, batch, theta):
        """The forward pass of message passing network

        Args:
            batch (_type_): the data batch
            theta (_type_): the voltage angle from the previous layer

        Returns:
            _type_: the updated voltage angles
        """
        self.theta = theta
        
        # Compute a message and propagate it to each node, it does 3 steps
        # 1) It computes a message (Look at the message function below)
        # 2) It propagates the message using an aggregation (sum here)
        # 3) It calls the update function which could be Neural Network
        aggr_msg = self.propagate(batch.edge_index_no_diag,
                                  y=self.theta,
                                  edge_weights=batch.edge_attr_no_diag * 100.0
                                 )

        n_bus = batch.ybus.size()[1]
        # keep only the diagonal elements of the ybus 3D tensors for denominator part
        ybus = batch.ybus.view(-1, n_bus, n_bus) * 100.0
        ybus = ybus * torch.eye(*ybus.shape[-2:], device=self.device).repeat(ybus.shape[0], 1, 1)
        denominator = ybus[ybus.nonzero(as_tuple=True)].view(-1,1)

        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        out = (input_node_power - aggr_msg) / denominator

        #we impose that reference node has theta=0
        out = out.view(-1, n_bus, 1) - out.view(-1,n_bus,1)[:,self.ref_node].repeat_interleave(n_bus, 1).view(-1, n_bus, 1)
        out = out.flatten().view(-1,1)
        
        return out, aggr_msg

    def message(self, y_i, y_j, edge_weights):
        tmp = y_j * edge_weights.view(-1,1)
        return tmp

    def update(self, aggr_out):
        return aggr_out

class LocalConservationLayer(MessagePassing):
    """Compute local conservation error

    This class computes the local conservation error without any update of voltage angles.

    Args:
        MessagePassing (_type_): _description_
    """
    def __init__(self):
        super().__init__(aggr="add")
        self.thetas = None
        
    def forward(self, batch, thetas=None):
        # theta from previous GNN layer
        self.thetas = thetas

        # The difference with GPG layers resides also in propagation which gets the edge_index
        # with self loops (with diagonal elements of adjacency matrix)
        aggr_message = self.propagate(batch.edge_index,
                                      y=self.thetas,
                                      edge_weights=batch.edge_attr * 100)

        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        # compute the local conservation error (at node level)
        nodal_error = input_node_power - aggr_message

        return nodal_error

    def message(self, y_i, y_j, edge_weights):
        """
        Compute the message
        """
        tmp = y_j * edge_weights.view(-1,1)
        return tmp

class GPGmodel_without_NN(torch.nn.Module):
    """Create a Graph Power Grid (GPG) model without learning
    """
    def __init__(self,
                 ref_node,
                 num_gnn_layers=10,
                 device="cpu"):
        super().__init__()
        self.ref_node = ref_node
        self.num_gnn_layers = num_gnn_layers
        self.device = device

        self.input_layer = None
        self.lc_layer = None
        self.inter_layers = None

        self.build_model()

    def build_model(self):
        """Build the GNN message passing model

        It composed of a first input layer and a number of intermediate message passing layers
        These layes interleave with local conservation layers which allow to compute the error
        at the layer level
        """
        self.input_layer = GPGinput_without_NN(ref_node=self.ref_node, device=self.device)
        self.lc_layer = LocalConservationLayer()
        self.inter_layers = torch.nn.ModuleList([GPGintermediate(ref_node=self.ref_node, 
                                                                 device=self.device) 
                                                 for _ in range(self.num_gnn_layers)])

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
    """
    Computes the active power (flows) from thetas (subs) for a specific index

    Parameters
    ----------
    dataset : _type_
        data
    obs : _type_
        Grid2op observation
    theta : _type_
        voltage angle
    index : _type_
        the observation index for which the active powers should be computed  

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
    Ybus = dataset["YBus"][index]
    if isinstance(dataset["YBus"], csr_matrix):
        Ybus = np.squeeze(np.asarray(Ybus.todense()))
        Ybus = Ybus.reshape(obs.n_sub*2, obs.n_sub*2)
    Ybus = Ybus[:obs.n_sub,:obs.n_sub]
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
    """Computes all the active powers 
    
    It computes the active powers for all the observations from thetas (voltage angles) at bus

    Parameters
    ----------
    dataset : _type_
        the data
    obs : _type_
        Grid2op observation
    theta_bus : _type_
        the voltage angles at buses

    Returns
    -------
    _type_
        numpy arrays corresponding to active powers at the origin and extremity side of power lines
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
