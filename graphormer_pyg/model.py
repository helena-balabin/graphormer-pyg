from typing import Union

import torch
from torch import nn
from torch_geometric.data import Data

from graphormer_pyg.layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding


class Graphormer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_node_dim: int,
                 node_dim: int,
                 input_edge_dim: int,
                 edge_dim: int,
                 output_dim: int,
                 n_heads: int,
                 ff_dim: int,
                 max_in_degree: int,
                 max_out_degree: int,
                 max_path_distance: int,
                 use_vnode: bool = True):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        :param use_vnode: whether to use virtual node (graph token) for graph-level representation
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance
        self.use_vnode = use_vnode

        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)

        # Virtual node (graph token) for graph-level representation
        if self.use_vnode:
            self.graph_token = nn.Embedding(1, self.node_dim)

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                n_heads=self.n_heads,
                ff_dim=self.ff_dim,
                max_path_distance=self.max_path_distance) for _ in range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def forward(self, data: Union[Data]) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings (including virtual node if use_vnode=True)
        """
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        if type(data) == Data:
            ptr = None
            num_graphs = 1
        else:
            ptr = data.ptr
            num_graphs = len(ptr) - 1
        
        # Get node_paths_length from data (must be provided during preprocessing)
        node_paths_length = data.node_paths_length

        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        # Add virtual node (graph token) if enabled
        if self.use_vnode:
            # Create graph token for each graph in the batch
            graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(num_graphs, 1, 1)
            # Reshape to [num_graphs * 1, node_dim]
            graph_token_feature = graph_token_feature.view(num_graphs, self.node_dim)
            
            # Insert graph tokens at the beginning of each graph
            if ptr is not None:
                # For batched graphs, insert one token per graph
                x_with_vnode = []
                for i in range(num_graphs):
                    start_idx = ptr[i]
                    end_idx = ptr[i + 1]
                    graph_nodes = x[start_idx:end_idx]
                    x_with_vnode.append(torch.cat([graph_token_feature[i:i+1], graph_nodes], dim=0))
                x = torch.cat(x_with_vnode, dim=0)
                
                # Update ptr to account for virtual nodes
                ptr = torch.tensor([ptr[i] + i for i in range(len(ptr))], device=ptr.device, dtype=ptr.dtype)
                
                # Update in_degree and out_degree to have 0 for virtual nodes
                vnode_degrees = torch.zeros(num_graphs, device=data.in_degree.device, dtype=data.in_degree.dtype)
                in_degree_with_vnode = []
                out_degree_with_vnode = []
                for i in range(num_graphs):
                    start_idx = data.ptr[i]
                    end_idx = data.ptr[i + 1]
                    in_degree_with_vnode.append(torch.cat([vnode_degrees[i:i+1], data.in_degree[start_idx:end_idx]], dim=0))
                    out_degree_with_vnode.append(torch.cat([vnode_degrees[i:i+1], data.out_degree[start_idx:end_idx]], dim=0))
                in_degree = torch.cat(in_degree_with_vnode, dim=0)
                out_degree = torch.cat(out_degree_with_vnode, dim=0)
            else:
                # Single graph: just prepend the token
                x = torch.cat([graph_token_feature, x], dim=0)
                # Add degree 0 for virtual node
                vnode_degree = torch.zeros(1, device=data.in_degree.device, dtype=data.in_degree.dtype)
                in_degree = torch.cat([vnode_degree, data.in_degree], dim=0)
                out_degree = torch.cat([vnode_degree, data.out_degree], dim=0)
        else:
            in_degree = data.in_degree
            out_degree = data.out_degree

        x = self.centrality_encoding(x, in_degree, out_degree)
        
        # Expand spatial encoding to include virtual node if needed
        if self.use_vnode:
            # Create expanded spatial position matrix with virtual node
            num_nodes_with_vnode = x.shape[0]
            device = node_paths_length.device
            
            if ptr is not None:
                # For batched graphs, need to insert virtual node positions
                expanded_spatial = torch.zeros((num_nodes_with_vnode, num_nodes_with_vnode), 
                                              dtype=node_paths_length.dtype, device=device)
                
                offset_old = 0
                offset_new = 0
                for i in range(num_graphs):
                    old_start = data.ptr[i]
                    old_end = data.ptr[i + 1]
                    num_nodes_graph = old_end - old_start
                    
                    new_start = ptr[i]
                    new_end = ptr[i + 1]
                    
                    # Virtual node to virtual node (same graph) = 0
                    expanded_spatial[new_start, new_start] = 0
                    
                    # Virtual node to real nodes = 1 (direct connection assumed)
                    expanded_spatial[new_start, new_start+1:new_end] = 1
                    expanded_spatial[new_start+1:new_end, new_start] = 1
                    
                    # Real nodes to real nodes (copy from original)
                    expanded_spatial[new_start+1:new_end, new_start+1:new_end] = \
                        node_paths_length[old_start:old_end, old_start:old_end]
                
                node_paths_length = expanded_spatial
            else:
                # Single graph case
                num_nodes_orig = node_paths_length.shape[0]
                expanded_spatial = torch.zeros((num_nodes_orig + 1, num_nodes_orig + 1), 
                                              dtype=node_paths_length.dtype, device=device)
                # Virtual node to virtual node = 0
                expanded_spatial[0, 0] = 0
                # Virtual node to all real nodes = 1
                expanded_spatial[0, 1:] = 1
                expanded_spatial[1:, 0] = 1
                # Real nodes to real nodes (copy from original)
                expanded_spatial[1:, 1:] = node_paths_length
                node_paths_length = expanded_spatial
        
        # Get spatial encoding
        b = self.spatial_encoding(x, node_paths_length)

        # Get edge paths information if available
        edge_paths_tensor = getattr(data, 'edge_paths_tensor', None)
        edge_paths_length = getattr(data, 'edge_paths_length', None)
        
        # Expand edge paths to include virtual node if needed
        if self.use_vnode and edge_paths_tensor is not None and edge_paths_length is not None:
            num_nodes_with_vnode = x.shape[0]
            device = edge_paths_tensor.device
            
            if ptr is not None:
                # For batched graphs
                expanded_edge_tensor = torch.full((num_nodes_with_vnode, num_nodes_with_vnode, self.max_path_distance), 
                                                 -1, dtype=edge_paths_tensor.dtype, device=device)
                expanded_edge_length = torch.zeros((num_nodes_with_vnode, num_nodes_with_vnode), 
                                                   dtype=edge_paths_length.dtype, device=device)
                
                for i in range(num_graphs):
                    old_start = data.ptr[i]
                    old_end = data.ptr[i + 1]
                    new_start = ptr[i]
                    new_end = ptr[i + 1]
                    
                    # Real nodes to real nodes (copy from original)
                    expanded_edge_tensor[new_start+1:new_end, new_start+1:new_end] = \
                        edge_paths_tensor[old_start:old_end, old_start:old_end]
                    expanded_edge_length[new_start+1:new_end, new_start+1:new_end] = \
                        edge_paths_length[old_start:old_end, old_start:old_end]
                
                edge_paths_tensor = expanded_edge_tensor
                edge_paths_length = expanded_edge_length
            else:
                # Single graph case
                num_nodes_orig = edge_paths_tensor.shape[0]
                expanded_edge_tensor = torch.full((num_nodes_orig + 1, num_nodes_orig + 1, self.max_path_distance), 
                                                 -1, dtype=edge_paths_tensor.dtype, device=device)
                expanded_edge_length = torch.zeros((num_nodes_orig + 1, num_nodes_orig + 1), 
                                                   dtype=edge_paths_length.dtype, device=device)
                
                # Real nodes to real nodes (copy from original)
                expanded_edge_tensor[1:, 1:] = edge_paths_tensor
                expanded_edge_length[1:, 1:] = edge_paths_length
                
                # Virtual node has no edge paths (all remain -1 and 0)
                edge_paths_tensor = expanded_edge_tensor
                edge_paths_length = expanded_edge_length

        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths_tensor, edge_paths_length, ptr)

        x = self.node_out_lin(x)

        return x

    def get_graph_repr(self, x: torch.Tensor, ptr: torch.Tensor = None) -> torch.Tensor:
        """
        Extract graph-level representation from node embeddings.
        If use_vnode=True, extracts the virtual node embeddings.
        Otherwise, performs mean pooling over node embeddings.
        
        :param x: node embeddings from forward pass, shape [num_nodes, node_dim]
        :param ptr: batch pointer for batched graphs
        :return: graph-level embeddings, shape [num_graphs, node_dim]
        """
        if self.use_vnode:
            # Extract virtual nodes (first node of each graph)
            if ptr is not None:
                # Batched case: extract first node of each graph
                num_graphs = len(ptr) - 1
                graph_reprs = []
                for i in range(num_graphs):
                    # Virtual node is at index ptr[i] (first node of each graph)
                    graph_reprs.append(x[ptr[i]:ptr[i]+1])
                return torch.cat(graph_reprs, dim=0)
            else:
                # Single graph: virtual node is at index 0
                return x[0:1]
        else:
            # Mean pooling over all nodes
            if ptr is not None:
                num_graphs = len(ptr) - 1
                graph_reprs = []
                for i in range(num_graphs):
                    start_idx = ptr[i]
                    end_idx = ptr[i + 1]
                    graph_reprs.append(x[start_idx:end_idx].mean(dim=0, keepdim=True))
                return torch.cat(graph_reprs, dim=0)
            else:
                return x.mean(dim=0, keepdim=True)
