from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_mean, scatter_add, scatter_max

    
class SumReadout(nn.Module):
    """Sum readout operator over graphs with variadic sizes."""
    def __init__(self, type="node"):
        self.type = type

    def get_index2graph(self, graph):
        if self.type == "node":
            input2graph = graph.node2graph
        elif self.type == "edge":
            input2graph = graph.edge2graph
        elif self.type == "residue":
            input2graph = graph.residue2graph
        else:
            raise ValueError("Unknown input type `%s` for readout functions" % self.type)
        return input2graph

    def forward(self, graph, input):
        input2graph = self.get_index2graph(graph)
        output = scatter_add(input, input2graph, dim=0, dim_size=graph.batch_size)
        return output


class MeanReadout(nn.Module):
    """Mean readout operator over graphs with variadic sizes."""
    def __init__(self, type="node"):
        self.type = type

    def get_index2graph(self, graph):
        if self.type == "node":
            input2graph = graph.node2graph
        elif self.type == "edge":
            input2graph = graph.edge2graph
        elif self.type == "residue":
            input2graph = graph.residue2graph
        else:
            raise ValueError("Unknown input type `%s` for readout functions" % self.type)
        return input2graph
    
    def forward(self, graph, input):
        input2graph = self.get_index2graph(graph)
        output = scatter_mean(input, input2graph, dim=0, dim_size=graph.batch_size)
        return output


class GeometricRelationalGraphConv(nn.Module):
    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        return message

    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation
        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation)
        update = update.view(graph.num_node, self.num_relation * self.input_dim)
        return update

    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation

        node_in, node_out, relation = graph.edge_list.t()
        node_out = node_out * self.num_relation + relation
        adjacency = torch.sparse_coo_tensor(torch.stack([node_in, node_out]), graph.edge_weight,
                                            (graph.num_node, graph.num_node * graph.num_relation))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = graph.edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            update += edge_update

        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def combine(self, input, update):
        output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden


class SpatialLineGraph(nn.Module):
    def __init__(self, num_angle_bin=8):
        super(SpatialLineGraph, self).__init__()
        self.num_angle_bin = num_angle_bin

    def forward(self, graph):
        """
        Generate the spatial line graph of the input graph.
        The edge types are decided by the angles between two adjacent edges in the input graph.

        Parameters:
            graph (PackedGraph): :math:`n` graph(s)

        Returns:
            graph (PackedGraph): the spatial line graph
        """
        line_graph = graph.line_graph()
        node_in, node_out = graph.edge_list[:, :2].t()
        edge_in, edge_out = line_graph.edge_list.t()

        # compute the angle ijk
        node_i = node_out[edge_out]
        node_j = node_in[edge_out]
        node_k = node_in[edge_in]
        vector1 = graph.node_position[node_i] - graph.node_position[node_j]
        vector2 = graph.node_position[node_k] - graph.node_position[node_j]
        x = (vector1 * vector2).sum(dim=-1)
        y = torch.cross(vector1, vector2).norm(dim=-1)
        angle = torch.atan2(y, x)
        relation = (angle / math.pi * self.num_angle_bin).long()
        edge_list = torch.cat([line_graph.edge_list, relation.unsqueeze(-1)], dim=-1)

        return type(line_graph)(edge_list, num_nodes=line_graph.num_nodes, offsets=line_graph._offsets,
                                num_edges=line_graph.num_edges, num_relation=self.num_angle_bin,
                                meta_dict=line_graph.meta_dict, **line_graph.data_dict)
    

class IEConvLayer(nn.Module):
    eps = 1e-6

    def __init__(self, input_dim, hidden_dim, output_dim, edge_input_dim, kernel_hidden_dim=32,
                dropout=0.05, dropout_before_conv=0.2, activation="relu", aggregate_func="sum"):
        super(IEConvLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.kernel_hidden_dim = kernel_hidden_dim
        self.aggregate_func = aggregate_func

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.kernel = MultiLayerPerceptron(edge_input_dim, [kernel_hidden_dim, (hidden_dim + 1) * hidden_dim])
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.input_batch_norm = nn.BatchNorm1d(input_dim)
        self.message_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.update_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.output_batch_norm = nn.BatchNorm1d(output_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout_before_conv = nn.Dropout(dropout_before_conv)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def message(self, graph, input, edge_input):
        node_in = graph.edge_list[:, 0]
        message = self.linear1(input[node_in])
        message = self.message_batch_norm(message)
        message = self.dropout_before_conv(self.activation(message))
        kernel = self.kernel(edge_input).view(-1, self.hidden_dim + 1, self.hidden_dim)
        message = torch.einsum('ijk, ik->ij', kernel[:, 1:, :], message) + kernel[:, 0, :]

        return message
    
    def aggregate(self, graph, message):
        node_in, node_out = graph.edge_list.t()[:2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        
        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node) 
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        return update

    def combine(self, input, update):
        output = self.linear2(update)
        return output

    def forward(self, graph, input, edge_input):
        input = self.input_batch_norm(input)
        layer_input = self.dropout(self.activation(input))
        
        message = self.message(graph, layer_input, edge_input)
        update = self.aggregate(graph, message)
        update = self.dropout(self.activation(self.update_batch_norm(update)))
        
        output = self.combine(input, update)
        output = self.output_batch_norm(output)
        return output
    

class GeometricRelationalGraphConv(nn.Module):
    eps = 1e-6

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, 
                batch_norm=False, activation="relu"):
        super(GeometricRelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input, edge_input=None):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        if edge_input is not None:
            assert edge_input.shape == message.shape
            message += edge_input
        return message
    
    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation

        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation)
        update = update.view(graph.num_node, self.num_relation * self.input_dim)

        return update
    
    def combine(self, input, update):
        output = self.linear(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    
    def forward(self, graph, input, edge_input=None):
        message = self.message(graph, input, edge_input)
        update = self.aggregate(graph, message)
        output = self.combine(input, update)
        return output
    

class GearNetIEConv(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dims, num_relation, edge_input_dim=None,
                 batch_norm=False, activation="relu", concat_hidden=False, short_cut=True, 
                 readout="sum", dropout=0, num_angle_bin=None, layer_norm=False, use_ieconv=False):
        super(GearNetIEConv, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [embedding_dim if embedding_dim > 0 else input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.concat_hidden = concat_hidden
        self.short_cut = short_cut
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.layer_norm = layer_norm
        self.use_ieconv = use_ieconv  

        if embedding_dim > 0:
            self.linear = nn.Linear(input_dim, embedding_dim)
            self.embedding_batch_norm = nn.BatchNorm1d(embedding_dim)

        self.layers = nn.ModuleList()
        self.ieconvs = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            # note that these layers are from gearnet.layer instead of torchdrug.layers
            self.layers.append(GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
            if use_ieconv:
                self.ieconvs.append(IEConvLayer(self.dims[i], self.dims[i] // 4, 
                                    self.dims[i+1], edge_input_dim=14, kernel_hidden_dim=32))
        if num_angle_bin:
            self.spatial_line_graph = SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if layer_norm:
            self.layer_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.layer_norms.append(nn.LayerNorm(self.dims[i + 1]))

        self.dropout = nn.Dropout(dropout)

        if readout == "sum":
            self.readout = SumReadout()
        elif readout == "mean":
            self.readout = MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def get_ieconv_edge_feature(self, graph):
        u = torch.ones_like(graph.node_position)
        u[1:] = graph.node_position[1:] - graph.node_position[:-1]
        u = F.normalize(u, dim=-1)
        b = torch.ones_like(graph.node_position)
        b[:-1] = u[:-1] - u[1:]
        b = F.normalize(b, dim=-1)
        n = torch.ones_like(graph.node_position)
        n[:-1] = torch.cross(u[:-1], u[1:])
        n = F.normalize(n, dim=-1)

        local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1)

        node_in, node_out = graph.edge_list.t()[:2]
        t = graph.node_position[node_out] - graph.node_position[node_in]
        t = torch.einsum('ijk, ij->ik', local_frame[node_in], t)
        r = torch.sum(local_frame[node_in] * local_frame[node_out], dim=1)
        delta = torch.abs(graph.atom2residue[node_in] - graph.atom2residue[node_out]).float() / 6
        delta = delta.unsqueeze(-1)

        return torch.cat([
            t, r, delta, 
            1 - 2 * t.abs(), 1 - 2 * r.abs(), 1 - 2 * delta.abs()
        ], dim=-1)

    def forward(self, graph, input, all_loss=None, metric=None):
        hiddens = []
        layer_input = input
        if self.embedding_dim > 0:
            layer_input = self.linear(layer_input)
            layer_input = self.embedding_batch_norm(layer_input)
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_hidden = line_graph.node_feature.float()
        else:
            edge_hidden = None
        ieconv_edge_feature = self.get_ieconv_edge_feature(graph)

        for i in range(len(self.layers)):
            # edge message passing
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_hidden)
            hidden = self.layers[i](graph, layer_input, edge_hidden)
            # ieconv layer
            if self.use_ieconv:
                hidden = hidden + self.ieconvs[i](graph, layer_input, ieconv_edge_feature)
            hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.layer_norm:
                hidden = self.layer_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }
    
    @classmethod
    def load_args(cls, args):
        return GearNetIEConv(args.input_dim, args.embedding_dim, args.hidden_dims, args.num_relation, args.edge_input_dim,
                 args.batch_norm, args.activation, args.concat_hidden, args.short_cut, args.readout, args.dropout, 
                 args.num_angle_bin, args.layer_norm, args.use_ieconv)
