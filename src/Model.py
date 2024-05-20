
import __future__

import numpy as np
import warnings
from modules.NetVLAD import NetVLAD, NetRVLAD, ContextualNetVLAD

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

from torch_geometric.nn.conv import EdgeConv, DynamicEdgeConv
from torch_geometric.nn.conv import GCNConv, GATConv, GINConv, GATv2Conv
from torch_geometric.nn.models import GAT, GIN, GCN
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.nn import global_max_pool, global_mean_pool

###########################################################################
#                          SEGMENTATION MODULES                           #
###########################################################################

class ContextAwareNetVladTemporal(nn.Module):
    def __init__(self, num_classes=3, args=None):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS:    1. The segmentation of the form (batch_size,chunk_size,num_classes)
                    2. The action spotting of the form (batch_size,num_detections,2+num_classes)
        """

        super(ContextAwareNetVladTemporal, self).__init__()

        self.args = args
        self.load_weights(weights=args.load_weights)

        # self.input_size = args.num_features
        self.num_classes = args.annotation_nr
        self.dim_capsule = args.dim_capsule
        self.receptive_field = args.receptive_field*args.fps
        # self.num_detections = args.num_detections
        self.chunk_size = args.chunk_size*args.fps
        self.fps = args.fps
        self.input_channel = args.input_channel

        # -------------------------------
        # Initialize the player backbone
        # -------------------------------
        self.init_GNN(multiplier=self.args.feature_multiplier)

        # -------------------------------
        # Initialize the NetVLAD pooling
        # -------------------------------
        self.init_NetVLAD(vocab_size=self.args.vocab_size, pooling=self.args.pooling, multiplier=self.args.feature_multiplier)

        # -------------------
        # Segmentation module
        # -------------------
        self.kernel_seg_size = 5
        self.pad_seg = nn.ZeroPad2d((0,0,(self.kernel_seg_size-1)//2, self.kernel_seg_size-1-(self.kernel_seg_size-1)//2))
        
        # Tackles fetaures from GNN backbone
        self.conv_seg = nn.Conv2d(in_channels=(76*self.args.feature_multiplier*self.vocab_size), out_channels=int(self.dim_capsule*self.num_classes), kernel_size=(self.kernel_seg_size,1))
        self.batch_seg = nn.BatchNorm2d(num_features=self.chunk_size, momentum=0.01,eps=0.001)
        

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, representation_inputs):

        # -----------------------------------
        # Feature input (chunks of the video)
        # -----------------------------------
        gnn_concatenation = self.forward_GNN(representation_inputs)
        # print("GNN output size: ", gnn_concatenation.size())
        # print(torch.cuda.memory_allocated()/10**9)

        # -----------------------------------
        # NetVLAD pooling
        # -----------------------------------
        netvlad_out = self.forward_NetVLAD(gnn_concatenation)
        # print("NetVLAD output size: ", netvlad_out.size())
        

        # -------------------
        # Segmentation module
        # -------------------

        conv_seg = self.conv_seg(self.pad_seg(netvlad_out))
        # print("Conv_seg size: ", conv_seg.size())
        # print(torch.cuda.memory_allocated()/10**9)

        conv_seg_permuted = conv_seg.permute(0,2,3,1)
        # print("Conv_seg_permuted size: ", conv_seg_permuted.size())
        # print(torch.cuda.memory_allocated()/10**9)

        conv_seg_reshaped = conv_seg_permuted.view(conv_seg_permuted.size()[0],conv_seg_permuted.size()[1],self.dim_capsule,self.num_classes)
        # print("Conv_seg_reshaped size: ", conv_seg_reshaped.size())
        # print(torch.cuda.memory_allocated()/10**9)

        #conv_seg_reshaped_permuted = conv_seg_reshaped.permute(0,3,1,2)
        #print("Conv_seg_reshaped_permuted size: ", conv_seg_reshaped_permuted.size())

        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
        # print("Conv_seg_norm: ", conv_seg_norm.size())
        # print(torch.cuda.memory_allocated()/10**9)

        #conv_seg_norm_permuted = conv_seg_norm.permute(0,2,3,1)
        #print("Conv_seg_norm_permuted size: ", conv_seg_norm_permuted.size())

        output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm-0.5), dim=2)*4/self.dim_capsule)
        # print("Output_segmentation size: ", output_segmentation.size())
        # print(torch.cuda.memory_allocated()/10**9)
        return output_segmentation

    def init_GNN(self,multiplier=1):

        # ---------------------
        # Representation branch
        # ---------------------
        # Base convolutional layers
        input_channel = self.input_channel
        # print("input_channel", input_channel)
        
        if self.args.backbone_player == "GCN":
            self.r_graph_1 = GCNConv(input_channel, 8*multiplier)
            self.r_graph_2 = GCNConv(8*multiplier, 16*multiplier)
            self.r_graph_3 = GCNConv(16*multiplier, 32*multiplier)
            self.r_graph_4 = GCNConv(32*multiplier, 76*multiplier)
        
        elif self.args.backbone_player == "GAT":
            self.r_graph_1 = GATConv(input_channel, 8*multiplier, heads=4, concat=False)
            self.r_graph_2 = GATConv(8*multiplier, 16*multiplier, heads=4, concat=False)
            self.r_graph_3 = GATConv(16*multiplier, 32*multiplier, heads=4, concat=False)
            self.r_graph_4 = GATConv(32*multiplier, 76*multiplier, heads=4, concat=False)
        
        elif self.args.backbone_player == "GIN":
            self.r_graph_1 = GINConv(
                                    Sequential(Linear(input_channel,  8*multiplier),
                                    BatchNorm1d(8*multiplier), ReLU(),
                                    Linear(8*multiplier, 8*multiplier), ReLU())
                                    )
            self.r_graph_2 = GINConv(
                                    Sequential(Linear(8*multiplier,  16*multiplier),
                                    BatchNorm1d(16*multiplier), ReLU(),
                                    Linear(16*multiplier, 16*multiplier), ReLU())
                                    )
            self.r_graph_3 = GINConv(
                                    Sequential(Linear(16*multiplier,  32*multiplier),
                                    BatchNorm1d(32*multiplier), ReLU(),
                                    Linear(32*multiplier, 32*multiplier), ReLU())
                                    )
            self.r_graph_4 = GINConv(
                                    Sequential(Linear(32*multiplier,  76*multiplier),
                                    BatchNorm1d(76*multiplier), ReLU(),
                                    Linear(76*multiplier, 76*multiplier), ReLU())
                                    )
        elif self.args.backbone_player == "EdgeConvGCN":
            self.r_graph_1 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*input_channel, 8*multiplier) ]))
            self.r_graph_2 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*8*multiplier, 16*multiplier) ]))
            self.r_graph_3 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*16*multiplier, 32*multiplier) ]))
            self.r_graph_4 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*32*multiplier, 76*multiplier) ]))
        
        elif self.args.backbone_player == "DynamicEdgeConvGCN":
            self.r_graph_1 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*input_channel, 8*multiplier) ]), k=3)
            self.r_graph_2 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*8*multiplier, 16*multiplier) ]), k=3)
            self.r_graph_3 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*16*multiplier, 32*multiplier) ]), k=3)
            self.r_graph_4 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*32*multiplier, 76*multiplier) ]), k=3)

        elif "resGCN" in self.args.backbone_player:
            # hidden_channels=64, num_layers=28
            # input_channel = 6
            output_channel = 76*multiplier
            hidden_channels = 64
            self.num_layers = int(self.args.backbone_player.split("-")[-1])

            self.node_encoder = nn.Linear(input_channel, hidden_channels)
            self.edge_encoder = nn.Linear(input_channel, hidden_channels)
            self.layers = torch.nn.ModuleList()
            for i in range(1, self.num_layers + 1):
                conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                            t=1.0, learn_t=True, num_layers=2, norm='layer')
                norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
                act = nn.ReLU(inplace=True)

                layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1,
                                    ckpt_grad=i % 3)
                self.layers.append(layer)

            self.lin = nn.Linear(hidden_channels, output_channel)


    def forward_GNN(self, representation_inputs):
        BS = self.args.batch_size
        T = self.chunk_size
        # --------------------
        # Representation input -> GCN
        # --------------------
        #print("Representation input size: ", representation_inputs.size())

        # Get node and edge information
        x = representation_inputs.x
        batch = representation_inputs.batch
        edge_index = representation_inputs.edge_index
        edge_attr = representation_inputs.edge_attr[:,:4]
        edge_weight = representation_inputs.edge_attr[:,4]
        
        if (self.args.backbone_player == "GCN") or (self.args.backbone_player == "EdgeConvGCN"):
            x = F.relu(self.r_graph_1(x, edge_index=edge_index, edge_weight=edge_weight))
            x = F.relu(self.r_graph_2(x, edge_index=edge_index, edge_weight=edge_weight))
            x = F.relu(self.r_graph_3(x, edge_index=edge_index, edge_weight=edge_weight))
            x = F.relu(self.r_graph_4(x, edge_index=edge_index, edge_weight=edge_weight))
        elif (self.args.backbone_player == "GAT"):
            x = F.relu(self.r_graph_1(x, edge_index=edge_index, edge_attr=edge_attr))
            x = F.relu(self.r_graph_2(x, edge_index=edge_index, edge_attr=edge_attr))
            x = F.relu(self.r_graph_3(x, edge_index=edge_index, edge_attr=edge_attr))
            x = F.relu(self.r_graph_4(x, edge_index=edge_index, edge_attr=edge_attr))
        elif (self.args.backbone_player == "GIN"):
            x = F.relu(self.r_graph_1(x, edge_index=edge_index))
            x = F.relu(self.r_graph_2(x, edge_index=edge_index))
            x = F.relu(self.r_graph_3(x, edge_index=edge_index))
            x = F.relu(self.r_graph_4(x, edge_index=edge_index))
        elif "DynamicEdgeConvGCN" in self.args.backbone_player: #EdgeConvGCN or DynamicEdgeConvGCN
            x = F.relu(self.r_graph_1(x, batch))
            x = F.relu(self.r_graph_2(x, batch))
            x = F.relu(self.r_graph_3(x, batch))
            x = F.relu(self.r_graph_4(x, batch))
        elif "resGCN" in self.args.backbone_player: #EdgeConvGCN or DynamicEdgeConvGCN
            x = self.node_encoder(x)
            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers[1:]:
                x = layer(x, edge_index)

            x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training)

            x = self.lin(x)
        # print("before max_pool", x.shape)
        x = global_mean_pool(x, batch) 
        # print("after max_pool", x.shape)
        # print(batch)
        # BS = inputs.shape[0]

        # magic fix with zero padding
        expected_size = BS * T
        x = torch.cat([x, torch.zeros(expected_size-x.shape[0], x.shape[1]).to(x.device)], 0)

        x = x.reshape(BS, T, x.shape[1]) #BSxTxFS
        x = x.permute((0,2,1)) #BSxFSxT
        x = x.unsqueeze(-1) #BSxFSxTx1
        r_concatenation = x

        return r_concatenation
    
    def init_NetVLAD(self, vocab_size=64, pooling="NetVLAD", multiplier=2):
        self.vocab_size = vocab_size
        input_size = 76*multiplier
        vlad_out_dim = int(input_size * vocab_size)

        if pooling == "NetVLAD":
            self.vlad = ContextualNetVLAD(cluster_size=int(vocab_size), feature_size=input_size,
                                            add_batch_norm=True)
            
            
    def forward_NetVLAD(self, gnn_output):
        BS,FS,T,_ = gnn_output.shape
        gnn_output = gnn_output.squeeze(-1)
        gnn_output_permuted = gnn_output.permute((0,2,1))
        
        vlad_output = self.vlad(gnn_output_permuted) # BSxTx(FSxC)
        vlad_output = vlad_output.permute((0,2,1))
        vlad_output = vlad_output.unsqueeze(-1)
        return vlad_output

class ContextAwareNetVladGlobal(nn.Module):
    def __init__(self, num_classes=3, args=None):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS:    1. The segmentation of the form (batch_size,chunk_size,num_classes)
                    2. The action spotting of the form (batch_size,num_detections,2+num_classes)
        """

        super(ContextAwareNetVladGlobal, self).__init__()

        self.args = args
        self.load_weights(weights=args.load_weights)

        # self.input_size = args.num_features
        self.num_classes = args.annotation_nr
        self.dim_capsule = args.dim_capsule
        self.receptive_field = args.receptive_field*args.fps
        # self.num_detections = args.num_detections
        self.chunk_size = args.chunk_size*args.fps
        self.fps = args.fps
        self.input_channel = args.input_channel

        # -------------------------------
        # Initialize the player backbone
        # -------------------------------
        self.init_GNN(multiplier=self.args.feature_multiplier)

        # -------------------------------
        # Initialize the NetVLAD pooling
        # -------------------------------
        self.init_NetVLAD(vocab_size=self.args.vocab_size, pooling=self.args.pooling, multiplier=self.args.feature_multiplier)

        # -------------------
        # Segmentation module
        # -------------------
        self.kernel_seg_size = 3
        self.pad_seg = nn.ZeroPad2d((0,0,(self.kernel_seg_size-1)//2, self.kernel_seg_size-1-(self.kernel_seg_size-1)//2))
        
        # Tackles fetaures from GNN backbone
        self.conv_seg = nn.Conv2d(in_channels=(152*self.args.feature_multiplier), out_channels=int(self.dim_capsule*self.num_classes), kernel_size=(self.kernel_seg_size,1))
        self.batch_seg = nn.BatchNorm2d(num_features=self.chunk_size, momentum=0.01,eps=0.001)
        

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, representation_inputs):

        # -----------------------------------
        # Feature input (chunks of the video)
        # -----------------------------------
        gnn_concatenation = self.forward_GNN(representation_inputs)
        # print("GNN output size: ", gnn_concatenation.size())
        # print(torch.cuda.memory_allocated()/10**9)

        # -----------------------------------
        # NetVLAD pooling
        # -----------------------------------
        netvlad_out = self.forward_NetVLAD(gnn_concatenation)
        # print("NetVLAD output size: ", netvlad_out.size())
        

        # -------------------
        # Segmentation module
        # -------------------
        full_concat = torch.cat((gnn_concatenation,netvlad_out), dim=1)
        # print("Full_concat size: ", full_concat.size())
        # print(torch.cuda.memory_allocated()/10**9)

        conv_seg = self.conv_seg(self.pad_seg(full_concat))
        # print("Conv_seg size: ", conv_seg.size())
        # print(torch.cuda.memory_allocated()/10**9)

        conv_seg_permuted = conv_seg.permute(0,2,3,1)
        # print("Conv_seg_permuted size: ", conv_seg_permuted.size())
        # print(torch.cuda.memory_allocated()/10**9)

        conv_seg_reshaped = conv_seg_permuted.view(conv_seg_permuted.size()[0],conv_seg_permuted.size()[1],self.dim_capsule,self.num_classes)
        # print("Conv_seg_reshaped size: ", conv_seg_reshaped.size())
        # print(torch.cuda.memory_allocated()/10**9)

        #conv_seg_reshaped_permuted = conv_seg_reshaped.permute(0,3,1,2)
        #print("Conv_seg_reshaped_permuted size: ", conv_seg_reshaped_permuted.size())

        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
        # print("Conv_seg_norm: ", conv_seg_norm.size())
        # print(torch.cuda.memory_allocated()/10**9)

        #conv_seg_norm_permuted = conv_seg_norm.permute(0,2,3,1)
        #print("Conv_seg_norm_permuted size: ", conv_seg_norm_permuted.size())

        output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm-0.5), dim=2)*4/self.dim_capsule)
        # print("Output_segmentation size: ", output_segmentation.size())
        # print(torch.cuda.memory_allocated()/10**9)
        return output_segmentation

    def init_GNN(self,multiplier=1):

        # ---------------------
        # Representation branch
        # ---------------------
        # Base convolutional layers
        input_channel = self.input_channel
        # print("input_channel", input_channel)
        
        if self.args.backbone_player == "GCN":
            self.r_graph_1 = GCNConv(input_channel, 8*multiplier)
            self.r_graph_2 = GCNConv(8*multiplier, 16*multiplier)
            self.r_graph_3 = GCNConv(16*multiplier, 32*multiplier)
            self.r_graph_4 = GCNConv(32*multiplier, 76*multiplier)
        
        elif self.args.backbone_player == "GAT":
            self.r_graph_1 = GATConv(input_channel, 8*multiplier, heads=4, concat=False)
            self.r_graph_2 = GATConv(8*multiplier, 16*multiplier, heads=4, concat=False)
            self.r_graph_3 = GATConv(16*multiplier, 32*multiplier, heads=4, concat=False)
            self.r_graph_4 = GATConv(32*multiplier, 76*multiplier, heads=4, concat=False)
        
        elif self.args.backbone_player == "GIN":
            self.r_graph_1 = GINConv(
                                    Sequential(Linear(input_channel,  8*multiplier),
                                    BatchNorm1d(8*multiplier), ReLU(),
                                    Linear(8*multiplier, 8*multiplier), ReLU())
                                    )
            self.r_graph_2 = GINConv(
                                    Sequential(Linear(8*multiplier,  16*multiplier),
                                    BatchNorm1d(16*multiplier), ReLU(),
                                    Linear(16*multiplier, 16*multiplier), ReLU())
                                    )
            self.r_graph_3 = GINConv(
                                    Sequential(Linear(16*multiplier,  32*multiplier),
                                    BatchNorm1d(32*multiplier), ReLU(),
                                    Linear(32*multiplier, 32*multiplier), ReLU())
                                    )
            self.r_graph_4 = GINConv(
                                    Sequential(Linear(32*multiplier,  76*multiplier),
                                    BatchNorm1d(76*multiplier), ReLU(),
                                    Linear(76*multiplier, 76*multiplier), ReLU())
                                    )
        elif self.args.backbone_player == "EdgeConvGCN":
            self.r_graph_1 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*input_channel, 8*multiplier) ]))
            self.r_graph_2 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*8*multiplier, 16*multiplier) ]))
            self.r_graph_3 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*16*multiplier, 32*multiplier) ]))
            self.r_graph_4 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*32*multiplier, 76*multiplier) ]))
        
        elif self.args.backbone_player == "DynamicEdgeConvGCN":
            self.r_graph_1 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*input_channel, 8*multiplier) ]), k=3)
            self.r_graph_2 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*8*multiplier, 16*multiplier) ]), k=3)
            self.r_graph_3 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*16*multiplier, 32*multiplier) ]), k=3)
            self.r_graph_4 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*32*multiplier, 76*multiplier) ]), k=3)

        elif "resGCN" in self.args.backbone_player:
            # hidden_channels=64, num_layers=28
            # input_channel = 6
            output_channel = 76*multiplier
            hidden_channels = 64
            self.num_layers = int(self.args.backbone_player.split("-")[-1])

            self.node_encoder = nn.Linear(input_channel, hidden_channels)
            self.edge_encoder = nn.Linear(input_channel, hidden_channels)
            self.layers = torch.nn.ModuleList()
            for i in range(1, self.num_layers + 1):
                conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                            t=1.0, learn_t=True, num_layers=2, norm='layer')
                norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
                act = nn.ReLU(inplace=True)

                layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1,
                                    ckpt_grad=i % 3)
                self.layers.append(layer)

            self.lin = nn.Linear(hidden_channels, output_channel)


    def forward_GNN(self, representation_inputs):
        BS = self.args.batch_size
        T = self.chunk_size
        # --------------------
        # Representation input -> GCN
        # --------------------
        #print("Representation input size: ", representation_inputs.size())

        # Get node and edge information
        x = representation_inputs.x
        batch = representation_inputs.batch
        edge_index = representation_inputs.edge_index
        edge_attr = representation_inputs.edge_attr[:,:4]
        edge_weight = representation_inputs.edge_attr[:,4]
        
        if (self.args.backbone_player == "GCN") or (self.args.backbone_player == "EdgeConvGCN"):
            x = F.relu(self.r_graph_1(x, edge_index=edge_index, edge_weight=edge_weight))
            x = F.relu(self.r_graph_2(x, edge_index=edge_index, edge_weight=edge_weight))
            x = F.relu(self.r_graph_3(x, edge_index=edge_index, edge_weight=edge_weight))
            x = F.relu(self.r_graph_4(x, edge_index=edge_index, edge_weight=edge_weight))
        elif (self.args.backbone_player == "GAT"):
            x = F.relu(self.r_graph_1(x, edge_index=edge_index, edge_attr=edge_attr))
            x = F.relu(self.r_graph_2(x, edge_index=edge_index, edge_attr=edge_attr))
            x = F.relu(self.r_graph_3(x, edge_index=edge_index, edge_attr=edge_attr))
            x = F.relu(self.r_graph_4(x, edge_index=edge_index, edge_attr=edge_attr))
        elif (self.args.backbone_player == "GIN"):
            x = F.relu(self.r_graph_1(x, edge_index=edge_index))
            x = F.relu(self.r_graph_2(x, edge_index=edge_index))
            x = F.relu(self.r_graph_3(x, edge_index=edge_index))
            x = F.relu(self.r_graph_4(x, edge_index=edge_index))
        elif "DynamicEdgeConvGCN" in self.args.backbone_player: #EdgeConvGCN or DynamicEdgeConvGCN
            x = F.relu(self.r_graph_1(x, batch))
            x = F.relu(self.r_graph_2(x, batch))
            x = F.relu(self.r_graph_3(x, batch))
            x = F.relu(self.r_graph_4(x, batch))
        elif "resGCN" in self.args.backbone_player: #EdgeConvGCN or DynamicEdgeConvGCN
            x = self.node_encoder(x)
            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers[1:]:
                x = layer(x, edge_index)

            x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training)

            x = self.lin(x)
        # print("before max_pool", x.shape)
        x = global_mean_pool(x, batch) 
        # print("after max_pool", x.shape)
        # print(batch)
        # BS = inputs.shape[0]

        # magic fix with zero padding
        expected_size = BS * T
        x = torch.cat([x, torch.zeros(expected_size-x.shape[0], x.shape[1]).to(x.device)], 0)

        x = x.reshape(BS, T, x.shape[1]) #BSxTxFS
        x = x.permute((0,2,1)) #BSxFSxT
        x = x.unsqueeze(-1) #BSxFSxTx1
        r_concatenation = x

        return r_concatenation
    
    def init_NetVLAD(self, vocab_size=64, pooling="NetVLAD", multiplier=2):
        input_size = 76*multiplier
        vlad_out_dim = int(input_size * vocab_size)

        if pooling == "NetVLAD":
            self.vlad = NetVLAD(cluster_size=int(vocab_size), feature_size=input_size,
                                            add_batch_norm=True)
        elif pooling == "NetRVLAD":
            self.vlad = NetRVLAD(cluster_size=int(vocab_size), feature_size=input_size,
                                            add_batch_norm=True)
            
        self.vlad_linear = nn.Linear(vlad_out_dim, input_size)
            
    def forward_NetVLAD(self, gnn_output):
        BS,FS,T,_ = gnn_output.shape
        gnn_output = gnn_output.squeeze(-1)
        gnn_output_permuted = gnn_output.permute((0,2,1))
        
        vlad_desc = self.vlad(gnn_output_permuted) # BSx(FSxC)
        vlad_output = self.vlad_linear(vlad_desc) # BSxFS
        vlad_output = vlad_output.unsqueeze(-1).expand(-1, -1, T) # BSxFSxT
        vlad_output = vlad_output.unsqueeze(-1) # BSxFSxTx1
        return vlad_output

class ContextAwareModel(nn.Module):
    def __init__(self, num_classes=3, args=None):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS:    1. The segmentation of the form (batch_size,chunk_size,num_classes)
                    2. The action spotting of the form (batch_size,num_detections,2+num_classes)
        """

        super(ContextAwareModel, self).__init__()

        self.args = args
        self.load_weights(weights=args.load_weights)

        # self.input_size = args.num_features
        self.num_classes = args.annotation_nr
        self.dim_capsule = args.dim_capsule
        self.receptive_field = args.receptive_field*args.fps
        # self.num_detections = args.num_detections
        self.chunk_size = args.chunk_size*args.fps
        self.fps = args.fps
        self.input_channel = args.input_channel

        # -------------------------------
        # Initialize the player backbone
        # -------------------------------
        self.init_GNN(multiplier=2*self.args.feature_multiplier)

        # -------------------
        # Segmentation module
        # -------------------
        self.kernel_seg_size = 3
        self.pad_seg = nn.ZeroPad2d((0,0,(self.kernel_seg_size-1)//2, self.kernel_seg_size-1-(self.kernel_seg_size-1)//2))
        self.conv_seg = nn.Conv2d(in_channels=152*self.args.feature_multiplier, out_channels=self.dim_capsule*self.num_classes, kernel_size=(self.kernel_seg_size,1))
        self.batch_seg = nn.BatchNorm2d(num_features=self.chunk_size, momentum=0.01,eps=0.001) 

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, representation_inputs):

        # -----------------------------------
        # Feature input (chunks of the video)
        # -----------------------------------
        r_concatenation = self.forward_GNN(representation_inputs)
        full_concatenation = r_concatenation
        # print(f"Concatenation size: {full_concatenation.size()}")

        # -------------------
        # Segmentation module
        # -------------------
        conv_seg = self.conv_seg(self.pad_seg(full_concatenation))
        # print("Conv_seg size: ", conv_seg.size())
        # print(torch.cuda.memory_allocated()/10**9)

        conv_seg_permuted = conv_seg.permute(0,2,3,1)
        # print("Conv_seg_permuted size: ", conv_seg_permuted.size())
        # print(torch.cuda.memory_allocated()/10**9)

        conv_seg_reshaped = conv_seg_permuted.view(conv_seg_permuted.size()[0],conv_seg_permuted.size()[1],self.dim_capsule,self.num_classes)
        # print("Conv_seg_reshaped size: ", conv_seg_reshaped.size())
        # print(torch.cuda.memory_allocated()/10**9)


        #conv_seg_reshaped_permuted = conv_seg_reshaped.permute(0,3,1,2)
        #print("Conv_seg_reshaped_permuted size: ", conv_seg_reshaped_permuted.size())

        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
        # print("Conv_seg_norm: ", conv_seg_norm.size())
        # print(torch.cuda.memory_allocated()/10**9)


        #conv_seg_norm_permuted = conv_seg_norm.permute(0,2,3,1)
        #print("Conv_seg_norm_permuted size: ", conv_seg_norm_permuted.size())

        output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm-0.5), dim=2)*4/self.dim_capsule)
        # print("Output_segmentation size: ", output_segmentation.size())
        # print(torch.cuda.memory_allocated()/10**9)
        return output_segmentation

    def init_GNN(self,multiplier=1):

        # ---------------------
        # Representation branch
        # ---------------------
        # Base convolutional layers
        input_channel = self.input_channel
        # print("input_channel", input_channel)
        
        if self.args.backbone_player == "GCN":
            self.r_graph_1 = GCNConv(input_channel, 8*multiplier)
            self.r_graph_2 = GCNConv(8*multiplier, 16*multiplier)
            self.r_graph_3 = GCNConv(16*multiplier, 32*multiplier)
            self.r_graph_4 = GCNConv(32*multiplier, 76*multiplier)
        
        elif self.args.backbone_player == "GAT":
            self.r_graph_1 = GATConv(input_channel, 8*multiplier, heads=4, concat=False)
            self.r_graph_2 = GATConv(8*multiplier, 16*multiplier, heads=4, concat=False)
            self.r_graph_3 = GATConv(16*multiplier, 32*multiplier, heads=4, concat=False)
            self.r_graph_4 = GATConv(32*multiplier, 76*multiplier, heads=4, concat=False)
        
        elif self.args.backbone_player == "GIN":
            self.r_graph_1 = GINConv(
                                    Sequential(Linear(input_channel,  8*multiplier),
                                    BatchNorm1d(8*multiplier), ReLU(),
                                    Linear(8*multiplier, 8*multiplier), ReLU())
                                    )
            self.r_graph_2 = GINConv(
                                    Sequential(Linear(8*multiplier,  16*multiplier),
                                    BatchNorm1d(16*multiplier), ReLU(),
                                    Linear(16*multiplier, 16*multiplier), ReLU())
                                    )
            self.r_graph_3 = GINConv(
                                    Sequential(Linear(16*multiplier,  32*multiplier),
                                    BatchNorm1d(32*multiplier), ReLU(),
                                    Linear(32*multiplier, 32*multiplier), ReLU())
                                    )
            self.r_graph_4 = GINConv(
                                    Sequential(Linear(32*multiplier,  76*multiplier),
                                    BatchNorm1d(76*multiplier), ReLU(),
                                    Linear(76*multiplier, 76*multiplier), ReLU())
                                    )
        elif self.args.backbone_player == "EdgeConvGCN":
            self.r_graph_1 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*input_channel, 8*multiplier) ]))
            self.r_graph_2 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*8*multiplier, 16*multiplier) ]))
            self.r_graph_3 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*16*multiplier, 32*multiplier) ]))
            self.r_graph_4 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*32*multiplier, 76*multiplier) ]))
        
        elif self.args.backbone_player == "DynamicEdgeConvGCN":
            self.r_graph_1 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*input_channel, 8*multiplier) ]), k=3)
            self.r_graph_2 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*8*multiplier, 16*multiplier) ]), k=3)
            self.r_graph_3 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*16*multiplier, 32*multiplier) ]), k=3)
            self.r_graph_4 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*32*multiplier, 76*multiplier) ]), k=3)

        elif "resGCN" in self.args.backbone_player:
            # hidden_channels=64, num_layers=28
            # input_channel = 6
            output_channel = 76*multiplier
            hidden_channels = 64
            self.num_layers = int(self.args.backbone_player.split("-")[-1])

            self.node_encoder = nn.Linear(input_channel, hidden_channels)
            self.edge_encoder = nn.Linear(input_channel, hidden_channels)
            self.layers = torch.nn.ModuleList()
            for i in range(1, self.num_layers + 1):
                conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                            t=1.0, learn_t=True, num_layers=2, norm='layer')
                norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
                act = nn.ReLU(inplace=True)

                layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1,
                                    ckpt_grad=i % 3)
                self.layers.append(layer)

            self.lin = nn.Linear(hidden_channels, output_channel)

    def forward_GNN(self, representation_inputs):
        BS = self.args.batch_size
        T = self.chunk_size
        # --------------------
        # Representation input -> GCN
        # --------------------
        #print("Representation input size: ", representation_inputs.size())

        # Get node and edge information
        x = representation_inputs.x
        batch = representation_inputs.batch
        edge_index = representation_inputs.edge_index
        edge_attr = representation_inputs.edge_attr[:,:4]
        edge_weight = representation_inputs.edge_attr[:,4]
        
        if (self.args.backbone_player == "GCN") or (self.args.backbone_player == "EdgeConvGCN"):
            x = F.relu(self.r_graph_1(x, edge_index=edge_index, edge_weight=edge_weight))
            x = F.relu(self.r_graph_2(x, edge_index=edge_index, edge_weight=edge_weight))
            x = F.relu(self.r_graph_3(x, edge_index=edge_index, edge_weight=edge_weight))
            x = F.relu(self.r_graph_4(x, edge_index=edge_index, edge_weight=edge_weight))
        elif (self.args.backbone_player == "GAT"):
            x = F.relu(self.r_graph_1(x, edge_index=edge_index, edge_attr=edge_attr))
            x = F.relu(self.r_graph_2(x, edge_index=edge_index, edge_attr=edge_attr))
            x = F.relu(self.r_graph_3(x, edge_index=edge_index, edge_attr=edge_attr))
            x = F.relu(self.r_graph_4(x, edge_index=edge_index, edge_attr=edge_attr))
        elif (self.args.backbone_player == "GIN"):
            x = F.relu(self.r_graph_1(x, edge_index=edge_index))
            x = F.relu(self.r_graph_2(x, edge_index=edge_index))
            x = F.relu(self.r_graph_3(x, edge_index=edge_index))
            x = F.relu(self.r_graph_4(x, edge_index=edge_index))
        elif "DynamicEdgeConvGCN" in self.args.backbone_player: #EdgeConvGCN or DynamicEdgeConvGCN
            x = F.relu(self.r_graph_1(x, batch))
            x = F.relu(self.r_graph_2(x, batch))
            x = F.relu(self.r_graph_3(x, batch))
            x = F.relu(self.r_graph_4(x, batch))
        elif "resGCN" in self.args.backbone_player: #EdgeConvGCN or DynamicEdgeConvGCN
            x = self.node_encoder(x)
            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers[1:]:
                x = layer(x, edge_index)

            x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training)

            x = self.lin(x)
        # print("before max_pool", x.shape)
        x = global_mean_pool(x, batch) 
        # print("after max_pool", x.shape)
        # print(batch)
        # BS = inputs.shape[0]

        # magic fix with zero padding
        expected_size = BS * T
        x = torch.cat([x, torch.zeros(expected_size-x.shape[0], x.shape[1]).to(x.device)], 0)

        x = x.reshape(BS, T, x.shape[1]) #BSxTxFS
        x = x.permute((0,2,1)) #BSxFSxT
        x = x.unsqueeze(-1) #BSxFSxTx1
        r_concatenation = x

        return r_concatenation

class BaseContextAwareModel(nn.Module):
    def __init__(self, num_classes=3, args=None):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS:    1. The segmentation of the form (batch_size,chunk_size,num_classes)
                    2. The action spotting of the form (batch_size,num_detections,2+num_classes)
        """

        super(ContextAwareModel, self).__init__()

        self.args = args
        self.load_weights(weights=args.load_weights)

        self.input_size = args.num_features
        self.num_classes = num_classes
        self.dim_capsule = args.dim_capsule
        self.receptive_field = args.receptive_field*args.fps
        self.num_detections = args.num_detections
        self.chunk_size = args.chunk_size*args.fps
        self.fps = args.fps
        self.representation_width = args.dim_representation_w
        self.representation_height = args.dim_representation_h
        self.representation_channels = args.dim_representation_c
        self.input_channel = args.input_channel

        self.pyramid_size_1 = int(np.ceil(self.receptive_field/7))
        self.pyramid_size_2 = int(np.ceil(self.receptive_field/3))
        self.pyramid_size_3 = int(np.ceil(self.receptive_field/2))
        self.pyramid_size_4 = int(np.ceil(self.receptive_field))
        self.pyramide_pool_size_h = int((((self.representation_height-4)/4)-4)/2)
        self.pyramide_pool_size_w = int((((self.representation_width-4)/4)-4)/2)

        # -------------------------------
        # Initialize the feature backbone
        # -------------------------------

        if args.backbone_feature is not None:
            if args.backbone_feature == "2DConv" and args.backbone_player is not None:
                self.init_2DConv(multiplier=1*self.args.feature_multiplier)
            elif args.backbone_feature == "2DConv" and args.backbone_player is None:
                self.init_2DConv(multiplier=2*self.args.feature_multiplier)

        # -------------------------------
        # Initialize the player backbone
        # -------------------------------

        if args.backbone_player is not None:
            if args.backbone_player == "3DConv" and args.backbone_feature is not None:
                self.init_3DConv(multiplier=1*self.args.feature_multiplier)
            elif args.backbone_player == "3DConv" and args.backbone_feature is None:
                self.init_3DConv(multiplier=2*self.args.feature_multiplier)
            elif "GCN" in self.args.backbone_player and args.backbone_feature is not None:
                self.init_GCN(multiplier=1*self.args.feature_multiplier)
            elif "GCN" in self.args.backbone_player and args.backbone_feature is None:
                self.init_GCN(multiplier=2*self.args.feature_multiplier)

        # -------------------
        # Segmentation module
        # -------------------

        self.kernel_seg_size = 3
        self.pad_seg = nn.ZeroPad2d((0,0,(self.kernel_seg_size-1)//2, self.kernel_seg_size-1-(self.kernel_seg_size-1)//2))
        self.conv_seg = nn.Conv2d(in_channels=152*self.args.feature_multiplier, out_channels=self.dim_capsule*self.num_classes, kernel_size=(self.kernel_seg_size,1))
        self.batch_seg = nn.BatchNorm2d(num_features=self.chunk_size, momentum=0.01,eps=0.001) 


        # -------------------
        # detection module
        # -------------------       
        self.max_pool_spot = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        self.kernel_spot_size = 3
        self.pad_spot_1 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_1 = nn.Conv2d(in_channels=self.num_classes*(self.dim_capsule+1), out_channels=32, kernel_size=(self.kernel_spot_size,1))
        self.max_pool_spot_1 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        self.pad_spot_2 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.kernel_spot_size,1))
        self.max_pool_spot_2 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))

        # Confidence branch
        self.conv_conf = nn.Conv2d(in_channels=16*(self.chunk_size//8-1), out_channels=self.num_detections*2, kernel_size=(1,1))

        # Class branch
        self.conv_class = nn.Conv2d(in_channels=16*(self.chunk_size//8-1), out_channels=self.num_detections*self.num_classes, kernel_size=(1,1))
        self.softmax = nn.Softmax(dim=-1)


    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, representation_inputs):

        # -----------------------------------
        # Feature input (chunks of the video)
        # -----------------------------------
        # input_shape: (batch,channel,frames,dim_features)
        #print("Input size: ", inputs.size())

        
        concatenation = None
        r_concatenation = None

        # if self.args.backbone_feature == "2DConv":
        #     concatenation = self.forward_2DConv(inputs)

        if self.args.backbone_player == "3DConv":
            r_concatenation = self.forward_3DConv(representation_inputs)
        elif "GCN" in self.args.backbone_player:
            r_concatenation = self.forward_GCN(representation_inputs)
        
        if r_concatenation is not None and concatenation is not None:
            if self.args.with_dropout == 0 or not self.training:
                full_concatenation = torch.cat((concatenation, r_concatenation), 1)
            elif self.args.with_dropout > 0 and self.training:
                random_number = torch.rand(1, device=concatenation.device)
                if (random_number < self.args.with_dropout/2)[0]:
                    concatenation = concatenation * torch.zeros(concatenation.shape, device=concatenation.device, dtype = torch.float )
                elif (random_number < self.args.with_dropout)[0]:
                    r_concatenation = r_concatenation * torch.zeros(r_concatenation.shape, device=r_concatenation.device, dtype = torch.float )
                full_concatenation = torch.cat((concatenation, r_concatenation), 1)


            # full_concatenation = torch.cat((concatenation, r_concatenation), 1)
        elif r_concatenation is None and concatenation is not None:
            full_concatenation = concatenation
        elif r_concatenation is not None and concatenation is None:
            full_concatenation = r_concatenation
        #print("full_concatenation size: ", full_concatenation.size())
        #print(torch.cuda.memory_allocated()/10**9)

        # -------------------
        # Segmentation module
        # -------------------

        conv_seg = self.conv_seg(self.pad_seg(full_concatenation))
        #print("Conv_seg size: ", conv_seg.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_seg_permuted = conv_seg.permute(0,2,3,1)
        #print("Conv_seg_permuted size: ", conv_seg_permuted.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_seg_reshaped = conv_seg_permuted.view(conv_seg_permuted.size()[0],conv_seg_permuted.size()[1],self.dim_capsule,self.num_classes)
        #print("Conv_seg_reshaped size: ", conv_seg_reshaped.size())
        #print(torch.cuda.memory_allocated()/10**9)


        #conv_seg_reshaped_permuted = conv_seg_reshaped.permute(0,3,1,2)
        #print("Conv_seg_reshaped_permuted size: ", conv_seg_reshaped_permuted.size())

        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
        #print("Conv_seg_norm: ", conv_seg_norm.size())
        #print(torch.cuda.memory_allocated()/10**9)


        #conv_seg_norm_permuted = conv_seg_norm.permute(0,2,3,1)
        #print("Conv_seg_norm_permuted size: ", conv_seg_norm_permuted.size())

        output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm-0.5), dim=2)*4/self.dim_capsule)
        #print("Output_segmentation size: ", output_segmentation.size())
        #print(torch.cuda.memory_allocated()/10**9)


        # ---------------
        # Spotting module
        # ---------------

        # Concatenation of the segmentation score to the capsules
        output_segmentation_reverse = 1-output_segmentation
        #print("Output_segmentation_reverse size: ", output_segmentation_reverse.size())
        #print(torch.cuda.memory_allocated()/10**9)

        output_segmentation_reverse_reshaped = output_segmentation_reverse.unsqueeze(2)
        #print("Output_segmentation_reverse_reshaped size: ", output_segmentation_reverse_reshaped.size())
        #print(torch.cuda.memory_allocated()/10**9)


        output_segmentation_reverse_reshaped_permutted = output_segmentation_reverse_reshaped.permute(0,3,1,2)
        #print("Output_segmentation_reverse_reshaped_permutted size: ", output_segmentation_reverse_reshaped_permutted.size())
        #print(torch.cuda.memory_allocated()/10**9)

        concatenation_2 = torch.cat((conv_seg, output_segmentation_reverse_reshaped_permutted), dim=1)
        #print("Concatenation_2 size: ", concatenation_2.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_spot = self.max_pool_spot(F.relu(concatenation_2))
        #print("Conv_spot size: ", conv_spot.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_spot_1 = F.relu(self.conv_spot_1(self.pad_spot_1(conv_spot)))
        #print("Conv_spot_1 size: ", conv_spot_1.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_spot_1_pooled = self.max_pool_spot_1(conv_spot_1)
        #print("Conv_spot_1_pooled size: ", conv_spot_1_pooled.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_spot_2 = F.relu(self.conv_spot_2(self.pad_spot_2(conv_spot_1_pooled)))
        #print("Conv_spot_2 size: ", conv_spot_2.size())
        #print(torch.cuda.memory_allocated()/10**9)

        conv_spot_2_pooled = self.max_pool_spot_2(conv_spot_2)
        #print("Conv_spot_2_pooled size: ", conv_spot_2_pooled.size())
        #print(torch.cuda.memory_allocated()/10**9)

        spotting_reshaped = conv_spot_2_pooled.view(conv_spot_2_pooled.size()[0],-1,1,1)
        #print("Spotting_reshape size: ", spotting_reshaped.size())
        #print(torch.cuda.memory_allocated()/10**9)

        # Confindence branch
        conf_pred = torch.sigmoid(self.conv_conf(spotting_reshaped).view(spotting_reshaped.shape[0],self.num_detections,2))
        #print("Conf_pred size: ", conf_pred.size())
        #print(torch.cuda.memory_allocated()/10**9)

        # Class branch
        conf_class = self.softmax(self.conv_class(spotting_reshaped).view(spotting_reshaped.shape[0],self.num_detections,self.num_classes))
        #print("Conf_class size: ", conf_class.size())
        #print(torch.cuda.memory_allocated()/10**9)

        output_spotting = torch.cat((conf_pred,conf_class),dim=-1)
        #print("Output_spotting size: ", output_spotting.size())
        #print(torch.cuda.memory_allocated()/10**9)


        return output_segmentation, output_spotting
        


    def init_2DConv(self, multiplier=1):

        # Base Convolutional Layers
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64*multiplier, kernel_size=(1,self.input_size))
        self.conv_2 = nn.Conv2d(in_channels=64*multiplier, out_channels=16*multiplier, kernel_size=(1,1))

        # Temporal Pyramidal Module
        self.pad_p_1 = nn.ZeroPad2d((0,0,(self.pyramid_size_1-1)//2, self.pyramid_size_1-1-(self.pyramid_size_1-1)//2))
        self.pad_p_2 = nn.ZeroPad2d((0,0,(self.pyramid_size_2-1)//2, self.pyramid_size_2-1-(self.pyramid_size_2-1)//2))
        self.pad_p_3 = nn.ZeroPad2d((0,0,(self.pyramid_size_3-1)//2, self.pyramid_size_3-1-(self.pyramid_size_3-1)//2))
        self.pad_p_4 = nn.ZeroPad2d((0,0,(self.pyramid_size_4-1)//2, self.pyramid_size_4-1-(self.pyramid_size_4-1)//2))
        self.conv_p_1 = nn.Conv2d(in_channels=16*multiplier, out_channels=4*multiplier, kernel_size=(self.pyramid_size_1,1))
        self.conv_p_2 = nn.Conv2d(in_channels=16*multiplier, out_channels=8*multiplier, kernel_size=(self.pyramid_size_2,1))
        self.conv_p_3 = nn.Conv2d(in_channels=16*multiplier, out_channels=16*multiplier, kernel_size=(self.pyramid_size_3,1))
        self.conv_p_4 = nn.Conv2d(in_channels=16*multiplier, out_channels=32*multiplier, kernel_size=(self.pyramid_size_4,1))

    def init_3DConv(self, multiplier=1):

        # ---------------------
        # Representation branch
        # ---------------------
        # Base convolutional layers
        self.r_conv_1 = nn.Conv3d(in_channels=self.representation_channels, out_channels=16, kernel_size=(1,5,5))
        self.r_max_pool_1 = nn.MaxPool3d(kernel_size=(1,4,4))
        self.r_conv_2 = nn.Conv3d(in_channels=16, out_channels=16*multiplier, kernel_size=(1,5,5))
        self.r_max_pool_2 = nn.MaxPool3d(kernel_size=(1,2,2))

        # Temporal pyramidal module
        self.r_pad_p_1 = (0,0,0,0,(self.pyramid_size_1-1)//2, self.pyramid_size_1-1-(self.pyramid_size_1-1)//2)
        self.r_pad_p_2 = (0,0,0,0,(self.pyramid_size_2-1)//2, self.pyramid_size_2-1-(self.pyramid_size_2-1)//2)
        self.r_pad_p_3 = (0,0,0,0,(self.pyramid_size_3-1)//2, self.pyramid_size_3-1-(self.pyramid_size_3-1)//2)
        self.r_pad_p_4 = (0,0,0,0,(self.pyramid_size_4-1)//2, self.pyramid_size_4-1-(self.pyramid_size_4-1)//2)
        self.r_conv_p_1 = nn.Conv3d(in_channels=16*multiplier, out_channels=4*multiplier, kernel_size=(self.pyramid_size_1,1,1))
        self.r_conv_p_2 = nn.Conv3d(in_channels=16*multiplier, out_channels=8*multiplier, kernel_size=(self.pyramid_size_2,1,1))
        self.r_conv_p_3 = nn.Conv3d(in_channels=16*multiplier, out_channels=16*multiplier, kernel_size=(self.pyramid_size_3,1,1))
        self.r_conv_p_4 = nn.Conv3d(in_channels=16*multiplier, out_channels=32*multiplier, kernel_size=(self.pyramid_size_4,1,1))
        self.r_maxpool_p_1 = nn.MaxPool3d(kernel_size=(1,self.pyramide_pool_size_h,self.pyramide_pool_size_w))
        self.r_maxpool_p_2 = nn.MaxPool3d(kernel_size=(1,self.pyramide_pool_size_h,self.pyramide_pool_size_w))
        self.r_maxpool_p_3 = nn.MaxPool3d(kernel_size=(1,self.pyramide_pool_size_h,self.pyramide_pool_size_w))
        self.r_maxpool_p_4 = nn.MaxPool3d(kernel_size=(1,self.pyramide_pool_size_h,self.pyramide_pool_size_w))
        self.r_maxpool_p_0 = nn.MaxPool3d(kernel_size=(1,self.pyramide_pool_size_h,self.pyramide_pool_size_w))

    def init_GCN(self,multiplier=1):

        # ---------------------
        # Representation branch
        # ---------------------
        # Base convolutional layers
        input_channel = self.input_channel
        # print("input_channel", input_channel)
        
            
        if self.args.backbone_player == "GCN":
            self.r_conv_1 = GCNConv(input_channel, 8*multiplier)
            self.r_conv_2 = GCNConv(8*multiplier, 16*multiplier)
            self.r_conv_3 = GCNConv(16*multiplier, 32*multiplier)
            self.r_conv_4 = GCNConv(32*multiplier, 76*multiplier)

        elif self.args.backbone_player == "EdgeConvGCN":
            self.r_conv_1 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*input_channel, 8*multiplier) ]))
            self.r_conv_2 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*8*multiplier, 16*multiplier) ]))
            self.r_conv_3 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*16*multiplier, 32*multiplier) ]))
            self.r_conv_4 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2*32*multiplier, 76*multiplier) ]))

        elif self.args.backbone_player == "DynamicEdgeConvGCN":
            self.r_conv_1 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*input_channel, 8*multiplier) ]), k=3)
            self.r_conv_2 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*8*multiplier, 16*multiplier) ]), k=3)
            self.r_conv_3 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*16*multiplier, 32*multiplier) ]), k=3)
            self.r_conv_4 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2*32*multiplier, 76*multiplier) ]), k=3)

        elif "resGCN" in self.args.backbone_player:
            # hidden_channels=64, num_layers=28
            # input_channel = 6
            output_channel = 76*multiplier
            hidden_channels = 64
            self.num_layers = int(self.args.backbone_player.split("-")[-1])

            self.node_encoder = nn.Linear(input_channel, hidden_channels)
            self.edge_encoder = nn.Linear(input_channel, hidden_channels)
            self.layers = torch.nn.ModuleList()
            for i in range(1, self.num_layers + 1):
                conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                            t=1.0, learn_t=True, num_layers=2, norm='layer')
                norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
                act = nn.ReLU(inplace=True)

                layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1,
                                    ckpt_grad=i % 3)
                self.layers.append(layer)

            self.lin = nn.Linear(hidden_channels, output_channel)



    def forward_2DConv(self, inputs):

        # -------------------------------------
        # Temporal Convolutional neural network
        # -------------------------------------

        # Base Convolutional Layers
        conv_1 = F.relu(self.conv_1(inputs))
        #print("Conv_1 size: ", conv_1.size())
        #print(torch.cuda.memory_allocated()/10**9)
        
        conv_2 = F.relu(self.conv_2(conv_1))
        #print("Conv_2 size: ", conv_2.size())
        #print(torch.cuda.memory_allocated()/10**9)


        # Temporal Pyramidal Module
        conv_p_1 = F.relu(self.conv_p_1(self.pad_p_1(conv_2)))
        #print("Conv_p_1 size: ", conv_p_1.size())
        #print(torch.cuda.memory_allocated()/10**9)
        conv_p_2 = F.relu(self.conv_p_2(self.pad_p_2(conv_2)))
        #print("Conv_p_2 size: ", conv_p_2.size())
        #print(torch.cuda.memory_allocated()/10**9)
        conv_p_3 = F.relu(self.conv_p_3(self.pad_p_3(conv_2)))
        #print("Conv_p_3 size: ", conv_p_3.size())
        #print(torch.cuda.memory_allocated()/10**9)
        conv_p_4 = F.relu(self.conv_p_4(self.pad_p_4(conv_2)))
        #print("Conv_p_4 size: ", conv_p_4.size())
        #print(torch.cuda.memory_allocated()/10**9)

        concatenation = torch.cat((conv_2,conv_p_1,conv_p_2,conv_p_3,conv_p_4),1)
        #print("Concatenation size: ", concatenation.size())
        #print(torch.cuda.memory_allocated()/10**9)
        
        return concatenation

    def forward_3DConv(self,representation_inputs):

        # --------------------
        # Representation input
        # --------------------
        #print("Representation input size: ", representation_inputs.size())

        #Base convolutional Layers

        r_conv_1 = F.relu(self.r_conv_1(representation_inputs))
        #print("r_conv_1 size: ", r_conv_1.size())
        #print(torch.cuda.memory_allocated()/10**9)
        r_conv_1_pooled = self.r_max_pool_1(r_conv_1)
        #print("r_conv_1_pooled size: ", r_conv_1_pooled.size())
        #print(torch.cuda.memory_allocated()/10**9)

        r_conv_2 = F.relu(self.r_conv_2(r_conv_1_pooled))
        #print("r_conv_2 size: ", r_conv_2.size())
        #print(torch.cuda.memory_allocated()/10**9)
        r_conv_2_pooled = self.r_max_pool_2(r_conv_2)
        #print("r_conv_2_pooled size: ", r_conv_2_pooled.size())
        #print(torch.cuda.memory_allocated()/10**9)

        # Temporal Pyramidal Module
        r_conv_p_1 = self.r_maxpool_p_1(F.relu(self.r_conv_p_1(F.pad(r_conv_2_pooled, self.r_pad_p_1, "constant", 0)))).squeeze(-1)
        #print("r_conv_p_1 size: ", r_conv_p_1.size())
        #print(torch.cuda.memory_allocated()/10**9)
        r_conv_p_2 = self.r_maxpool_p_2(F.relu(self.r_conv_p_2(F.pad(r_conv_2_pooled, self.r_pad_p_2, "constant", 0)))).squeeze(-1)
        #print("r_conv_p_2 size: ", r_conv_p_2.size())
        #print(torch.cuda.memory_allocated()/10**9)
        r_conv_p_3 = self.r_maxpool_p_3(F.relu(self.r_conv_p_3(F.pad(r_conv_2_pooled, self.r_pad_p_3, "constant", 0)))).squeeze(-1)
        #print("r_conv_p_3 size: ", r_conv_p_3.size())
        #print(torch.cuda.memory_allocated()/10**9)
        r_conv_p_4 = self.r_maxpool_p_4(F.relu(self.r_conv_p_4(F.pad(r_conv_2_pooled, self.r_pad_p_4, "constant", 0)))).squeeze(-1)
        #print("r_conv_p_4 size: ", r_conv_p_4.size())
        #print(torch.cuda.memory_allocated()/10**9)
        r_conv_2_pooled_pooled = self.r_maxpool_p_4(r_conv_2_pooled).squeeze(-1)
        #print("r_conv_2_pooled_pooled size: ", r_conv_2_pooled_pooled.size())
        #print(torch.cuda.memory_allocated()/10**9)

        r_concatenation = torch.cat((r_conv_2_pooled_pooled,r_conv_p_1,r_conv_p_2,r_conv_p_3,r_conv_p_4),1)
        #print("r_concatenation size: ", r_concatenation.size())
        #print(torch.cuda.memory_allocated()/10**9)

        return r_concatenation

    def forward_GCN(self, representation_inputs):

        # BS, _, T, C = representation_inputs.shape
        BS = self.args.batch_size
        T = self.chunk_size
        # --------------------
        # Representation input -> GCN
        # --------------------
        #print("Representation input size: ", representation_inputs.size())

        #Base convolutional Layers
        x, edge_index, batch = representation_inputs.x, representation_inputs.edge_index, representation_inputs.batch
        edge_attr = representation_inputs.edge_attr
        # batch_list = batch.tolist()

        batch_unique = list(set(batch.tolist()))
        batch_max = max(batch.tolist())
        # print("batch", len(batch_unique), batch_max, len(batch.tolist()))
        # list1 = batch_unique
        # list2 = [i for i in range(max(batch.tolist()))]
        # print("additional", set(list1).difference(list2))
        # print("missing", set(list2).difference(list1))
        if self.args.backbone_player == "GCN" or self.args.backbone_player == "EdgeConvGCN":
            x = F.relu(self.r_conv_1(x, edge_index))
            x = F.relu(self.r_conv_2(x, edge_index))
            x = F.relu(self.r_conv_3(x, edge_index))
            x = F.relu(self.r_conv_4(x, edge_index))
        elif "DynamicEdgeConvGCN" in self.args.backbone_player: #EdgeConvGCN or DynamicEdgeConvGCN
            x = F.relu(self.r_conv_1(x, batch))
            x = F.relu(self.r_conv_2(x, batch))
            x = F.relu(self.r_conv_3(x, batch))
            x = F.relu(self.r_conv_4(x, batch))
        elif "resGCN" in self.args.backbone_player: #EdgeConvGCN or DynamicEdgeConvGCN
            x = self.node_encoder(x)
            # edge_attr = self.edge_encoder(edge_attr)

            # x = self.layers[0].conv(x, edge_index, edge_attr)
            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers[1:]:
                # x = layer(x, edge_index, edge_attr)
                x = layer(x, edge_index)

            x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training)

            x = self.lin(x)
        # print("before max_pool", x.shape)
        x = global_max_pool(x, batch) 
        # print("after max_pool", x.shape)
        # print(batch)
        # BS = inputs.shape[0]

        # magic fix with zero padding
        expected_size = BS* T
        x = torch.cat([x, torch.zeros(expected_size-x.shape[0], x.shape[1]).to(x.device)], 0)

        x = x.reshape(BS, T, x.shape[1]) #BSxTxFS
        x = x.permute((0,2,1)) #BSxFSxT
        x = x.unsqueeze(-1) #BSxFSxTx1
        r_concatenation = x

        return r_concatenation

# class ContextAwareModelExtended(ContextAwareModel):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def forward(self, representation_inputs):

#         # -----------------------------------
#         # Feature input (chunks of the video)
#         # -----------------------------------
#         r_concatenation = self.forward_GNN(representation_inputs)
#         full_concatenation = r_concatenation
#         # print(f"Concatenation size: {full_concatenation.size()}")

#         # -------------------
#         # Segmentation module
#         # -------------------
#         conv_seg = self.conv_seg(self.pad_seg(full_concatenation))
#         # print("Conv_seg size: ", conv_seg.size())
#         # print(torch.cuda.memory_allocated()/10**9)

#         conv_seg_permuted = conv_seg.permute(0,2,3,1)
#         # print("Conv_seg_permuted size: ", conv_seg_permuted.size())
#         # print(torch.cuda.memory_allocated()/10**9)

#         conv_seg_reshaped = conv_seg_permuted.view(conv_seg_permuted.size()[0],conv_seg_permuted.size()[1],self.dim_capsule,self.num_classes)
#         # print("Conv_seg_reshaped size: ", conv_seg_reshaped.size())
#         # print(torch.cuda.memory_allocated()/10**9)


#         #conv_seg_reshaped_permuted = conv_seg_reshaped.permute(0,3,1,2)
#         #print("Conv_seg_reshaped_permuted size: ", conv_seg_reshaped_permuted.size())

#         conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
#         # print("Conv_seg_norm: ", conv_seg_norm.size())
#         # print(torch.cuda.memory_allocated()/10**9)


#         #conv_seg_norm_permuted = conv_seg_norm.permute(0,2,3,1)
#         #print("Conv_seg_norm_permuted size: ", conv_seg_norm_permuted.size())

#         output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm-0.5), dim=2)*4/self.dim_capsule)
#         # print("Output_segmentation size: ", output_segmentation.size())
#         # print(torch.cuda.memory_allocated()/10**9)
#         return output_segmentation, r_concatenation


###########################################################################
#                             SPOTTING MODULES                            #
###########################################################################

class SpottingModel(nn.Module):
    def __init__(self, args=None):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS: The spotting output with a shape (batch_size,chunk_size-receptive_field,num_classes)
        """
        super(SpottingModel, self).__init__()

        self.args = args
        self.num_classes = args.annotation_nr 
        self.dim_capsule = args.dim_capsule
        self.receptive_field = args.receptive_field*args.fps
        self.chunk_size = args.chunk_size*args.fps
        self.fps = args.fps
        self.input_channel = args.input_channel

        # ---------------------------------
        # Initialize the segmentation model
        # ---------------------------------

        self.model = torch.load(args.sgementation_path)
        self.model.eval()
        if args.freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        # -------------------
        # Spotting layers
        # ------------------- 

        # Apply pooling to smooth results for 1s each 
        self.max_pool_smooth = nn.MaxPool1d(kernel_size=self.fps*2+1, stride=self.fps, padding=4)
        # Captures the whole context
        # self.conv1d_layer_1 = nn.Conv1d(in_channels=self.num_classes-1, out_channels=2*self.num_classes, kernel_size=self.chunk_size/self.fps, stride=1, padding=0)
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        # Further convolutions layers
        self.conv1d_layer_1 = nn.Conv1d(in_channels=self.num_classes, out_channels=2*self.num_classes, kernel_size=5, stride=1, padding=2)
        self.conv1d_layer_2 = nn.Conv1d(in_channels=2*self.num_classes, out_channels=4*self.num_classes, kernel_size=5, stride=1, padding=2)
        self.conv1d_layer_3 = nn.Conv1d(in_channels=4*self.num_classes, out_channels=8*self.num_classes, kernel_size=5, stride=1, padding=2)
        self.dropout = nn.Dropout(p=0.2)

        # Classifier
    
        self.classifier = nn.Conv1d(in_channels=15*self.num_classes, out_channels=self.num_classes, kernel_size=1)
        # self.layer_normalisation = nn.LayerNorm(self.num_classes) 

    def forward(self, representation):

        # -------------------
        # Segmentation model
        # -------------------
        segmentation_output = self.model(representation)
        # Reverse the output to get the probabilities
        reversed_segmentation = 1 - segmentation_output
        # Remove the receptive field 
        main_field_segmentation = reversed_segmentation[:, int(self.receptive_field/2):-int(self.receptive_field/2), :]
        # Adjust the size for applying convolution layers
        reshaped_segmentation = main_field_segmentation.permute(0,2,1)
        # print("Smoothing size: ", reshaped_segmentation.size())

        # -------------------
        # Spotting layers
        # ------------------- 
        
        # Smoothen contextual information
        max_pool_smooth = self.max_pool_smooth(F.relu(reshaped_segmentation))
        # print("Smoothing size: ", max_pool_smooth.size())
        
        # Get the whole context
        # conv1d_layer_1 = self.conv1d_layer_1(self.dropout(max_pool_smooth))
        # conv1d_layer_1 = conv1d_layer_1.repeat_interleave(max_pool_smooth.shape[2], dim=2)
        global_avg_pooling = self.global_avg_pooling(F.relu(reshaped_segmentation))
        global_avg_pooling = global_avg_pooling.expand(-1, -1, max_pool_smooth.shape[2])
        # print("Whole context size: ", global_avg_pooling.size())

        # Get more detailed information
        conv1d_layer_1 = self.conv1d_layer_1(self.dropout(max_pool_smooth))
        # print("Conv1d_layer_2 size: ", conv1d_layer_2.size())

        conv1d_layer_2 = self.conv1d_layer_2(self.dropout(F.relu(conv1d_layer_1)))
        # print("Conv1d_layer_3 size: ", conv1d_layer_3.size())
        
        conv1d_layer_3 = self.conv1d_layer_3(self.dropout(F.relu(conv1d_layer_2)))
        # print("Conv1d_layer_4 size: ", conv1d_layer_4.size())

        # Concatenated information
        concatenation = torch.cat((global_avg_pooling, conv1d_layer_1, conv1d_layer_2, conv1d_layer_3), dim=1)

        # Classification
        spotting_output = self.classifier(self.dropout(F.relu(concatenation)))

        # Layer normalisation
        # spotting_output = self.layer_normalisation(spotting_output)

        return spotting_output.permute(0,2,1)


class FinetunedSpottingModel(nn.Module):
    def __init__(self, args=None):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS: The spotting output with a shape (batch_size,chunk_size-receptive_field,num_classes)
        """
        super(FinetunedSpottingModel, self).__init__()

        self.args = args
        self.num_classes = args.annotation_nr 
        self.dim_capsule = args.dim_capsule
        self.receptive_field = args.receptive_field*args.fps
        self.chunk_size = args.chunk_size*args.fps
        self.fps = args.fps
        self.input_channel = args.input_channel

        # ---------------------------------
        # Initialize the segmentation model
        # ---------------------------------
        self.models = []
        for path in args.sgementation_path:
            model = torch.load(path)
            model.eval()
            if args.freeze_model:
                for param in model.parameters():
                    param.requires_grad = False
            self.models.append(model)

        # -------------------
        # Spotting layers
        # ------------------- 

        # Apply pooling to smooth results for 1s each 
        self.max_pool_smooth = nn.MaxPool1d(kernel_size=self.fps*2+1, stride=self.fps, padding=4)
        # Captures the whole context
        # self.conv1d_layer_1 = nn.Conv1d(in_channels=self.num_classes-1, out_channels=2*self.num_classes, kernel_size=self.chunk_size/self.fps, stride=1, padding=0)
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        # Further convolutions layers
        self.conv1d_layer_1 = nn.Conv1d(in_channels=self.num_classes, out_channels=2*self.num_classes, kernel_size=5, stride=1, padding=2)
        self.conv1d_layer_2 = nn.Conv1d(in_channels=2*self.num_classes, out_channels=4*self.num_classes, kernel_size=5, stride=1, padding=2)
        self.conv1d_layer_3 = nn.Conv1d(in_channels=4*self.num_classes, out_channels=8*self.num_classes, kernel_size=5, stride=1, padding=2)
        self.dropout = nn.Dropout(p=0.2)

        # Classifier
    
        self.classifier = nn.Conv1d(in_channels=15*self.num_classes, out_channels=self.num_classes, kernel_size=1)
        # self.layer_normalisation = nn.LayerNorm(self.num_classes) 

    def forward(self, representation):

        # -------------------
        # Segmentation model
        # -------------------
        segmentation_output = torch.empty((self.args.batch_size, self.chunk_size, 0))
        
        for model in self.models:
            output = model(representation)
            segmentation_output = torch.cat([segmentation_output, output], dim=-1)
        # Reverse the output to get the probabilities
        reversed_segmentation = 1 - segmentation_output
        # Remove the receptive field 
        main_field_segmentation = reversed_segmentation[:, int(self.receptive_field/2):-int(self.receptive_field/2), :]
        # Adjust the size for applying convolution layers
        reshaped_segmentation = main_field_segmentation.permute(0,2,1)
        # print("Smoothing size: ", reshaped_segmentation.size())

        # -------------------
        # Spotting layers
        # ------------------- 
        
        # Smoothen contextual information
        max_pool_smooth = self.max_pool_smooth(F.relu(reshaped_segmentation))
        # print("Smoothing size: ", max_pool_smooth.size())
        
        # Get the whole context
        # conv1d_layer_1 = self.conv1d_layer_1(self.dropout(max_pool_smooth))
        # conv1d_layer_1 = conv1d_layer_1.repeat_interleave(max_pool_smooth.shape[2], dim=2)
        global_avg_pooling = self.global_avg_pooling(F.relu(reshaped_segmentation))
        global_avg_pooling = global_avg_pooling.expand(-1, -1, max_pool_smooth.shape[2])
        # print("Whole context size: ", global_avg_pooling.size())

        # Get more detailed information
        conv1d_layer_1 = self.conv1d_layer_1(self.dropout(max_pool_smooth))
        # print("Conv1d_layer_2 size: ", conv1d_layer_2.size())

        conv1d_layer_2 = self.conv1d_layer_2(self.dropout(F.relu(conv1d_layer_1)))
        # print("Conv1d_layer_3 size: ", conv1d_layer_3.size())
        
        conv1d_layer_3 = self.conv1d_layer_3(self.dropout(F.relu(conv1d_layer_2)))
        # print("Conv1d_layer_4 size: ", conv1d_layer_4.size())

        # Concatenated information
        concatenation = torch.cat((global_avg_pooling, conv1d_layer_1, conv1d_layer_2, conv1d_layer_3), dim=1)

        # Classification
        spotting_output = self.classifier(self.dropout(F.relu(concatenation)))

        # Layer normalisation
        # spotting_output = self.layer_normalisation(spotting_output)

        return spotting_output.permute(0,2,1)


# class SpottingModel(nn.Module):
#     def __init__(self, args=None):
#         """
#         INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
#         OUTPUTS: The spotting output with a shape (batch_size,chunk_size-receptive_field,num_classes)
#         """
#         super(SpottingModel, self).__init__()

#         self.args = args
#         self.num_classes = args.annotation_nr 
#         self.dim_capsule = args.dim_capsule
#         self.receptive_field = args.receptive_field*args.fps
#         self.chunk_size = args.chunk_size*args.fps
#         self.fps = args.fps
#         self.input_channel = args.input_channel

#         # ---------------------------------
#         # Initialize the segmentation model
#         # ---------------------------------
#         # Load the original trained model
#         original_model = torch.load(args.sgementation_path)

#         # Instantiate the extended model with features 
#         self.model = ContextAwareModelExtended(args=args)
#         self.model.load_state_dict(original_model.state_dict())
#         self.model.eval()
#         if args.freeze_model:
#             for param in self.model.parameters():
#                 param.requires_grad = False

#         # -------------------
#         # Spotting layers
#         # ------------------- 

#         # Apply pooling to smooth results for 1s each 
#         self.max_pool_smooth = nn.MaxPool1d(kernel_size=self.fps*2+1, stride=self.fps, padding=4)
#         # Captures the whole context
#         # self.conv1d_layer_1 = nn.Conv1d(in_channels=self.num_classes-1, out_channels=2*self.num_classes, kernel_size=self.chunk_size/self.fps, stride=1, padding=0)
#         self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

#         # Further convolutions layers
#         self.conv1d_layer_1 = nn.Conv1d(in_channels=self.num_classes, out_channels=2*self.num_classes, kernel_size=5, stride=1, padding=2)
#         self.conv1d_layer_2 = nn.Conv1d(in_channels=2*self.num_classes, out_channels=4*self.num_classes, kernel_size=5, stride=1, padding=2)
#         self.conv1d_layer_3 = nn.Conv1d(in_channels=4*self.num_classes, out_channels=8*self.num_classes, kernel_size=5, stride=1, padding=2)
#         self.dropout = nn.Dropout(p=0.2)

#         # Classifier
    
#         self.classifier = nn.Conv1d(in_channels=15*self.num_classes, out_channels=self.num_classes, kernel_size=1)
#         # self.layer_normalisation = nn.LayerNorm(self.num_classes) 

#     def forward(self, representation):

#         # -------------------
#         # Segmentation model
#         # -------------------
#         segmentation_output = self.model(representation)
#         # Reverse the output to get the probabilities
#         reversed_segmentation = 1 - segmentation_output
#         # Remove the receptive field 
#         main_field_segmentation = reversed_segmentation[:, int(self.receptive_field/2):-int(self.receptive_field/2), :]
#         # Adjust the size for applying convolution layers
#         reshaped_segmentation = main_field_segmentation.permute(0,2,1)
#         # print("Smoothing size: ", reshaped_segmentation.size())

#         # -------------------
#         # Features layers
#         # ------------------- 

#         # -------------------
#         # Spotting layers
#         # ------------------- 
        
#         # Smoothen contextual information
#         max_pool_smooth = self.max_pool_smooth(F.relu(reshaped_segmentation))
#         # print("Smoothing size: ", max_pool_smooth.size())
        
#         # Get the whole context
#         # conv1d_layer_1 = self.conv1d_layer_1(self.dropout(max_pool_smooth))
#         # conv1d_layer_1 = conv1d_layer_1.repeat_interleave(max_pool_smooth.shape[2], dim=2)
#         global_avg_pooling = self.global_avg_pooling(F.relu(reshaped_segmentation))
#         global_avg_pooling = global_avg_pooling.expand(-1, -1, max_pool_smooth.shape[2])
#         # print("Whole context size: ", global_avg_pooling.size())

#         # Get more detailed information
#         conv1d_layer_1 = self.conv1d_layer_1(self.dropout(max_pool_smooth))
#         # print("Conv1d_layer_2 size: ", conv1d_layer_2.size())

#         conv1d_layer_2 = self.conv1d_layer_2(self.dropout(F.relu(conv1d_layer_1)))
#         # print("Conv1d_layer_3 size: ", conv1d_layer_3.size())
        
#         conv1d_layer_3 = self.conv1d_layer_3(self.dropout(F.relu(conv1d_layer_2)))
#         # print("Conv1d_layer_4 size: ", conv1d_layer_4.size())

#         # Concatenated information
#         concatenation = torch.cat((global_avg_pooling, conv1d_layer_1, conv1d_layer_2, conv1d_layer_3), dim=1)

#         # Classification
#         spotting_output = self.classifier(self.dropout(F.relu(concatenation)))

#         # Layer normalisation
#         # spotting_output = self.layer_normalisation(spotting_output)

#         return spotting_output.permute(0,2,1)

class EnsembleSpottingModel(nn.Module):
    def __init__(self, ensemble_models, args=None):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS: The spotting output with a shape (batch_size,chunk_size-receptive_field,num_classes)
        """
        super(EnsembleSpottingModel, self).__init__()

        self.args = args
        self.num_classes = args.annotation_nr 
        self.dim_capsule = args.dim_capsule
        self.receptive_field = args.receptive_field*args.fps
        self.chunk_size = args.chunk_size*args.fps
        self.fps = args.fps
        self.input_channel = args.input_channel

        # ---------------------------------
        # Initialize segmentation models
        # ---------------------------------
        self.models = []
        for model_path, update in ensemble_models:
            # Load model
            model = torch.load(model_path)
            # Update models arguments
            for key, value in update.items():
                setattr(model.args, key, value)
            self.models.append(model)
             
        
        if args.freeze_model:
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = False

        # -------------------
        # Spotting layers
        # ------------------- 

        # Apply pooling to smooth results for 1s each 
        self.max_pool_smooth = nn.MaxPool2d(kernel_size=(self.fps*2+1,1), stride=(self.fps, 1), padding=(4,0))
        # Captures the whole context
        # self.conv1d_layer_1 = nn.Conv1d(in_channels=self.num_classes-1, out_channels=2*self.num_classes, kernel_size=self.chunk_size/self.fps, stride=1, padding=0)
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        # Further convolutions layers
        self.conv2d_layer_1 = nn.Conv2d(in_channels=self.num_classes, out_channels=2*self.num_classes, kernel_size=(5,1), stride=1, padding=(2,0))
        self.conv2d_layer_2 = nn.Conv2d(in_channels=2*self.num_classes, out_channels=4*self.num_classes, kernel_size=(5,1), stride=1, padding=(2,0))
        self.conv2d_layer_3 = nn.Conv2d(in_channels=4*self.num_classes, out_channels=8*self.num_classes, kernel_size=(5,1), stride=1, padding=(2,0))
        self.dropout = nn.Dropout(p=0.2)

        # Classifier
        self.classifier = nn.Conv2d(in_channels=15*self.num_classes, out_channels=self.num_classes, kernel_size=(1,1))
        self.layer_normalisation = nn.LayerNorm([args.chunk_size-args.receptive_field, len(self.models)]) 

    def forward(self, representation):

        # -------------------
        # Segmentation model
        # -------------------
        # Generate models outputs
        segmentations_outputs = torch.empty((self.args.batch_size, self.chunk_size, self.num_classes, 0))
        for model in self.models:
            segmentation_output = model(representation)
            segmentation_output = segmentation_output.unsqueeze(-1)
            segmentations_outputs = torch.cat((segmentations_outputs, segmentation_output), dim=-1)
        print("Segmentation outputs size:", {segmentations_outputs.shape})
        
        # Reverse the output to get the probabilities
        reversed_segmentation = 1 - segmentations_outputs
        # Remove the receptive field 
        main_field_segmentation = reversed_segmentation[:, int(self.receptive_field/2):-int(self.receptive_field/2), :, :]
        # Adjust the size for applying convolution layers
        permuted_segmentation = main_field_segmentation.permute(0,2,1,3)
        print("Permuted segmentation size: ", permuted_segmentation.size())

        # -------------------
        # Spotting layers
        # ------------------- 
        
        # Smoothen contextual information
        max_pool_smooth = self.max_pool_smooth(F.relu(permuted_segmentation))
        print("Smoothing size: ", max_pool_smooth.size())
        
        # Get the whole context
        # conv1d_layer_1 = self.conv1d_layer_1(self.dropout(max_pool_smooth))
        # conv1d_layer_1 = conv1d_layer_1.repeat_interleave(max_pool_smooth.shape[2], dim=2)
        reshaped_segmentation = permuted_segmentation.reshape(-1, permuted_segmentation.shape[1], permuted_segmentation.shape[2])
        print("Reshaped_segmentation size: ", reshaped_segmentation.size())
        global_avg_pooling = self.global_avg_pooling(F.relu(reshaped_segmentation))
        print("Global_avg_pooling size: ", reshaped_segmentation.size())

        # Get back models dimension
        global_avg_pooling = global_avg_pooling.reshape(permuted_segmentation.shape[0], permuted_segmentation.shape[3], permuted_segmentation.shape[1], 1)
        print("Reshaped global_avg_pooling size: ", reshaped_segmentation.size())
        global_avg_pooling = global_avg_pooling.permute(0, 2, 3, 1)
        print("Permuted global_avg_pooling size: ", reshaped_segmentation.size())

        global_avg_pooling = global_avg_pooling.expand(-1, -1, max_pool_smooth.shape[2], -1)
        print("Whole context size: ", global_avg_pooling.size())

        # Get more detailed information
        conv2d_layer_1 = self.conv2d_layer_1(self.dropout(max_pool_smooth))
        print("Conv1d_layer_2 size: ", conv2d_layer_1.size())

        conv2d_layer_2 = self.conv2d_layer_2(self.dropout(F.relu(conv2d_layer_1)))
        print("Conv2d_layer_2 size: ", conv2d_layer_2.size())
        
        conv2d_layer_3 = self.conv2d_layer_3(self.dropout(F.relu(conv2d_layer_2)))
        print("Conv2d_layer_3 size: ", conv2d_layer_3.size())

        # Concatenated information
        concatenation = torch.cat((global_avg_pooling, conv2d_layer_1, conv2d_layer_2, conv2d_layer_3), dim=1)
        print("Concatenation size: ", conv2d_layer_3.size())

        # Classification
        spotting_output = self.classifier(self.dropout(F.relu(concatenation)))
        print("Spotting size: ", conv2d_layer_3.size())

        # Layer normalisation
        spotting_output = self.layer_normalisation(spotting_output)
        print("Normalisation size: ", conv2d_layer_3.size())

        # Mean results
        spotting_output = torch.mean(spotting_output, dim=3)
        print("Average probability size: ", conv2d_layer_3.size())

        return spotting_output.permute(0,2,1)
    
# from dataclasses import dataclass
# from helpers.classes import get_K_params
# @dataclass
# class Args:
#     # DATA
#     chunk_size = 60
#     batch_size = 32
#     input_channel = 13
#     annotation_nr = 10
#     receptive_field = 12
#     fps = 5
#     K_parameters = get_K_params(chunk_size)
#     focused_annotation = None
#     generate_augmented_data = True
#     class_split = "alive"
#     generate_artificial_targets = False
    
#     # TRAINING
#     chunks_per_epoch = 1824
#     lambda_coord=5.0
#     lambda_noobj=0.5
#     patience=25
#     LR=1e-03
#     max_epochs=180
#     GPU=0 
#     max_num_worker=1
#     loglevel='INFO'
    
#     # SEGMENTATION MODULE
#     feature_multiplier=1
#     backbone_player = "GCN"
#     load_weights=None
#     model_name="Testing_Model"
#     dim_capsule=16
#     # VLAD pooling if applicable
#     vocab_size=None
#     pooling=None

#     # SPOTTING MODULE
#     sgementation_path = None
#     freeze_model = True
#     spotting_fps = 1
#     ensemble_models = [
#         "models/edge_attr_GCN.pth.tar",
#         "models/backbone_GIN.pth.tar"
#     ]


# GCN_updates = {"backbone_player":"GCN"}
# GIN_updates = {"backbone_player":"GIN"}

# ensemble_models = [
#     ("models/edge_attr_GCN.pth.tar", GCN_updates),
#     ("models/backbone_GIN.pth.tar", GIN_updates)
#     ]

# for (model_path, arg) in ensemble_models:
#     args = arg
#     model = torch.load(model_path)
# args = Args()
# model = torch.load(ensemble_models[1][0])
# model.args.backbone_player = "GIN"
# model(x)
# del Args
# from data_management.DataManager import CALFData, collateGCN
# args = Args()
# model = EnsembleSpottingModel(ensemble_models, args=args)
# collate_fn = collateGCN
# train_dataset = CALFData(split="train", args=args)
# train_loader = torch.utils.data.DataLoader(train_dataset,
#             batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

# _,_,x = next(iter(train_loader))
# res = model(x)