import __future__
import torch
import torch.nn as nn
import torch.nn.functional as F



class SpottingModel(nn.Module):
    def __init__(self, args=None):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS: The spotting output with a shape (batch_size,chunk_size-receptive_field,num_classes)
        """
        super(SpottingModel, self).__init__()

        self.args = args
        self.num_classes = args.annotation_nr # Additional None annotation to use softmax
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

        return spotting_output.permute(0,2,1)





