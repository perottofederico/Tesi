import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import config

np.random.seed(45)
torch.manual_seed(45)

# class CNNEEGFeatureExtractor(nn.Module):
#     def __init__(self, input_shape, feat_dim=256, projection_dim=256, num_filters=[16, 32, 64, 128, 256, 512], kernel_sizes=[7, 7, 7, 5, 5, 3], strides=[2, 2, 2, 2, 2, 2], padding=[1, 1, 1, 1, 1, 1]):
#         super(CNNEEGFeatureExtractor, self).__init__()

#         # Define the convolutional layers
#         self.layers = nn.ModuleList()
#         in_channels = input_shape[0]
#         for i, out_channels in enumerate(num_filters):
#             self.layers.append( 
#                                 nn.Sequential(
#                                     nn.Conv2d(in_channels, out_channels, kernel_sizes[i], strides[i], padding[i]),\
#                                     nn.LeakyReLU(inplace=True),\
#                                     nn.BatchNorm2d(out_channels)
#                                 )
#                              )
#             in_channels = out_channels

#         # Calculate the size of the flattened output
#         with torch.no_grad():
#             x = torch.zeros(1, *input_shape)
#             for layer in self.layers:
#                 x = layer(x)
#             self.num_flat_features = x.view(1, -1).size(1)

#         # Define the fully connected layers
#         self.feat_layers = nn.Sequential(
#             nn.Linear(self.num_flat_features, feat_dim),
#             nn.LeakyReLU(inplace=True),
#         )

#         # Define the fully connected layers
#         self.proj_layers = nn.Sequential(
#             nn.Linear(feat_dim, 1024),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(1024, projection_dim),
#         )

#     def forward(self, x):
#         # Apply the convolutional layers
#         for layer in self.layers:
#             x = layer(x)
#             # print(x.shape)

#         # Flatten the output
#         x = x.view(-1, self.num_flat_features)
#         # print(x.shape)

#         # Apply the fully connected layers
#         feat = self.feat_layers(x)
#         x    = self.proj_layers(feat)
#         # print(x.shape)
#         return x, feat


class EEGFeatNet(nn.Module):
    def __init__(self, n_classes=40, in_channels=128, n_features=128, projection_dim=512, num_layers=1):
        super(EEGFeatNet, self).__init__()
        self.hidden_size= n_features
        self.num_layers = num_layers
        # self.embedding  = nn.Embedding(num_embeddings=in_channels, embedding_dim=n_features)
        self.encoder    = nn.LSTM(input_size=in_channels, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc         = nn.Linear(in_features=n_features, out_features=projection_dim, bias=False)
        # self.feat_layer = nn.Linear(in_features=self.hidden_size, out_features=n_features)
        # self.proj_layer = nn.Sequential(
        #                                     nn.LeakyReLU(),
        #                                     nn.Linear(in_features=n_features, out_features=projection_dim)
        #                                 )
    def forward(self, x):

        h_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        c_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


        _, (h_n, c_n) = self.encoder( x, (h_n, c_n) )

        feat = h_n[-1]
        x = self.fc(feat)
        
        # Uncomment this while training with triplet loss
        x = F.normalize(x, dim=-1)

        # print(x.shape, feat.shape)
        return x, feat #return the feat since linear projection and normalization is done by gram







#################################################


class EEGFeatNetMultiHead(nn.Module):
    def __init__(self,
                 in_channels=128,
                 hidden_size=512,
                 projection_dim = 512,
                 num_layers=1,
                 num_heads=8,
                 dropout=0.1):
        super(EEGFeatNetMultiHead, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.projection_dim = projection_dim

        # LSTM encoder
        self.encoder = nn.LSTM(
            input_size=in_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
            batch_first=True
        )

        # attention 
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True, 
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(hidden_size)   # norm for query token
        self.norm_out = nn.LayerNorm(hidden_size) # norm after residual

        #self._init_weights()

    def forward(self, x, conditioning = None):
        """
        x is the EEG signal of shape  [B, 440,128] 
        conditioning is the concatenation of image and text embeddings, with shape [B, contrastive_dim]
        """

        batch_size = x.size(0)

        # LSTM encoder
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        lstm_out, (h_n, c_n) = self.encoder(x, (h0, c0))  

        if conditioning is not None: #only use cross attention if we have the conditioning input
            #TODO should account for text padding ?
            query = lstm_out
            query = self.norm_q(query)  
            attn_output, attn_weights = self.cross_attn(query, conditioning, conditioning) # use the eeeg as query and the conditioning as keys/values
            
            feat = self.norm_out(query + self.dropout(attn_output))  # residual + ln
            
            # print("feat shape (before pooling): ", feat.shape)
            feat = feat.mean(dim=1) # pooling to get [b ,512] and feed to itm head

            return feat, attn_weights  # return attention weights for debugging/viz maybe


        else:
            #use last hidden state from lstm as standard
            feat = h_n[-1]
            return  feat  
    
#################################################
