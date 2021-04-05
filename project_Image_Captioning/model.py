import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param embed_size: The size of embeddings, should you choose to use them        
        :param hidden_size: The size of the hidden layer outputs
        :param num_layers : number of layers
        """
        super(DecoderRNN, self).__init__()
        dropout=0.3
        # set class variables
        self.num_layers=num_layers
        self.hidden_dim=hidden_size
        self.vocab_size=vocab_size

        # embedding and LSTM layers
      
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                        dropout=dropout, batch_first=True)



        #define fully connected Layer
        self.fc =nn.Linear(hidden_size,vocab_size)
    
    def forward(self, features, captions):
        """
        Forward propagation of the neural network
        :param features: The input to the neural network
        :return: Two Tensors, the output of the neural network and the latest hidden(captions) state
        """
        # TODO: Implement function   
        batch_size = features.size(0)
        seq_length = captions.shape[1]#features.size(1)
        # embeddings and lstm_out
        embeds = self.embedding(captions[:, :-1])
        #concatenate captions and features
        embeds = torch.cat((features.unsqueeze(dim = 1), embeds), dim = 1)
        lstm_out,hidden = self.lstm(embeds)
        
        # stack up lstm outputs
        # reshape into (batch_size, seq_length, output_size)
        #output = output.view(output.size()[0]*output.size()[1], self.hidden_dim)
        output = self.fc(lstm_out)
       
        # get last batch
        #output = output[:, -1]

        # return one batch of output word scores and the hidden state
        return output
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass