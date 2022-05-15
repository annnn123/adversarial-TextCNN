import torch
import torch.nn as nn
import torch.nn.functional as F



class TextCNN(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    from "Convolutional Neural Networks for Sentence Classification"
    """
    def __init__(self, config):
        super(TextCNN, self).__init__()

        self.config = config
        self.init_method = config.init_method

        if config.embedding_type == 'rand':
            self.embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size, padding_idx=config.padding_idx)
        elif config.embedding_type == 'static':
            self.embeddings = nn.Embedding.from_pretrained(config.pretrained_embeddings, freeze=True, padding_idx=config.padding_idx)
        elif config.embedding_type == 'non_static':
            self.embeddings = nn.Embedding.from_pretrained(config.pretrained_embeddings, freeze=False, padding_idx=config.padding_idx)
        elif config.embedding_type == 'multichannel':
            pass
        else:
            raise ValueError('embedding type is invalid')

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=config.in_channels, out_channels=config.num_filters,
                                              kernel_size=(kernel_size, config.embedding_size)) for kernel_size in config.kernel_sizes])
        self.fc_layer = nn.Linear(config.num_filters * len(config.kernel_sizes), config.num_classes)
        self.dropout = nn.Dropout(config.dropout_rate)


        self.reset_parameters()


    def reset_parameters(self):
        """Initialize the transition parameters.
        """
        for n, p in self.named_parameters():
            if 'weight' in n:
                if self.init_method == 'xavier':
                    nn.init.xavier_normal_(p)
                elif self.init_method == 'kaiming':
                    nn.init.kaiming_normal_(p)
                elif self.init_method == 'normal':
                    nn.init.normal_(p, 0, 0.1)
                elif self.init_method == 'uniform':
                    nn.init.uniform_(p, -0.1, 0.1)
                else:
                    pass
            elif 'bias' in n:
                nn.init.constant_(p, 0)
            else:
                pass


    def conv_and_maxpooling(self, x, conv):
        x = F.relu(conv(x).squeeze())
        x = F.max_pool1d(x, x.size(-1)).squeeze()
        return x


    def forward(self, input_ids):

        x = self.embeddings(input_ids)
        x = x.unsqueeze(1)

        x = [self.conv_and_maxpooling(x, conv) for conv in self.convs]
        x = torch.cat(x, dim=-1)
        x = self.fc_layer(self.dropout(x))
        logits = F.log_softmax(x, dim=-1)

        return logits


    @property
    def device(self):
        return self.embeddings.weight.device

    def save(self, model_save_path):
        save_dict = {'state dict': self.state_dict(), 'config': self.config}
        torch.save(save_dict, model_save_path)
        # print(f"Save model at {model_save_path}")

    @staticmethod
    def load(model_save_path):
        save_dict = torch.load(model_save_path, map_location=lambda storage, loc: storage)
        model = TextCNN(save_dict['config'])
        model.load_state_dict(save_dict['state dict'])
        return model