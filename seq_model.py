import torch
import torch.nn as nn

MAX_OUTPUT_CHARS = 32
class NormalModel(nn.Module):
    
    def __init__(self, input_size = 48, hidden_size = 256, output_size = 29, seq = nn.GRU, dropout = 0):
        super(NormalModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.encoder_rnn_cell = seq(input_size, hidden_size, num_layers = 2 , batch_first=True, dropout = dropout)
        self.decoder_rnn_cell = seq(output_size, hidden_size,num_layers = 2, batch_first=True, dropout = dropout)
        
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, input, max_output_chars = MAX_OUTPUT_CHARS, device = 'cpu', ground_truth = None):
        
        # encoder
        out, hidden = self.encoder_rnn_cell(input)
        # here N = 1
        # input -> [N, letters, features]
        # out -> [N, letters, hidden_size]
        # hidden -> [num_layers, N, hidden_size]
        
        # decoder
        decoder_state = hidden
        decoder_input = torch.zeros(input.shape[0], 1, self.output_size).to(device) # also can be variable i.e learned from data.
        decoder_input[:,:,0] = 1

        outputs = []

        reset_var = torch.zeros((1, input.shape[0], 1), dtype= torch.bool).to(device)
        
        for i in range(max_output_chars):
            out, decoder_state = self.decoder_rnn_cell(decoder_input, decoder_state)
            # print(decoder_state.shape)
            # decoder_state -> [num_layers,N,hidden_size]
            out = self.h2o(decoder_state)
            # out -> [num_layers,N,output_size]
            out = out[-1,:,:].unsqueeze(0)
            out = self.softmax(out)
            # out -> [num_layers,N,output_size]
            outputs.append(out.squeeze(0))
            # out -> [num_layers,N,output_size]
            
            max_idx = torch.argmax(out, 2, keepdim=True)
            # max_idx -> [1,N,1]

            if not ground_truth is None:
                new_id = torch.argmax(ground_truth[:,i,:], dim =1, keepdim=True).unsqueeze(0).to(device)
                cond = (max_idx != new_id) & (reset_var == False)
                reset_var[cond] = True
                max_idx[cond] = new_id[cond]
                
            one_hot = torch.FloatTensor(out.shape).to(device)
            one_hot.zero_() # all the elements will be 0.
            one_hot.scatter_(2, max_idx, 1)
            one_hot = one_hot.permute(1,0,2)
            
            decoder_input = one_hot.detach() # don't pass gradient with this tensor.
            
        return outputs, []



if __name__ == "__main__":
    import time
    from utils import *

    model = NormalModel()
    print(model)
    print(f"Total model params for training : {TotalModelParams(model):,}")
    data = torch.rand((3, 32, 48), dtype=torch.float, requires_grad=False)
    print(data.shape)
    model = model.eval()
    for i in range(5):
        start = time.time()
        out = model.forward(data)
        end = time.time()
        print(end - start)
    
    # print(out.shape)
    # for ou in out:
        # print(ou.size())
        # max_idx = torch.argmax(ou, 1, keepdim=True)
        # print(max_idx)
    # print(len(out))

    # torch.save(model, './logs/best_model.pth')
    # print('Model saved!')
