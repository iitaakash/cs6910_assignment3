import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_OUTPUT_CHARS = 32
class AttentionModel(nn.Module):
    
    def __init__(self, input_size = 48, hidden_size = 256, output_size = 29, droupout = 0.1):
        super(AttentionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.encoder_rnn_cell = nn.GRU(input_size, hidden_size, batch_first=True)
        self.decoder_rnn_cell = nn.GRU(hidden_size*2, hidden_size, batch_first=True)
        
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        
        # self.dropout_p = droupout

        # self.dropout = nn.Dropout(self.dropout_p)
        
        self.U = nn.Linear(self.hidden_size, self.hidden_size)
        self.W = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, 1)
        self.out2hidden = nn.Linear(self.output_size, self.hidden_size)   
        
    def forward(self, input, max_output_chars = MAX_OUTPUT_CHARS, device = 'cpu', ground_truth = None):
        
        # encoder
        encoder_outputs, hidden = self.encoder_rnn_cell(input)
        # input -> [N, letters, features]
        # encoder_outputs -> [N, letters, hidden_size]
        # hidden -> [1, N, hidden_size]
        # encoder_outputs = encoder_outputs.reshape(-1, self.hidden_size)
        # encoder_outputs -> [N * letters, hidden_size][64, hidden_size]
        # hidden = hidden.permute(1,0,2)
        # hidden -> [N, 1, hidden_size]
        
        
        # decoder
        decoder_state = hidden
        decoder_input = torch.zeros(input.shape[0], 1, self.output_size).to(device)
        decoder_input[:,:,0] = 1
        # here N = 1
        # input -> [N, letters, features]
        # out -> [N, letters, hidden_size]
        # hidden -> [N, 1, hidden_size]
        
        outputs = []
        U = self.U(encoder_outputs)
        # U = self.dropout(U)
        # U -> [N, letters, hidden_size] 
        # # [64 -> N * MAX_OUTPUT_CHARS]

        att_maps = []

        reset_var = torch.zeros((1, input.shape[0], 1), dtype= torch.bool).to(device)
        
        for i in range(max_output_chars):
            # print(decoder_state.view(1, -1).repeat(encoder_outputs.shape[0], 1).shape)
            # [64,512]
            # W = self.W(decoder_state.view(1, -1).repeat(1, encoder_outputs.shape[1], 1))
            W = self.W(decoder_state.permute(1,0,2))
            # W = self.dropout(W)
            
            V = self.attn(torch.tanh(U + W))
            attn_weights = F.softmax(V, dim = 1).permute(0,2,1)
            # [N , 1, letter]

            att_maps.append(attn_weights)
            
            
            attn_applied = torch.bmm(attn_weights,
                                 encoder_outputs)
            # [N, 1, hidden_size]

            
            embedding = self.out2hidden(decoder_input)
            # [N, 1, hidden_size]
            
            decoder_input = torch.cat((embedding, attn_applied), 2)
            # [N, 1, 2* hidden_size]

            out, decoder_state = self.decoder_rnn_cell(decoder_input, decoder_state)
                
            out = self.h2o(decoder_state)
            out = self.softmax(out)
            # out -> [1,N,output_size]
            outputs.append(out.squeeze(0))
            
            # teacher forcing
            max_idx = torch.argmax(out, 2, keepdim=True)
            
            if not ground_truth is None:
                new_id = torch.argmax(ground_truth[:,i,:], dim =1, keepdim=True).unsqueeze(0).to(device)
                cond = (max_idx != new_id) & (reset_var == False)
                reset_var[cond] = True
                max_idx[cond] = new_id[cond]
            
            one_hot = torch.zeros(out.shape, device=device)
            one_hot.scatter_(2, max_idx, 1) 
            one_hot = one_hot.permute(1,0,2)
            
            decoder_input = one_hot.detach()
            
        return outputs, att_maps



if __name__ == "__main__":
    from utils import *
    import time

    model = AttentionModel()
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
