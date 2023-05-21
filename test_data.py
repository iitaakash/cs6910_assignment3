import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
from Meter import *
from utils import *
from Aksharantar import *



# params
device = GetDevice()
model = torch.load("best_model.pth")
model = model.to(device)

# define loss function
# loss = nn.NLLLoss(ignore_index = -1)
loss = nn.CrossEntropyLoss()

print(f"Using device : {device}")


for ascii in Aksharantar.tamil_ascii:
    print(f"{ascii} -> {chr(ascii)}")

# 2950,2965,3006,2999,3021

input = str(input("Type the tamil letter in ascii, comma seperated : "))
# input = "2950,2965,3006,2999,3021"
input = input.strip().replace(" ", "").split(",")
input = [chr(eval(i)) for i in input]
tamil = "".join(input)
print(tamil)

input = encode_input(tamil)
input = input.to(device)
output_size = input.shape[1]

with torch.no_grad():
    outputs,_ = model(input, output_size, device)


out_vec = torch.stack(outputs, axis = 1)

out_text = decode_out(out_vec)
print(out_text)


