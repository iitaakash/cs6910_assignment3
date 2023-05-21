import os
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from Meter import *
from utils import *
from Aksharantar import Aksharantar as MyDataset
from att_model import AttentionModel 
from seq_model import NormalModel 

# Get dataset
train_dataset = MyDataset(mode = 'train')
val_dataset = MyDataset(mode = 'val')




# params
num_epochs = 10000
learning_rate = 0.006
train_batch_size = 200
val_batch_size = 500
device = GetDevice()

# data loaders
train_dataloader = DataLoader(train_dataset, train_batch_size, shuffle = True, drop_last= False)
val_dataloader = DataLoader(val_dataset, val_batch_size, shuffle = False, drop_last= False)


# define the model here 
model = AttentionModel()
# model = NormalModel()
# model = torch.load("best_model.pth")


# define loss function
loss = nn.CrossEntropyLoss()

# define optimizer & scheduler
optimizer = torch.optim.Adam([ dict(params=model.parameters(), lr=learning_rate), ])
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 20, eta_min= 0.0001)

# verbose
print("Total Train Data Size : {}".format(len(train_dataset)))
print("Total Val Data Size : {}".format(len(val_dataset)))
print("Train batch Size : {}".format(train_batch_size))
print("Val Batch Size : {}".format(val_batch_size))
print(f"Total Epochs : {num_epochs}")
print(f"Using device : {device}")
print(f"Learning rate : {learning_rate}")


# train function 
def train(epoch, model,  dataloader, criterion):
    model = model.train().to(device)
    running_loss = AverageValueMeter()
    running_accuracy = AverageValueMeter()

    with tqdm(dataloader, file=sys.stdout, disable=False) as iterator:
        for i, (inputs, labels) in enumerate(iterator):
            inputs, labels = inputs.to(device), labels.to(device)

            output_size = labels.shape[1]
            batch_size = labels.shape[0]

            optimizer.zero_grad()

            outputs, _ = model(inputs, output_size, device, ground_truth = labels)

            
            total_loss = 0
            for index, output in enumerate(outputs):
                loss = criterion(output, labels[:,index,:])
                loss.backward(retain_graph = True)
                total_loss += loss
            
            total_correct = 0
            total_iter = 0
            for index, output in enumerate(outputs):
                _, indices = output.topk(1)
                english_pos = indices.tolist()
                _, indicel = labels[:,index,:].topk(1)
                english_posl = indicel.tolist()
                for x, y in zip(english_pos, english_posl):
                    if y[0] == 2 or y[0] == 0:
                        continue
                    total_iter = total_iter + 1
                    if x[0] == y[0]:
                        total_correct += 1
            
            current_acc = (total_correct * 100) / (total_iter)
                
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            running_accuracy.add(total_correct * 100, total_iter)
            running_loss.add(total_loss.item() * output_size , output_size)
            
            post_fix_s = "lr = {:.4f}, loss = {:.4f}, acc = {:.2f}, cacc = {:.2f}".format(lr, running_loss.mean, running_accuracy.mean, current_acc)
            iterator.set_postfix_str(post_fix_s)
            pre_fix_s = "train = {}".format(epoch)
            iterator.set_description_str(pre_fix_s)

    return running_loss.mean, running_accuracy.mean



# test function 
def test(epoch, model, dataloader, criterion):
    model = model.eval().to(device)
    running_loss = AverageValueMeter()
    running_accuracy = AverageValueMeter()

    with tqdm(dataloader, file=sys.stdout, disable=False) as iterator:
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(iterator):

                inputs, labels = inputs.to(device), labels.to(device)

                output_size = labels.shape[1]
                batch_size = labels.shape[0]

                outputs,_ = model(inputs, output_size, device, ground_truth =  None)

                total_loss = 0
                for index, output in enumerate(outputs):
                    loss = criterion(output, labels[:,index,:])
                    total_loss += loss

                total_correct = 0
                total_iter = 0
                for index, output in enumerate(outputs):
                    _, indices = output.topk(1)
                    english_pos = indices.tolist()
                    _, indicel = labels[:,index,:].topk(1)
                    english_posl = indicel.tolist()
                    for x, y in zip(english_pos, english_posl):
                        if y[0] == 2 or y[0] == 0:
                            continue
                        total_iter = total_iter + 1
                        if x[0] == y[0]:
                            total_correct += 1

                running_accuracy.add(total_correct * 100, total_iter)
                running_loss.add(total_loss.item() * output_size, output_size)
                post_fix_s = "loss = {:.4f} , acc = {:.2f}".format(running_loss.mean, running_accuracy.mean)
                iterator.set_postfix_str(post_fix_s)
                pre_fix_s = "test = {}".format(epoch)
                iterator.set_description_str(pre_fix_s)

    return running_loss.mean, running_accuracy.mean



best_acc = 30

# train and val loop 
for i in range(0, num_epochs):
    print("Epoch : {}".format(i + 1))

    # Perform training & testing
    train_loss, train_acc = train(i + 1, model, train_dataloader, loss)
    val_loss, val_acc = test(i + 1, model, val_dataloader, loss)

    scheduler.step()

    # Save model if a better val acc score is obtained
    if best_acc < val_acc:
        best_acc = val_acc
        model = model.cpu()
        torch.save(model, "best_model.pth")
        model = model.to(device)
        print('Model saved!')




