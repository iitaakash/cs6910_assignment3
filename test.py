import torch
import torch.nn as nn
import sys
from tqdm import tqdm as tqdm
from Meter import *
from utils import *
from Aksharantar import Aksharantar as MyDataset
from torch.utils.data import DataLoader
import Aksharantar
import matplotlib.pyplot as plt
import numpy as np

# Get dataset
test_dataset = MyDataset(mode = 'test')

batch_size = 100
test_dataloader = DataLoader(test_dataset, batch_size, True, drop_last= False)

# params
device = GetDevice()
model = torch.load("best_model.pth")

# define loss function
loss = nn.CrossEntropyLoss()

print("Total Test Data Size : {}".format(len(test_dataset)))
print(f"Using device : {device}")


def test(epoch, model, dataloader, criterion):
    model = model.eval().to(device)
    running_loss = AverageValueMeter()
    running_accuracy = AverageValueMeter()

    predictions = []
    ground_truth = []
    input_data = []
    accuracy = 0
    samples_9 = None
    samples_9_e = []
    samples_9_t = []
    with tqdm(dataloader, file=sys.stdout, disable=False) as iterator:
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(iterator):

                inputs, labels = inputs.to(device), labels.to(device)

                output_size = labels.shape[1]
                batch_size = labels.shape[0]

                

                outputs, att_maps = model(inputs, output_size, device, ground_truth =  None)

                

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
                
                outputs = torch.stack(outputs, axis = 0).permute(1,0,2)
                att_maps = torch.stack(att_maps, axis = 1).permute(2,0,1,3).squeeze(0)
                samples_9 = att_maps[0:9,:,:]
                for i in range(9):
                    samples_9_e.append(Aksharantar.decode_out(outputs[i,:,:]))
                    samples_9_t.append(Aksharantar.decode_input(inputs[i,:,:]))
                for i in range(batch_size):
                    ground_truth.append(Aksharantar.decode_out(labels[i,:,:]))
                    predictions.append(Aksharantar.decode_out(outputs[i,:,:]))
                    input_data.append(Aksharantar.decode_input(inputs[i,:,:]))

                running_accuracy.add(total_correct * 100, total_iter)
                running_loss.add(total_loss.item() * output_size, output_size)
                post_fix_s = "loss = {:.4f} , acc = {:.2f}".format(running_loss.mean, running_accuracy.mean)
                iterator.set_postfix_str(post_fix_s)
                pre_fix_s = "test = {}".format(epoch)
                iterator.set_description_str(pre_fix_s)

    correct = 0
    for data0, data1, data2 in zip(input_data, ground_truth, predictions):
        # print(f"{data0} -> {data1} -> {data2}")
        if data1 == data2:
            correct = correct + 1
    import pandas as pd
    df = pd.DataFrame({'input_data':input_data, 'ground_truth':ground_truth, 'predictions':predictions})
    # print(df)
    # df.to_csv("predictions_attention.csv", sep='\t')

    # uncomment for heatmap

    # f, axes  = plt.subplots(nrows = 3, ncols = 3, figsize=(15, 15))   
        
    # for i, ax in enumerate(axes.flat):
    #     lene = len(samples_9_e[i])
    #     lent = len(samples_9_t[i])
    #     att_map = samples_9[i,:lene,-lent:].cpu().numpy()
    #     im = ax.imshow(att_map, cmap='hot', interpolation='nearest', )
    #     # ax.set_xlabel(samples_9_t[i], size='large')
    #     ax.set_xticks(np.arange(len(samples_9_t[i])), labels=samples_9_t[i])
    #     # ax.set_ylabel(samples_9_e[i], size='large')
    #     ax.set_yticks(np.arange(len(samples_9_e[i])), labels=samples_9_e[i])
    # plt.show()

    print(f"Word Accuracy : {correct / len(ground_truth)}")
    return running_loss.mean, running_accuracy.mean



test_loss, test_acc = test(1, model, test_dataloader, loss)



