# cs6910_assignment3

### Steps to Run
1. Clone the Repo
```
git clone https://github.com/iitaakash/cs6910_assignment3.git
```
2. To train the model run. Edit the line in 
```
# define the model here 
model = AttentionModel()
# model = NormalModel()
```
`train.py` to switch between model with attention and the default model(as shown above) and then run
``` 
python train.py
```
3. For testing
``` 
python test.py 
```
Edit the model path in `test.py` as shown below
```
model = torch.load("best_model.pth")
```

Change the hyperparameters in the `cfg.py` file.
