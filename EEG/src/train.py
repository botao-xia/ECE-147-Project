from tqdm import tqdm
import torch
import torch.nn as nn
import os


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.2):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True


def train(model, train_dataloader, val_dataloader, device=None, num_epochs=100,
            optimizer=torch.optim.Adam, criterion=nn.CrossEntropyLoss(), 
            use_earlystopping=True, **kwargs):
    
    early_stopping = None
    if use_earlystopping:
        early_stopping = EarlyStopping()
    
    optimizer = optimizer(model.parameters(), **kwargs)

    loss_hist, acc_hist, val_loss_hist, val_acc_hist = [], [], [], []

    if device is not None:
        model = model.to(device)

    pbar = tqdm(
        range(num_epochs), position=0, leave=True,
        bar_format='{l_bar}{bar:30}{r_bar}',
    )

    def eval(dataloader, val=False):
        ns = 0
        nc = 0
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                if device is not None:
                    x = x.to(device)
                    y = y.to(device)
                out = model(x)
                loss = None
                if val:
                    loss = criterion(out, y)
                    loss = loss.item()
                ns += len(y)
                nc += (out.max(1)[1] == y).detach().cpu().numpy().sum()
        return nc/ns, loss

    for i, _ in enumerate(pbar):
        model.train() # set model to training mode.
        loss = None
        for j, batch in enumerate(train_dataloader):
            optimizer.zero_grad() # clear gradients of parameters that optimizer is optimizing
            x, y = batch
            if device is not None:
                x = x.to(device) # necessary if X is not on the same device as model
                y = y.to(device)
            
            model.zero_grad()

            out = model(x) # shape (batch_size, n_classes)
            loss = criterion(out, y) # calculate the cross entropy loss

            loss.backward() # backpropagate
            optimizer.step() # perform optimization step
        loss_hist.append(loss.item())
        
        model.eval()
        acc_hist.append(eval(train_dataloader)[0])
        acc, loss = eval(val_dataloader, True)
        val_acc_hist.append(acc)
        val_loss_hist.append(loss)
        torch.save(model.state_dict(), f"./model/model{loss:.3f}")

        if early_stopping.early_stop:
            print(f"early stopping at {i}")
            print(f"loss: train {loss_hist[i]}, valid {val_loss_hist[i]}")
            print(f"acc: train {acc_hist[i]}, valid {val_acc_hist[i]}")
            break

        pbar.set_postfix({'acc': acc_hist[-1], 'val_acc': val_acc_hist[-1]})
    
    return loss_hist, acc_hist, val_loss_hist, val_acc_hist
