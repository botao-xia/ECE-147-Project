from tqdm import tqdm
import torch
import torch.nn as nn # pytorch's neural networks module

def train_loop(model, train_dataloader, val_dataloader, device=None, optimizer=torch.optim.Adam, criterion=nn.NLLLoss()):
    optimizer = optimizer(model.parameters())
    num_epochs = 30
    loss_hist, acc_hist, val_loss_hist, val_acc_hist = [], [], [], []
    if device is not None:
        model = model.to(device)

    pbar = tqdm(
        range(num_epochs), position=0, leave=True,
        bar_format='{l_bar}{bar:30}{r_bar}',
    )
    for epoch in pbar:
        model.train() # set model to training mode.
        for batch in train_dataloader:
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

            # IMPORTANT: DO NOT store 'loss' by itself, since it references its entire computational graph.
            # Otherwise you will run out of memory.
            # You MUST use .item() to convert to a scalar or call .detach().
            loss_hist.append(loss.item())
        
        model.eval() # set model to evaluation mode. Relevant for dropout, batchnorm, layernorm, etc.
        # calculate accuracy for training and validation sets
        ns = 0 # number of samples
        nc = 0 # number of correct outputs
        with torch.no_grad():
            for batch in train_dataloader:
                x, y = batch
                x = x.to(device) # necessary if X is not on the same device as model
                y = y.to(device)
                out = model(x)
                ns += len(y)
                nc += (out.max(1)[1] == y).detach().cpu().numpy().sum()
        acc_hist.append(nc/ns)

        ns = 0 # number of samples
        nc = 0 # number of correct outputs
        with torch.no_grad():
            for batch in val_dataloader:
                x, y = batch
                x = x.to(device) # necessary if X is not on the same device as model
                y = y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss_hist.append(loss.item())
                ns += len(y)
                nc += (out.max(1)[1] == y).detach().cpu().numpy().sum()
        val_acc_hist.append(nc/ns)


        # update progress bar postfix
        pbar.set_postfix({'acc': acc_hist[-1], 'val_acc': val_acc_hist[-1]})
    
    return loss_hist, acc_hist, val_loss_hist, val_acc_hist