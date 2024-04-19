import torch
from typing import Optional,List,Tuple



def forward_pass(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    criterion: torch.nn.Module = None,
    gradient_clip:int =1
):

    e_g,v_g = batch[0].to(device),batch[1].to(device)
    optimizer.zero_grad() if optimizer else None

    output = model(e_g,v_g)
    loss = criterion(output, e_g.y[0],e_g.y[1]) if criterion else None
    if optimizer:
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clip)
        optimizer.step()
    return loss.item() if loss else None

def process_dataloader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    criterion: torch.nn.Module = None,
    gradient_clip:int =1

):
    total_loss = 0.0
    model.train() if optimizer else model.eval()
    for batch in dataloader:
        loss = forward_pass(model, batch, device, optimizer, criterion,gradient_clip=gradient_clip)
        total_loss += loss if loss else 0
    return total_loss / len(dataloader) if total_loss else None


