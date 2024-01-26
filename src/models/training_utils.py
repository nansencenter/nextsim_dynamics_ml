import torch
from typing import Optional,List,Tuple



def forward_pass(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    criterion: Optional[torch.nn.Module] = None
) -> Optional[float]:
    batch = batch.to(device)
    optimizer.zero_grad() if optimizer else None
    output = model(batch.x, batch.edge_index, batch.edge_attr)
    loss = criterion(output, batch.y[0]) if criterion else None
    if optimizer:
        loss.backward()
        optimizer.step()
    return loss.item() if loss else None

def process_dataloader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    criterion: Optional[torch.nn.Module] = None
) -> Optional[float]:
    total_loss = 0.0
    model.train() if optimizer else model.eval()
    for batch in dataloader:
        loss = forward_pass(model, batch, device, optimizer, scheduler, criterion)
        total_loss += loss if loss else 0
    return total_loss / len(dataloader) if total_loss else None