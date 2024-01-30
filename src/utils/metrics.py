import torch

def velocity_from_coords(coords:torch.tensor):
    """
    Function to compute velocity from coordinates

    Arguments:
        coords: torch.tensor
            coordinates of shape (n_samples,2,n_timesteps)
    Returns:
        vel: torch.tensor
            velocity of shape (2,n_timesteps-1)

    """
    us,vs = [],[]
    for i in range(coords.shape[-1]-1):
        u = (coords[:,0,i] - coords[:,0,i+1]) / (1 * 60 * 60)
        v = (coords[:,1,i] - coords[:,1,i+1]) / (1 * 60 * 60)
        us.append(u)
        vs.append(v)
    vs = torch.tensor(vs)
    us = torch.tensor(us)
    return torch.stack([us,vs])

def velocity_norm(velocity: torch.tensor):
    """
    Function to compute velocity norm

    Arguments:
        velocity: torch.tensor
            velocity of shape (2,n_timesteps,samples)
    Returns:
        vel: torch.tensor
            norm of shape (n_timesteps,samples)

    """
    return torch.sqrt(velocity[0,:,:]**2 + velocity[1,:,:]**2)

def velocity_angle(pred_vel,target_vel: torch.tensor):
    """
    Function to compute velocity angle between prediction and target tensors

    Arguments:
        pred_vel: torch.tensor
            velocity of shape (2,n_timesteps,samples)
         pred_vel: torch.tensor
            velocity of shape (2,n_timesteps,samples)
    Returns:
        angle: torch.tensor
            norm of shape (n_timesteps,samples)

    """
    unit_pred_vel = pred_vel / velocity_norm(pred_vel)
    unit_target_vel = target_vel / velocity_norm(target_vel)
    # Perform element-wise multiplication followed by sum along the second dimension
    dot_products = torch.clip(torch.sum(unit_pred_vel * unit_target_vel, dim=0),-1.0,1.0)
    angles = torch.acos(dot_products) * 180 / np.pi #to degrees
    return angles