import torch
import torch.nn as nn
import numpy as np



def compute_metrics(
        targets:torch.tensor,
        predictions:torch.tensor,
        init_coords:torch.tensor,
        velocity:bool,
        iterations:int=1,
        step:int=1,
        d_time:int=3600
):
    """ 
    Compute metrics for the model
    Arguments:
        targets: torch.tensor
            target values
        predictions: torch.tensor
            predicted values
        init_coords: torch.tensor
            initial coordinates
        velocity: bool
            if the model is predicting velocity or coordinates
        iterations: int
            number of iterations to predict
        step: int
            step ahead of prediction
        d_time: int
            time step in seconds
    Returns:
        rmse_position: torch.tensor, 
            rmse over position
        mae_vel_norm: torch.tensor
            mae over velocity norm
        angles: torch.tensor
            angle between predicted and target velocity
    """
    mse =  nn.MSELoss(reduction='none')
    mae = nn.L1Loss(reduction='none')
    if velocity:
        #convert velocity to coordinates
        target_vel = targets.reshape(2,-1).unsqueeze(1)
        predicted_vel = predictions.reshape(2,-1).unsqueeze(1)
        targets = targets*step*d_time + init_coords
        predictions = predictions*step*d_time + init_coords
    
            
    #get original coordinates in meters
    target_coords = torch.stack([targets[:,:iterations],targets[:,iterations:]],dim=1)
    predicted_coords = torch.stack([predictions[:,:iterations],predictions[:,iterations:]],dim=1)

    #error over position
    rmse_position = torch.sqrt(torch.mean(mse(target_coords,predicted_coords),dim=[1]))
    #error over velocity
    if not velocity:
            target_coords = torch.stack([init_coords,target_coords.squeeze()],dim=-1)
            predicted_coords = torch.stack([init_coords,predicted_coords.squeeze()],dim=-1)
            target_vel = velocity_from_coords(np.array(target_coords))
            predicted_vel = velocity_from_coords(np.array(predicted_coords))
    #speed
    target_vel_norm = velocity_norm(target_vel)
    predicted_vel_norm =  velocity_norm(predicted_vel)
    mae_vel_norm = mae(target_vel_norm,predicted_vel_norm)
    #angle
    angles = velocity_angle(predicted_vel,target_vel)

    return rmse_position,mae_vel_norm,angles,target_vel,predicted_vel
            



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
    return torch.stack([us,vs],dim=0)

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
    return torch.sqrt(velocity[0]**2 + velocity[1]**2)

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



class CustomIceLoss(nn.Module):
    def __init__(self, A=1,B=0,C=0,step=1,d_time=3600):
        super(CustomIceLoss, self).__init__()
        self.mae = nn.L1Loss()
        self.A = A
        self.B = B
        self.C = C
        self.step = step
        self.d_time = d_time

    def velocity_angle(self,real_v,v_pred):
        v_pred = v_pred / v_pred.norm(p=2,dim=1).unsqueeze(1)
        real_v = real_v / real_v.norm(p=2,dim=1).unsqueeze(1)
        dot = (v_pred*real_v).sum(dim=1)
        dot = torch.clamp(dot,min=-0.99,max=0.99)
        return torch.acos(dot).mean()

    def forward(self,v_pred,v_real,init_coords):

        position_pred = v_pred * self.step * self.d_time + init_coords
        position_real = v_real * self.step * self.d_time + init_coords

        position_error = 0
        angle_error = 0
        velocity_error = 0

        if self.A >0:
            velocity_error = self.mae(v_real,v_pred)

        if self.B >0:
            position_error = self.mae(position_real,position_pred)

        if self.C >0:
            angle_error = self.velocity_angle(v_pred,v_real)
            if torch.isnan(angle_error):
                print("nan error")
                    
        error = self.A * velocity_error + self.B * position_error + self.C * angle_error

        return  error