import torch
import torch.nn as nn
import numpy as np

from scipy.spatial import Delaunay

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






class StepForwardLoss(nn.Module):
    def __init__(self, loss=nn.MSELoss()):
        super(StepForwardLoss, self).__init__()
        self.loss_fn = loss
   

    def forward(self,v_pred,v_real,input,mask):

        v_pred = v_pred[mask]
        v_real = v_real[mask]
        input = input[mask]

        error_target = v_real - v_pred
        error_input =  input - v_pred
        print((error_target**2).mean())
        print((torch.abs(error_target)/ (torch.abs(error_input )+ 1e-18)).mean())
        print()

        error_weighted = error_target**2 * (torch.abs(error_target)/ (torch.abs(error_input )+ 1e-18))


        error = torch.mean(error_weighted)

        return  error


class Constrained_loss(nn.Module):
    def __init__(self, loss=nn.MSELoss(), alpha=10):
        super(Constrained_loss, self).__init__()
        self.loss_fn = loss
        self.alpha = alpha   

    def forward(self,v_pred,v_real):
        
        
        local_loss = self.loss_fn(v_pred,v_real)
        v_pred_norm = torch.sqrt(v_pred[:,0]**2 + v_pred[:,1]**2)
        v_real_norm = torch.sqrt(v_real[:,0]**2 + v_real[:,1]**2)
        global_loss = torch.mean(v_pred_norm - v_real_norm)**2

        error = local_loss + self.alpha * global_loss

        return  error



def jacobian(x0, y0, x1, y1, x2, y2):
    return (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)




class Deformation_loss(nn.Module):
    def __init__(self, loss=nn.MSELoss(), alpha=1e-15):
        super(Deformation_loss, self).__init__()
        self.loss_fn = loss
        self.alpha = alpha
        self.d_time = 3600

    def forward(self,v_pred,v_real,init_coords):
        position_pred = v_pred * self.d_time + init_coords
        position_real = v_real *  self.d_time + init_coords
      
        ### Deformation ###
        x0,y0 = init_coords[:,0],init_coords[:,1]
        x1,y1 = position_real[:,0],position_real[:,1]
        x_pred,y_pred = position_pred[:,0],position_pred[:,1]

        # create subsampled mesh for matching nodes only
        pos = torch.stack([x0,y0],axis=1).cpu().numpy()

        t0 = torch.tensor(Delaunay(pos,qhull_options='QJ').simplices).to(v_pred.device)
        #t0 = Triangulation(x0, y0).triangles # other possible function
        # find starting / ending coordinates for each elements
       
        x1a, x1b, x1c = x1[t0].T
        y1a, y1b, y1c = y1[t0].T
        xpa, xpb, xpc = x_pred[t0].T
        ypa, ypb, ypc = y_pred[t0].T

        # compute area at the first and second snapshots (subsampled mesh)
        a1 = jacobian(x1a, y1a, x1b, y1b, x1c, y1c)
        ap = jacobian(xpa, ypa, xpb, ypb, xpc, ypc)

        #divergence
        div_true = a1#(a0/a1)
        div_pred = ap#(a0/ap)

        #compute loss
        div_loss = self.alpha *self.loss_fn(div_true,div_pred)
        vel_loss = self.loss_fn(v_real,v_pred)


        error =  div_loss + vel_loss
       
        
        return  error.to(v_pred.device)




class Vector_loss(nn.Module):
    def __init__(self, args= {'A':1, 'B':0,'C':0,'d_time':3600},loss=nn.MSELoss()):
        super(Vector_loss, self).__init__()
        self.loss_fn = loss
        print(args)
        self.A,self.B,self.C,self.d_time = args['A'],args['B'],args['C'],args['d_time']

    def velocity_angle(self,real_v,v_pred):
        v_pred = v_pred / v_pred.norm(p=2,dim=1).unsqueeze(1)
        real_v = real_v / real_v.norm(p=2,dim=1).unsqueeze(1)
   
        dot = torch.abs((v_pred*real_v).sum(dim=1)-1)
        
        return torch.tensor(dot.mean())

    def forward(self,v_pred,v_real,init_coords):

        position_pred = v_pred * self.d_time + init_coords
        position_real = v_real *  self.d_time + init_coords

        position_error = 0
        angle_error = 0
        velocity_error = 0

        if self.A >0:
            velocity_error = self.loss_fn(v_real,v_pred)
            #print("velocity error",velocity_error.item())


        if self.B >0:
            position_error = self.loss_fn(position_real,position_pred)
            #print("position error",position_error.item())


        if self.C >0:
            angle_error = self.velocity_angle(v_pred,v_real)
            #print("angle error",angle_error.item())

            if torch.isnan(angle_error):
                print("nan error")

        #print()
                    
        error = self.A * velocity_error + self.B * position_error + self.C * angle_error

        return  error