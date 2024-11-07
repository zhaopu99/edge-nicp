# Copyright 2024 by Yaopu Zhao, Beihang University, School of Automation Science and Electrical Engineering.
# All rights reserved.
# This file is part of the edge-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import json
import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.ops import knn_points, sample_points_from_meshes
from utils import corresponding_alignment
from bfm import MorphableModel


device = torch.device('cuda:0')

path_json = 'config/path.json'
path_config = json.load(open(path_json))


kpt = np.load(path_config['before_landmark_path'] )
kpt_scan = torch.tensor(kpt,dtype=torch.float32).to(device)[17:,:]
kpt_scan_all = torch.tensor(kpt,dtype=torch.float32).to(device)

face_scan_mesh = load_objs_as_meshes([path_config['before_scan_path']])
points = face_scan_mesh.verts_list()[0].to(device)

# Initialize the model
model = MorphableModel(device=device)

initial_shape_coeffs, initial_expression_coeffs = model.init_coeff_tensors('zero')
bfm_shape = model.forward(initial_shape_coeffs.to(device), initial_expression_coeffs.to(device)).view(-1,3)
kpt_bfm = model.get_landmarks(bfm_shape)
kpt_bfm = kpt_bfm[17:,:]


R, T, s = corresponding_alignment(kpt_scan.unsqueeze(0), kpt_bfm.unsqueeze(0), estimate_scale = True, device = device)
transformed_vertex = s[:, None, None] * torch.bmm(points.unsqueeze(0), R) + T[:, None, :]
kpt_transformer_vertex = s[:, None, None] * torch.bmm(kpt_scan.unsqueeze(0), R) + T[:, None, :]
kpt_scan_all = s[:, None, None] * torch.bmm(kpt_scan_all.unsqueeze(0), R) + T[:, None, :]


np.savetxt(path_config['after_landmark_name'] ,kpt_scan_all[0].detach().cpu().numpy())
save_obj(path_config['after_scan_name'] ,torch.tensor(transformed_vertex[0]).to(torch.device('cpu')), face_scan_mesh.faces_packed())


after_mesh = face_scan_mesh.to(device).update_padded(transformed_vertex)
# Sample points from the mesh surface
sampled_points = sample_points_from_meshes(after_mesh, num_samples=points.shape[0]*5).to(device)


# Define stopping criteria
convergence_threshold = 1e-7
max_iterations = 10000
prev_loss = 1000


# Convert to leaf tensors
shape_coeffs = torch.nn.Parameter(initial_shape_coeffs)
expression_coeffs = torch.nn.Parameter(initial_expression_coeffs)

optimizer_shape = torch.optim.Adam([shape_coeffs], lr=5e-1)
optimizer_exp = torch.optim.Adam([expression_coeffs], lr=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_shape, mode='min', patience=10, factor=0.5, verbose=True)

# Optimization loop
for iteration in range(max_iterations):
    optimizer_shape.zero_grad()
    optimizer_exp.zero_grad()

    # Forward pass: Compute the predicted shape from the model
    bfm_shape = model.forward(shape_coeffs.to(device), expression_coeffs.to(device)).reshape(-1,3)
    kpt = bfm_shape[model.kpt_inds + 1,:].to(device)[17:, :]    
    distances, _, _ = knn_points(bfm_shape.unsqueeze(0), sampled_points, K=1)

    # Compute the nearest neighbors on the sampled points
    nn_output = knn_points(bfm_shape.unsqueeze(0), sampled_points, K=1, return_nn=True)
    # Compute the distances of the points to the nearest neighbors
    distances = nn_output.dists[..., 0].sqrt()


    # Compute the loss as the Chamfer distance between the scan and the model
    distances_kpt = torch.norm(kpt.to(device) -torch.tensor(kpt_transformer_vertex,dtype=torch.float32).squeeze(0).to(device), dim=1)
    loss0 = torch.mean(distances)*0.5
    losslm = torch.mean(distances_kpt)
    loss1 = torch.mean(torch.square(shape_coeffs)) #.sum()
    loss2 = torch.mean(torch.square(expression_coeffs)) * 100#.sum()
    loss = loss1 + loss2 + losslm + loss0 
    
    # Backward pass and optimization
    loss.backward()
    optimizer_shape.step()
    optimizer_exp.step()

    scheduler.step(loss)
    current_lr = optimizer_shape.param_groups[0]['lr']

    # Check for convergence
    with torch.no_grad():
        loss_change = abs(prev_loss - loss)
        if loss_change < convergence_threshold:
            print(f"Converged at iteration {iteration}")
            bfm_shape = model(shape_coeffs.to(device), expression_coeffs.to(device)).reshape(-1, 3)
            bfm_meshes = Meshes(bfm_shape.unsqueeze(0)*0.1, model.faces.unsqueeze(0))
            
            bfm_name = f'result/bfm_mesh.obj'
            save_obj(bfm_name, bfm_meshes.verts_packed(), bfm_meshes.faces_packed())
            
            path_config['bfm_name'] = bfm_name
            with open(path_json, 'w') as json_file:
                json.dump(path_config, json_file, indent=4)
            break

        prev_loss = loss

    print(f"Iteration {iteration}: Loss {loss.item()}, loss_lks {losslm}, shape_cof {loss1}, exp_cof {loss2} , dis_loss {loss0},Learning Rate: {current_lr}") #