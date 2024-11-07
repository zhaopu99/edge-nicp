# Copyright 2024 by Yaopu Zhao, Beihang University, School of Automation Science and Electrical Engineering.
# All rights reserved.
# This file is part of the edge-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points, sample_points_from_meshes

from utils import corresponding_alignment
from bfm import MorphableModel

def fitting_3dmm(kpt_scan, face_scan_mesh, model = MorphableModel(device = torch.device('cuda:0'))):
    device = model.device
    points = face_scan_mesh.verts_list()[0].to(device)
    
    # 初始化bfm模型，得到模型和landmark
    initial_shape_coeffs, initial_expression_coeffs = model.init_coeff_tensors('zero')
    bfm_shape = model.forward(initial_shape_coeffs.to(device), initial_expression_coeffs.to(device)).view(-1,3)
    kpt_bfm = model.get_landmarks(bfm_shape)
    kpt_bfm = kpt_bfm[17:, :] if len(kpt_bfm) == 68 else kpt_bfm

    # 计算扫描数据和bfm的旋转矩阵, 分别将扫描数据与landmark旋转平移到bfm位置
    R, T, s = corresponding_alignment(kpt_scan.unsqueeze(0), kpt_bfm.unsqueeze(0), device = device, estimate_scale = True)
    transformed_vertex = s[:, None, None] * torch.bmm(points.unsqueeze(0), R) + T[:, None, :]
    kpt_scan_transformer_vertex = s[:, None, None] * torch.bmm(kpt_scan.unsqueeze(0), R) + T[:, None, :]

    new_mesh = face_scan_mesh.to(device).update_padded(transformed_vertex)
    # Sample points from the mesh surface
    sampled_points = sample_points_from_meshes(new_mesh, num_samples=points.shape[0]*5).to(device) 

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
        distances_kpt = torch.norm(kpt.to(device) -torch.tensor(kpt_scan_transformer_vertex,dtype=torch.float32).squeeze(0).to(device), dim=1)
        loss0 = torch.mean(distances)*0.5 
        losslm = torch.mean(distances_kpt)
        loss1 = torch.mean(torch.square(shape_coeffs))
        loss2 = torch.mean(torch.square(expression_coeffs)) * 100
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
                bfm_shape = model(shape_coeffs.to(device), expression_coeffs.to(device)).reshape(-1, 3)
                bfm_meshes = Meshes(bfm_shape.unsqueeze(0)*0.1, model.faces.unsqueeze(0))
                return shape_coeffs, expression_coeffs, bfm_meshes
            prev_loss = loss

        #print(f"Iteration {iteration}: Loss {loss.item()}, loss_lks {losslm}, shape_cof {loss1}, exp_cof {loss2} , dis_loss {loss0},Learning Rate: {current_lr}") #


'''
def fitting_3dmm_express(kpt_scan, face_scan_mesh, shape_coeffs, model = MorphableModel(device = torch.device('cuda:0'))):
    device = model.device
    points = face_scan_mesh.verts_list()[0].to(device)

    # 初始化bfm模型，得到模型和landmark
    initial_shape_coeffs, initial_expression_coeffs = model.init_coeff_tensors('zero')
    bfm_shape = model.forward(torch.tensor(shape_coeffs).to(device), initial_expression_coeffs.to(device)).view(-1,3)
    kpt_bfm = model.get_landmarks(bfm_shape)
    kpt_bfm = kpt_bfm[17:,:]

    # 计算扫描数据和bfm的旋转矩阵, 分别将扫描数据与landmark旋转平移到bfm位置
    R, T, s = corresponding_alignment(kpt_scan.unsqueeze(0), kpt_bfm.unsqueeze(0), estimate_scale = True)
    transformed_vertex = s[:, None, None] * torch.bmm(points.unsqueeze(0), R) + T[:, None, :]
    kpt_scan_transformer_vertex = s[:, None, None] * torch.bmm(kpt_scan.unsqueeze(0), R) + T[:, None, :]
    new_mesh = face_scan_mesh.to(device).update_padded(transformed_vertex)
    sampled_points = sample_points_from_meshes(new_mesh, num_samples=points.shape[0]*5).to(device) # Sample points from the mesh surface
    #np.savetxt(f"scan_express.txt",kpt_scan_transformer_vertex[0].detach().cpu().numpy())
    #save_obj(f'scan_express.obj',torch.tensor(transformed_vertex[0]).to(torch.device('cpu')), face_scan_mesh.faces_packed())

    # Define stopping criteria
    convergence_threshold = 1e-7
    max_iterations = 10000
    prev_loss = 1000


    # Convert to leaf tensors
    expression_coeffs = torch.nn.Parameter(initial_expression_coeffs)

    optimizer_exp = torch.optim.Adam([expression_coeffs], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_exp, mode='min', patience=10, factor=0.5, verbose=True)

    # Optimization loop
    for iteration in range(max_iterations):
        optimizer_exp.zero_grad()

        # Forward pass: Compute the predicted shape from the model
        bfm_shape = model.forward(shape_coeffs.to(device), expression_coeffs.to(device)).reshape(-1,3)
        kpt = bfm_shape[model.kpt_inds + 1,:].to(device)[17:, :]
    
        distances, indices, ss = knn_points(bfm_shape.unsqueeze(0), sampled_points, K=1)

        # Compute the nearest neighbors on the sampled points
        nn_output = knn_points(bfm_shape.unsqueeze(0), sampled_points, K=1, return_nn=True)
        # Compute the distances of the points to the nearest neighbors
        distances = nn_output.dists[..., 0].sqrt()


        # Compute the loss as the Chamfer distance between the scan and the model
        distances_kpt = torch.norm(kpt.to(device) -torch.tensor(kpt_scan_transformer_vertex,dtype=torch.float32).squeeze(0).to(device), dim=1)
        #bfm_normals = compute_vertex_normals(bfm_shape.to(device), model.faces.to(device)).unsqueeze(0).to(device)
        loss0 = torch.mean(distances) #chamfer_distance(bfm_shape.unsqueeze(0).to(device), torch.tensor(transformed_vertex, dtype=torch.float32).to(device))[0] *9 # , loss_normal , x_normals=bfm_normals, y_normals=scan_normal)
        losslm = torch.mean(distances_kpt)
        loss2 = torch.mean(torch.square(expression_coeffs))*2 #.sum()
        loss = loss2 + losslm + loss0 
        # Backward pass and optimization
        loss.backward()
        optimizer_exp.step()

        scheduler.step(loss)
        current_lr = optimizer_exp.param_groups[0]['lr']

        # Check for convergence
        with torch.no_grad():
            loss_change = abs(prev_loss - loss)
            if loss_change < convergence_threshold:
                print(f"Converged at iteration {iteration}")
                bfm_shape = model(shape_coeffs.to(device), expression_coeffs.to(device)).reshape(-1, 3)
                mesh = Meshes(bfm_shape.unsqueeze(0)*0.1, model.faces.unsqueeze(0))
            
                save_obj(f'obj_express_{iteration}.obj', mesh.verts_packed(), mesh.faces_packed())
                return expression_coeffs, mesh

            prev_loss = loss


        print(f"Iteration {iteration}: Loss {loss.item()}, loss_lks {losslm}, exp_cof {loss2} , dis_loss {loss0},Learning Rate: {current_lr}") #
'''

def fitting_3dmm_with_express(kpt_scan, face_scan_mesh, model = MorphableModel(device = torch.device('cuda:0'))):
    device = model.device
    points = face_scan_mesh.verts_list()[0].to(device)
    # 初始化bfm模型，得到模型和landmark
    initial_shape_coeffs, initial_expression_coeffs = model.init_coeff_tensors('zero')
    bfm_shape = model.forward(initial_shape_coeffs.to(device), initial_expression_coeffs.to(device)).view(-1,3)
    kpt_bfm = model.get_landmarks(bfm_shape)
    kpt_bfm = kpt_bfm[17:,:]

    # 计算扫描数据和bfm的旋转矩阵, 分别将扫描数据与landmark旋转平移到bfm位置
    R, T, s = corresponding_alignment(kpt_scan.unsqueeze(0), kpt_bfm.unsqueeze(0), device = device, estimate_scale = True)
    transformed_vertex = s[:, None, None] * torch.bmm(points.unsqueeze(0), R) + T[:, None, :]
    kpt_scan_transformer_vertex = s[:, None, None] * torch.bmm(kpt_scan.unsqueeze(0), R) + T[:, None, :]
    new_mesh = face_scan_mesh.to(device).update_padded(transformed_vertex)
    sampled_points = sample_points_from_meshes(new_mesh, num_samples=points.shape[0]*5).to(device) # Sample points from the mesh surface
    

    # Define stopping criteria
    convergence_threshold = 1e-5
    max_iterations = 10000
    prev_loss = 1000


    # Convert to leaf tensors
    shape_coeffs = torch.nn.Parameter(initial_shape_coeffs)
    expression_coeffs = torch.nn.Parameter(initial_expression_coeffs)

    optimizer_shape = torch.optim.Adam([shape_coeffs], lr=5e-1)
    optimizer_exp = torch.optim.Adam([expression_coeffs], lr=5e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_shape, mode='min', patience=10, factor=0.5, verbose=True)

    # Optimization loop
    for iteration in range(max_iterations):
        optimizer_shape.zero_grad()
        optimizer_exp.zero_grad()

        # Forward pass: Compute the predicted shape from the model
        bfm_shape = model.forward(shape_coeffs.to(device), expression_coeffs.to(device)).reshape(-1,3)
        kpt = bfm_shape[model.kpt_inds + 1,:].to(device)[17:, :]
    
        distances, indices, _ = knn_points(bfm_shape.unsqueeze(0), sampled_points, K=1)

        # Compute the nearest neighbors on the sampled points
        nn_output = knn_points(bfm_shape.unsqueeze(0), sampled_points, K=1, return_nn=True)
        # Compute the distances of the points to the nearest neighbors
        distances = nn_output.dists[..., 0].sqrt()


        # Compute the loss as the Chamfer distance between the scan and the model
        distances_kpt = torch.norm(kpt.to(device) -torch.tensor(kpt_scan_transformer_vertex,dtype=torch.float32).squeeze(0).to(device), dim=1)
        #bfm_normals = compute_vertex_normals(bfm_shape.to(device), model.faces.to(device)).unsqueeze(0).to(device)
        loss0 = torch.mean(distances)*0.5 #chamfer_distance(bfm_shape.unsqueeze(0).to(device), torch.tensor(transformed_vertex, dtype=torch.float32).to(device))[0] *9 # , loss_normal , x_normals=bfm_normals, y_normals=scan_normal)
        losslm = torch.mean(distances_kpt)
        loss1 = torch.mean(torch.square(shape_coeffs)) #.sum()
        loss2 = torch.mean(torch.square(expression_coeffs))*0.2#.sum()
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
                #print(f"Converged at iteration {iteration}")
                bfm_shape = model(shape_coeffs.to(device), expression_coeffs.to(device)).reshape(-1, 3)
                mesh = Meshes(bfm_shape.unsqueeze(0)*0.1, model.faces.unsqueeze(0))
                return shape_coeffs, expression_coeffs, mesh
            prev_loss = loss

        print(f"Iteration {iteration}: Loss {loss.item()}, loss_lks {losslm}, shape_cof {loss1}, exp_cof {loss2} , dis_loss {loss0},Learning Rate: {current_lr}") #
