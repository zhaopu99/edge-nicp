# Copyright 2024 by Yaopu Zhao, Beihang University, School of Automation Science and Electrical Engineering.
# All rights reserved.
# This file is part of the edge-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import numpy as np
from pytorch3d.structures import Meshes, Pointclouds
from utils import batch_vertex_sample
from pytorch3d.ops import (
    knn_points,
    knn_gather
)
from pytorch3d.loss import mesh_laplacian_smoothing
from local_affine import LocalAffine
from tqdm import tqdm
from utils import convert_mesh_to_pcl, mesh_boundary, corresponding_alignment


#template, meshes, template_lms, target_lm
def tranformAtoB(
        template: Meshes,
        meshes: Pointclouds,
        template_lms: torch.LongTensor,
        target_lm: torch.LongTensor,
        device=torch.device('cuda:0'),
):
    '''
        deform template mesh to target pointclouds

        The template mesh and target pcl should be normalized with utils.normalize_mesh api.
        The mesh should look at +z axis, the x define the width of mesh, and the y define the height of mesh
    '''

    source_vertex = template.verts_padded()
    target_vertex = meshes.verts_padded()

    # rigid align
    #template_lm = batch_vertex_sample(template_lms, source_vertex)
    template_lm = template_lms
    if (template_lm.shape!=target_lm.shape):
        print('the shape of template_lm and target_lm are different!')
    R, T, s = corresponding_alignment(target_lm, template_lm, device = device, estimate_scale=True)
    transformed_vertex = s[:, None, None] * torch.bmm(target_vertex, R) + T[:, None, :]
    after_target_lm = s[:, None, None] * torch.bmm(target_lm, R) + T[:, None, :]
    after_target = meshes.update_padded(transformed_vertex)
    return after_target, after_target_lm
  
def non_rigid_icp_edge(
        template_mesh: Meshes,
        target_mesh: Meshes,
        template_lm_index: torch.LongTensor,
        target_lm: torch.LongTensor,
        config: dict,
        device,
        with_edge = True
):
    target_pcl = convert_mesh_to_pcl(target_mesh)  
    return non_rigid_icp_edge_mesh2pcl(template_mesh, target_pcl, template_lm_index, target_lm, config, device, with_edge = with_edge)

def non_rigid_icp_edge_mesh2pcl(
        template_mesh: Meshes,
        target_pcl: torch.LongTensor,
        template_lm_index: torch.LongTensor,
        target_lm: torch.LongTensor,
        config: dict,
        device=torch.device('cuda:1'),
        out_affine=False,
        in_affine=None,
        with_edge = True
):

    internal_nostrils_np = np.loadtxt('BFM/nose_internal_withear.txt')
    indexmask = torch.tensor(internal_nostrils_np).to(device)
    exclude_list = [0 if index in internal_nostrils_np else 1 for index in range(template_mesh.verts_padded().shape[1])]
    exclude_tensor = torch.tensor(exclude_list).to(device)
    template_edges_length = cal_edge(template_mesh)
    template_vertex = template_mesh.verts_padded()
    target_vertex = target_pcl.points_padded()

    boundary_mask = mesh_boundary(template_mesh.faces_padded()[0], template_vertex.shape[1])
    boundary_mask = boundary_mask.unsqueeze(0).unsqueeze(2)
    inner_mask = torch.logical_not(boundary_mask)

    # rigid align
    template_lm = torch.tensor(batch_vertex_sample(template_lm_index, template_vertex), dtype=torch.float32)
    R, T, s = corresponding_alignment(target_lm, template_lm, device = device, estimate_scale=True)
    target_vertex = s[:, None, None] * torch.bmm(target_vertex, R) + T[:, None, :]

    assert target_vertex.shape[0] == 1
    
    template_edges = template_mesh.edges_packed()
    mask = ~(torch.isin(template_edges, indexmask).any(dim=1))
    template_edges = template_edges[mask] # 选择需要保留的边


    if in_affine is None:
        local_affine_model = LocalAffine(template_vertex.shape[1], template_vertex.shape[0], template_edges).to(device)
    else:
        local_affine_model = in_affine
    optimizer = torch.optim.AdamW([{'params': local_affine_model.parameters()}], lr=1e-4, amsgrad=True)

    # train param config
    inner_iter = config['inner_iter']
    outer_iter = config['outer_iter']
    loop = tqdm(range(outer_iter))
    log_iter = config['log_iter']

    milestones = set(config['milestones'])
    stiffness_weights = np.array(config['stiffness_weights'])
    landmark_weights = np.array(config['landmark_weights'])
    laplacian_weight = config['laplacian_weight']
    w_idx = 0

    loss = 100
    for i in loop:
        new_deformed_verts, stiffness = local_affine_model(template_vertex, pool_num=0, return_stiff=True)
        new_deformed_lm = batch_vertex_sample(template_lm_index, new_deformed_verts)
        old_verts = new_deformed_verts
        new_deform_mesh = template_mesh.update_padded(new_deformed_verts)

        # we can randomly sample the target point cloud for speed up
        target_sample_verts = target_vertex

        knn = knn_points(new_deformed_verts, target_sample_verts)
        close_points = knn_gather(target_sample_verts, knn.idx)[:, :, 0]

        if (i == 0) and (in_affine is None):
            inner_loop = range(100)
        else:
            inner_loop = range(inner_iter)

        for _ in inner_loop:
            optimizer.zero_grad()
            new_edge_length = cal_edge(new_deform_mesh)
            len = template_edges_length - new_edge_length
            loss_edge_distance = torch.mean(torch.sum(len ** 2))

            vert_distance = (new_deformed_verts - close_points) ** 2
            vert_distance = vert_distance * exclude_tensor.unsqueeze(0).unsqueeze(2)
            vert_distance_mask = torch.sum(vert_distance, dim=2) < 0.04 ** 2
            weight_mask = torch.logical_and(inner_mask, vert_distance_mask.unsqueeze(2))

            vert_distance = weight_mask * vert_distance
            landmark_distance = (new_deformed_lm - target_lm) ** 2

            bsize = vert_distance.shape[0]
            vert_distance = vert_distance.view(bsize, -1)
            vert_sum = torch.sum(vert_distance) / bsize
            landmark_distance = landmark_distance.view(bsize, -1)
            landmark_sum = torch.sum(landmark_distance) * landmark_weights[w_idx] / bsize
            stiffness = stiffness.view(bsize, -1)
            stiffness_sum = torch.sum(stiffness) * stiffness_weights[w_idx] / bsize
            laplacian_loss = mesh_laplacian_smoothing(new_deform_mesh) * laplacian_weight
            
            if with_edge:
                loss = torch.sqrt(vert_sum + landmark_sum + stiffness_sum + loss_edge_distance) + laplacian_loss 
            else:
                loss = torch.sqrt(vert_sum + landmark_sum + stiffness_sum) + laplacian_loss 
            
            loss.backward()
            optimizer.step()
            new_deformed_verts, stiffness = local_affine_model(template_vertex, pool_num=0, return_stiff=True)
            new_deformed_lm = batch_vertex_sample(template_lm_index, new_deformed_verts)
            new_deform_mesh = template_mesh.update_padded(new_deformed_verts)
            
        distance = torch.mean(torch.sqrt(torch.sum((old_verts - new_deformed_verts) ** 2, dim=2)))
        if i % log_iter == 0:
            print(distance.item(), stiffness_sum.item(), landmark_sum.item(), vert_sum.item(), laplacian_loss.item())
        if i in milestones:
            w_idx += 1
    
    if out_affine:
        return new_deform_mesh, local_affine_model
    else:
        return new_deform_mesh

def cal_edge(mesh: Meshes):
    # 假设mesh是一个Meshes对象
    # verts是顶点坐标，faces是面数据
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()

    # 计算每个面的三条边的长度
    edge1 = torch.norm(verts[faces[:, 0]] - verts[faces[:, 1]], dim=1)
    edge2 = torch.norm(verts[faces[:, 1]] - verts[faces[:, 2]], dim=1)
    edge3 = torch.norm(verts[faces[:, 2]] - verts[faces[:, 0]], dim=1)
    edge = torch.cat((edge1, edge2, edge3), dim=0)
    return edge

def cal_edge_mask(mesh: Meshes, mask: torch.LongTensor):
    # 假设mesh是一个Meshes对象
    # verts是顶点坐标，faces是面数据
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()

    mask = mask.unsqueeze(1)
    # 输出原始张量的大小
    #print("原始faces大小：", faces.size())
    #print("原始mask大小：", mask.size())
    
    # 构造一个boolean mask，该mask检查faces张量的每一行是否包含mask中的任何一个元素
    bool_mask = torch.any(torch.isin(faces, mask), dim=1)

    # 使用这个boolean mask来选择those行，这些行不包含mask中的任何元素
    filtered_faces = faces[~bool_mask]

    # 输出过滤后张量的大小
    #print("过滤后faces大小：", filtered_faces.size())


    # 计算每个面的三条边的长度
    edge1 = torch.norm(verts[filtered_faces[:, 0]] - verts[filtered_faces[:, 1]], dim=1)
    edge2 = torch.norm(verts[filtered_faces[:, 1]] - verts[filtered_faces[:, 2]], dim=1)
    edge3 = torch.norm(verts[filtered_faces[:, 2]] - verts[filtered_faces[:, 0]], dim=1)
    edge = torch.cat((edge1, edge2, edge3), dim=0)
    return edge