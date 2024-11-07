# Copyright 2024 by Yaopu Zhao, Beihang University, School of Automation Science and Electrical Engineering.
# All rights reserved.
# This file is part of the edge-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import json
import io3d
import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from utils import batch_vertex_sample
from bfm import  MorphableModel
from fitting_3dmm import fitting_3dmm, fitting_3dmm_with_express
from edge_nicp import tranformAtoB, non_rigid_icp_edge


device = torch.device('cuda:0')
fine_config = json.load(open('config/fine_grain.json'))
obj_file_path='testdata/00002/mouth_open.obj'
landmark_path = 'testdata/00002/mouth_open_landmarks.txt'
out_mesh_path = f"result/man_out_openmouth_with_template_no_edge.obj"

face_scan_mesh = load_objs_as_meshes([obj_file_path], device = device)
kpt_np = np.loadtxt(landmark_path, dtype=np.float32)
kpt = torch.from_numpy(kpt_np).to(device)
model = MorphableModel(device=device)
shape_coeffs, exp_coeffs, bfm_meshes = fitting_3dmm_with_express(kpt, face_scan_mesh, model)

with torch.no_grad():
    bfm_lm_index = torch.unsqueeze(model.kpt_inds[17:], 0).to(device)
    bfm_lm = torch.tensor(batch_vertex_sample(bfm_lm_index, bfm_meshes.verts_padded()), dtype=torch.float32).to(device)
    target_lm = torch.unsqueeze(kpt, 0).to(device)
    after_target, after_target_lm = tranformAtoB(bfm_meshes, face_scan_mesh, bfm_lm, target_lm, device)


registered_mesh = non_rigid_icp_edge(bfm_meshes, after_target, bfm_lm_index, after_target_lm, fine_config, device)
io3d.save_meshes_as_objs([out_mesh_path], registered_mesh, save_textures = False)