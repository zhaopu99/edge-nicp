# Copyright 2024 by Yaopu Zhao, Beihang University, School of Automation Science and Electrical Engineering.
# All rights reserved.
# This file is part of the edge-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import io3d
import json
import numpy as np
from bfm import MorphableModel
from utils import batch_vertex_sample
from edge_nicp import tranformAtoB, non_rigid_icp_edge


device = torch.device('cuda:0')
fine_config = json.load(open('config/fine_grain.json'))
path_config = json.load(open('config/path.json'))

model = MorphableModel(device=device)

meshes = io3d.load_obj_as_mesh(path_config['after_scan_name'], device = device)
target_lm_np = np.loadtxt(path_config['after_landmark_name']).astype(np.float32)[17:]
target_lm = torch.unsqueeze(torch.from_numpy(target_lm_np), 0).to(device)


with torch.no_grad():
    bfm_path = path_config["bfm_name"]
    bfm_meshes = io3d.load_obj_as_mesh(bfm_path, device = device)
    bfm_lm_index = torch.unsqueeze(model.kpt_inds[17:], 0).to(device) 
    bfm_lm = torch.tensor(batch_vertex_sample(bfm_lm_index, bfm_meshes.verts_padded()), dtype=torch.float32).to(device)
    numpy_bfm_lm = bfm_lm.detach().cpu().numpy()[0]

    after_target, after_target_lm = tranformAtoB(bfm_meshes, meshes, bfm_lm, target_lm, device)

registered_mesh = non_rigid_icp_edge(bfm_meshes, after_target, bfm_lm_index, after_target_lm,fine_config, device, with_edge=True)
io3d.save_meshes_as_objs([path_config['path_out_mesh']], registered_mesh, save_textures = False)