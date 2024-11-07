# Copyright 2024 by Yaopu Zhao, Beihang University, School of Automation Science and Electrical Engineering.
# All rights reserved.
# This file is part of the edge-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio

#load BFM 3DMM model
def load_BFM(model_path):
    #Args:
    #    model_path: path to BFM model.
    #Returns:
    #    model: (nver = 53215, ntri = 105840). nver: number of vertices. ntri: number of triangles.
    #        'shapeMU': [3*nver, 1]
    #        'shapePC': [3*nver, 199]
    #        'shapeEV': [199, 1]
    #        'expMU': [3*nver, 1]
    #        'expPC': [3*nver, 29]
    #        'expEV': [29, 1]
    #        'texMU': [3*nver, 1]
    #        'texPC': [3*nver, 199]
    #        'texEV': [199, 1]
    #        'tri': [ntri, 3] (start from 1, should sub 1 in python and c++)
    #        'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles)
    #        'kpt_inds': [68,] (start from 1)
    #PS:
    #    You can change codes according to your own saved data.
    #    Just make sure the model has corresponding attributes.
    
    C = sio.loadmat(model_path)
    model = C['model']
    model = model[0, 0]

    # change dtype from double(np.float64) to np.float32,
    # since big matrix process(espetially matrix dot) is too slow in python.
    model['shapeMU'] = (model['shapeMU'] + model['expMU']).astype(np.float32)/1e04
    model['shapePC'] = model['shapePC'].astype(np.float32)/1e04
    model['shapeEV'] = model['shapeEV'].astype(np.float32)
    model['expPC'] = model['expPC'].astype(np.float32)/1e04
    model['expEV'] = model['expEV'].astype(np.float32)

    # matlab start with 1. change to 0 in python.
    model['tri'] = model['tri'].T.copy(order='C').astype(np.int32) - 1
    model['tri_mouth'] = model['tri_mouth'].T.copy(order='C').astype(np.int32) - 1

    # kpt ind
    model['kpt_ind'] = (np.squeeze(model['kpt_ind']) - 1).astype(np.int32)

    return model
# Define the 3DMM model as a PyTorch module
class MorphableModel(nn.Module):
    def __init__(self, model_path = 'BFM/BFM.mat', device = torch.device('cuda:0')):
        super().__init__()
        self.device = device
        self.model = load_BFM(model_path)
        self.kpt_inds = torch.tensor(self.model['kpt_ind'], dtype=torch.int64).to(self.device)
        self.n_shape_para = self.model['shapePC'].shape[1]
        main_coff = self.model['shapePC'].shape[1]#50
        self.n_exp_para = self.model['expPC'].shape[1]
        self.n_tex_para = self.model['texPC'].shape[1]
        self.mean_shape = torch.tensor(self.model['shapeMU'], dtype=torch.float32).to(self.device)
        self.shape_pcs = torch.tensor(self.model['shapePC'][:,:main_coff], dtype=torch.float32).to(self.device)
        self.expression_pcs = torch.tensor(self.model['expPC'], dtype=torch.float32).to(self.device)
        self.faces = torch.cat((torch.tensor(self.model['tri'],dtype=torch.int64),torch.tensor(self.model['tri_mouth'],dtype=torch.int64)),dim=0).to(self.device)
        self.faces[:,[1,2]]=self.faces[:,[2,1]].to(self.device)
        self.evshape = torch.tensor(self.model['shapeEV'][:main_coff,:], dtype=torch.float32).to(self.device)


    def init_coeff_tensors(self, type='zero'):
        if type == 'zero':
            self.id_tensor = torch.tensor(np.zeros((self.n_shape_para, 1)), dtype=torch.float32)
            self.exp_tensor = torch.tensor(np.zeros((self.n_exp_para, 1)), dtype=torch.float32)
        if type == 'ones':
            self.id_tensor = torch.tensor(np.ones((self.n_shape_para, 1)), dtype=torch.float32)
            self.exp_tensor = torch.tensor(np.zeros((self.n_exp_para, 1)), dtype=torch.float32)
        elif type == 'random':
            self.id_tensor = torch.tensor(np.random.rand(self.n_shape_para, 1), dtype=torch.float32)
            self.exp_tensor = torch.tensor(np.zeros((self.n_exp_para, 1)), dtype=torch.float32)
        return self.id_tensor, self.exp_tensor

    def get_landmarks(self, vs):
        lms = vs[self.kpt_inds, :].to(self.device)
        return lms


    def forward(self, shape_coeffs, expression_coeffs):
        shape_coeffs = shape_coeffs*self.evshape
        shape_deformation = torch.matmul(self.shape_pcs, shape_coeffs)
        expression_deformation = torch.matmul(self.expression_pcs, expression_coeffs)
        return self.mean_shape + shape_deformation + expression_deformation