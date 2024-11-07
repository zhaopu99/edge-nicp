# Copyright 2024 by Yaopu Zhao, Beihang University, School of Automation Science and Electrical Engineering.
# All rights reserved.
# This file is part of the edge-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

from pytorch3d.io import (
    load_objs_as_meshes,
    save_obj,
)

def load_obj_as_mesh(fp, device = None, load_textures = True):
    '''
        if load mesh with texture, only a single texture image with its mtl is permitted
        return mesh structure
    '''
    mesh = load_objs_as_meshes([fp], device, load_textures)
    return mesh

def save_meshes_as_objs(fp_list, mesh, save_textures = True):
    '''
        input Meshes object
        save obj
    '''
    for idx, fp in enumerate(fp_list):
        verts = mesh.verts_padded()[idx]
        faces = mesh.faces_padded()[idx]
        if save_textures:
            if mesh.textures.isempty():
                raise Exception('Save untextured mesh with param save_textures=True')
            texture_map = mesh.textures.maps_padded()[idx]
            verts_uvs = mesh.textures.verts_uvs_padded()[idx]
            faces_uvs = mesh.textures.faces_uvs_padded()[idx]
            save_obj(fp, verts, faces, verts_uvs = verts_uvs, faces_uvs = faces_uvs, texture_map = texture_map)
        else:
            save_obj(fp, verts, faces)
