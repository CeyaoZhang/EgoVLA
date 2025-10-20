import trimesh


def load_obj_mesh(obj_model_path, unit=1.0):
    mesh = trimesh.load_mesh(obj_model_path)
    vertices = mesh.vertices * unit  # unit: m
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    return new_mesh
