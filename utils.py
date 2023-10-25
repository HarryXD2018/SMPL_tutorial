import numpy as np
import torch
import torch.nn.functional as F
import pickle


def load_pkl(file_name, to_torch=True):
    with open(file_name, 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    if to_torch:
        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                if key in ['joint2num', 'part2num']:
                    continue
                elif key in ['ft', 'f']:
                    value = value.astype(np.int64)
                obj[key] = torch.tensor(value)
    return obj


def load_obj(file_name, to_torch=True):
    assert file_name.endswith('.obj'), f'file_name must end with .obj, but got {file_name}'
    vertices = []
    faces = []
    with open(file_name, 'rb') as f:
        for line in f.readline():
            if line.startswith('v '):
                vertices.append([float(v) for v in line.split()[1:]])
            elif line.startswith('f '):
                faces.append([int(v) for v in line.split()[1:]])
    if to_torch:
        vertices = torch.tensor(vertices)
        faces = torch.tensor(faces)
    else:
        vertices = np.array(vertices)
        faces = np.array(faces)
    return vertices, faces


def write_obj(faces, vertices, file_name):
    with open(file_name, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')