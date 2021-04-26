import numpy as np
import torch
from utils import vtk_reader,vtk_write_image,numpy_to_vtk,vtk_write
from scipy.spatial.ckdtree import cKDTree,cKDTreeNode

def idw(res,data):
    coord = data[:,:3]
    x = np.linspace(-5,5,res)
    y = np.linspace(-5,5,res)
    z = np.linspace(0,10,res)
    xx,yy,zz = np.meshgrid(x,y,z)
    xyz = np.stack([xx.flatten(),yy.flatten(),zz.flatten()],axis=-1)
    kd = cKDTree(coord,leafsize=100)
    dist,nn = kd.query(xyz,32,n_jobs=-1)
    neighbor = data[nn]
    weight = 1/dist**2
    grid = np.average(neighbor[:,:,3],axis=-1,weights=weight)
    return grid
    # vtk_write_image(res,res,res,grid,"test.vti")

def upsample(res,grid,data):
    coord = data[:,:3]
    grid = grid.reshape(res,res,res)
    B,_ = coord.shape
    concentration = np.zeros((B))
    interval = 10/(res-1)

    for i,xyz in enumerate(coord):
        print(i,"/",B)
        x, y, z = xyz
        xi = int((x+5) // interval)
        yi = int((y+5) // interval)
        zi = int(z // interval)
        xi = xi - 1 if xi == res-1 else xi
        yi = yi - 1 if yi == res-1 else yi
        zi = zi - 1 if zi == res-1 else zi
        xd = (x - xi * interval + 5)/interval
        yd = (y - yi * interval + 5)/interval
        zd = (z - zi * interval)/interval
        c000 = grid[xi,yi,zi]
        c100 = grid[xi+1,yi,zi]
        c110 = grid[xi+1,yi+1,zi]
        c101 = grid[xi+1,yi,zi+1]
        c111 = grid[xi+1,yi+1,zi+1]
        c010 = grid[xi,yi+1,zi]
        c011 = grid[xi,yi+1,zi+1]
        c001 = grid[xi,yi,zi+1]
        
        c00 = c000 * (1-xd) + c100 * xd
        c01 = c001 * (1-xd) + c101 * xd
        c10 = c010 * (1-xd) + c110 * xd
        c11 = c011 * (1-xd) + c111 * xd

        c0 = c00 * (1-yd) + c10 * yd
        c1 = c01 * (1-yd) + c11 * yd

        c =  c0 * (1-zd) + c1 * zd
        concentration[i] = c

    array_dict = {
            "concentration": concentration,
        }
    vtk_data = numpy_to_vtk(coord,array_dict)
    vtk_write(vtk_data,"upsample.vtu")


if __name__ == "__main__":
    filename = "F:/0.20/run01/025.vtu"
    res = 96
    data = vtk_reader(filename)
    # print(data.shape)
    # grid = idw(res,data)
    # upsample(res,grid,data)
    
    from vtkmodules import all as vtk
    from vtkmodules.util import numpy_support
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName('upsample.vtu')
    reader.Update()
    vtk_data = reader.GetOutput()
    coord = numpy_support.vtk_to_numpy(vtk_data.GetPoints().GetData())
    concen = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(0))

    # print(concen.shape)
    sum = 0
    for i,c in enumerate(data):
        cc = concen[i]
        c = c[3]
        sum += (cc-c) ** 2
    mse = sum / len(data)
    print(mse)