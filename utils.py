import random
import os
import sys
import math
import time

from vtkmodules import all as vtk
from vtkmodules.util import numpy_support
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.ckdtree import cKDTree,cKDTreeNode
from multiprocessing import Pool
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset,DataLoader
from yt.utilities.sdf import SDFRead
from thingking import loadtxt

def grid_forward(model,lat_vec,xyz):
    """
    Network forward when assuming grid latent vectors
    """
    B,_ = xyz.shape
    lat_dim, res, _, _ = lat_vec.shape
    interval = 1/(res-1)
    index = (xyz // interval).long()
    index[index==res-1] -= 1
    idd = (xyz - index * interval)/interval
    xd = idd[:,0]
    yd = idd[:,1]
    zd = idd[:,2]
    xi,yi,zi = index.T
    rep_lat = lat_vec.view(1,*lat_vec.shape).repeat(B,1,1,1,1)
    bi = torch.arange(B)

    code = rep_lat[bi,:,xi,yi,zi]
    disp = torch.stack([xd,yd,zd],dim=1)
    c000 = model(torch.cat([code,disp],dim=1))

    code = rep_lat[bi,:,xi+1,yi,zi]
    disp = torch.stack([xd-1,yd,zd],dim=1)
    c100 = model(torch.cat([code,disp],dim=1))

    code = rep_lat[bi,:,xi+1,yi+1,zi]
    disp = torch.stack([xd-1,yd-1,zd],dim=1)
    c110 = model(torch.cat([code,disp],dim=1))

    code = rep_lat[bi,:,xi+1,yi,zi+1]
    disp = torch.stack([xd-1,yd,zd-1],dim=1)
    c101 = model(torch.cat([code,disp],dim=1))

    code = rep_lat[bi,:,xi+1,yi+1,zi+1]
    disp = torch.stack([xd-1,yd-1,zd-1],dim=1)
    c111 = model(torch.cat([code,disp],dim=1))

    code = rep_lat[bi,:,xi,yi+1,zi]
    disp = torch.stack([xd,yd-1,zd],dim=1)
    c010 = model(torch.cat([code,disp],dim=1))

    code = rep_lat[bi,:,xi,yi+1,zi+1]
    disp = torch.stack([xd,yd-1,zd-1],dim=1)
    c011 = model(torch.cat([code,disp],dim=1))

    code = rep_lat[bi,:,xi,yi,zi+1]
    disp = torch.stack([xd,yd,zd-1],dim=1)
    c001 = model(torch.cat([code,disp],dim=1))

    output = trilinear_interpolation(c000,c100,c110,c101,c111,c010,c011,c001,xd,yd,zd)
    return output

def trilinear_interpolation(c000,c100,c110,c101,c111,c010,c011,c001,xd,yd,zd):
    """
    Return the trilinear interpolated given the eight corner values
    idd: normalized distance to c000
    """
    B, _ = c000.shape

    xd = xd.view(B,1)
    c00 = c000 * (1-xd) + c100 * xd
    c01 = c001 * (1-xd) + c101 * xd
    c10 = c010 * (1-xd) + c110 * xd
    c11 = c011 * (1-xd) + c111 * xd

    yd = yd.view(B,1)
    c0 = c00 * (1-yd) + c10 * yd
    c1 = c01 * (1-yd) + c11 * yd

    zd = zd.view(B,1)
    c =  c0 * (1-zd) + c1 * zd
    return c

class PointDataFPM(Dataset):
    # dataset in blocks
    def __init__(self,file_name):
        data = vtk_reader(file_name)
        # normalize data to [0, 1]
        data_max = np.array([ 5, 5, 10, 357.19000244, 38.62746811, 48.47133255, 50.60621262])
        data_min = np.array([-5, -5, 0, 0, -5.63886223e+01, -3.69567909e+01, -7.22953186e+01])
        data = (data - data_min) / (data_max-data_min)
        data = np.clip(data,0,1)
        num_bins = 100
        hist, bin_edges = np.histogram(data[:,3],num_bins,density=True)
        bin_index = np.digitize(data[:,3],bin_edges) -1
        bin_index[bin_index==num_bins] = num_bins-1
        self.weights = 1 / hist[bin_index]
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
                    

def halo_reader(filename):
    try:
        ID, DescID, Mvir, Vmax, Vrms, Rvir, Rs, Np, x, y, z, VX, VY, VZ, JX, JY, JZ, Spin, rs_klypin, Mvir_all, M200b, M200c, M500c, M2500c, Xoff, Voff, spin_bullock, b_to_a, c_to_a, A_x_, A_y_, A_z_, b_to_a_500c_, c_to_a_500c_, A_x__500c_, A_y__500c_, A_z__500c_, TU, M_pe_Behroozi, M_pe_Diemer = \
            loadtxt(filename, unpack=True)
        r = Rvir/1000
        try:
            halo_num = len(x)
            return np.stack([x,y,z],axis=1),r
        except TypeError:
            return np.array([[x,y,z]]),np.array([r])
    except ValueError:
        return [],[]

def IoU(predict,target):
    assert len(predict) == len(target)
    predict = np.array(predict)
    target = np.array(target)
    union = np.logical_or(predict,target)
    inter = np.logical_and(predict,target)
    return np.sum(inter)/np.sum(union)

def scatter_3d(array,vmin=None,vmax=None,threshold = -1e10,center=None,save=False,fname=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    array = array[array[:,3] > threshold]
    ax.scatter(array[:,0],array[:,1],array[:,2],c=array[:,3],marker='.',vmin=vmin,vmax=vmax)
    if center is not None:
        ax.scatter(center[0],center[1],center[2],c="red",marker='o')
    # ax2 = fig.add_subplot(122,projection='3d',sharex=ax,sharey=ax,sharez=ax)
    # ax2.scatter(array2[:,0],array2[:,1],array2[:,2],c=array2[:,3],marker='^',vmin=-1,vmax=1)
    if save:
        plt.savefig(fname)
    else:
        plt.show()

def halo_writer(center,Rvir,outputname):
    haloData = vtk.vtkAppendPolyData()
    for i in range(len(center)):
        print(i,"/",len(center),end='\r')
        s = vtk.vtkSphereSource()
        s.SetCenter(*center[i])
        s.SetRadius(Rvir[i])
        s.Update()
        input1 = vtk.vtkPolyData()
        input1.ShallowCopy(s.GetOutput())
        haloData.AddInputData(input1)
    haloData.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputConnection(haloData.GetOutputPort())
    writer.SetFileName(outputname)
    writer.Write()

def vtk_reader(filename):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()
    coord = numpy_support.vtk_to_numpy(vtk_data.GetPoints().GetData())
    concen = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(0))[:,None]
    velocity = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(1))
    point_data = np.concatenate((coord,concen,velocity),axis=-1)
    return point_data

def sdf_reader(filename):
    particles = SDFRead(filename)
    h_100 = particles.parameters['h_100']
    width = particles.parameters['L0']
    cosmo_a = particles.parameters['a']
    kpc_to_Mpc = 1./1000
    convert_to_cMpc = lambda proper: (proper ) * h_100 * kpc_to_Mpc / cosmo_a + 31.25
    numpy_data = np.array(list(particles.values())[2:-1]).T
    numpy_data[:,:3] = convert_to_cMpc(numpy_data[:,:3])
    return numpy_data

def all_leaf_nodes(node):
    if node.greater==None and node.lesser==None:
        return [node]
    else:
        return(all_leaf_nodes(node.lesser)+all_leaf_nodes(node.greater))

def collect_file(directory,mode="fpm",shuffle=False):
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for filename in filenames:
            if mode == "fpm":
                if filename.endswith(".vtu") and filename != "000.vtu":
                    inp = os.sep.join([dirpath, filename])
                    file_list.append(inp)
            if mode == "cos":
                if "ds14" in filename.split('_'):
                    inp = os.sep.join([dirpath, filename])
                    file_list.append(inp)
    if shuffle:
        random.shuffle(file_list)
    return file_list
    
def min_max(file_list,mode="fpm"):
    all_min = None
    all_max = None
    for i,f in enumerate(file_list):
        print("processing file {}/{}".format(i,len(file_list)),end='\r')
        data = vtk_reader(f) if mode == "fpm" else sdf_reader(f)
        f_min = np.min(data,axis=0)
        f_max = np.max(data,axis=0)
        if all_min is None:
            all_min = f_min
            all_max = f_max
            total = np.zeros((data.shape[1]))
            length = 0
        else:
            all_min = np.where(all_min < f_min,all_min,f_min)
            all_max = np.where(all_max > f_max,all_max,f_max)
        total += np.sum(data,axis = 0)
        length += len(data)
    mean = total/length
    print("mean: ", mean)
    print("min: ", all_min)
    print("max: ", all_max)
    return mean,all_min,all_max

def std(file_list,mean,mode="fpm"):
    for i,f in enumerate(file_list):
        print("processing file {}/{}".format(i,len(file_list)),end='\r')
        data = vtk_reader(f) if mode == "fpm" else sdf_reader(f)
        if i==0:
            total = np.zeros((data.shape[1]))
            length = 0
        data = (data - mean) ** 2
        total += np.sum(data,axis = 0)
        length += len(data)
    std = np.sqrt(total/length)
    print("std: ", std)
    return std

def numpy_to_vtk(position:np.array,array_dict:dict):
    vtk_position = numpy_support.numpy_to_vtk(position)
    points = vtk.vtkPoints()
    points.SetData(vtk_position)
    data_save = vtk.vtkUnstructuredGrid()
    data_save.SetPoints(points)
    pd = data_save.GetPointData()
    for k, v in array_dict.items():
        vtk_array = numpy_support.numpy_to_vtk(v)
        vtk_array.SetName(k)
        pd.AddArray(vtk_array)
    return data_save

def vtk_write(data_save,filename:str):
    writer = vtk.vtkXMLDataSetWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data_save)
    writer.Write()

def vtk_write_image(x_dim,y_dim,z_dim,data,filename):
    # data flattened in xyz order
    assert len(data) == x_dim * y_dim * z_dim
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(x_dim,y_dim,z_dim)
    imageData.AllocateScalars(vtk.VTK_DOUBLE, 1)

    dims = imageData.GetDimensions()

    # Fill every entry of the image data with "2.0"
    i = 0
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                imageData.SetScalarComponentFromDouble(z, y, x, 0, data[i])
                i+=1


    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imageData)
    writer.Write()

def plot_loss(filename):
    epoch = 1
    loss_sum = 0.0
    loss_list = []
    count = 0
    with open(filename,"r") as f:
        for line in f:
            if "====>" in line:
                line_list = line.split(' ')
                loss = float(line_list[5])
                if epoch == int(line_list[2]):
                    loss_sum += loss
                    count += 1
                else:
                    epoch = int(line_list[2])
                    loss_list.append(loss_sum/count)
                    loss_sum = loss
                    count = 1
    print(loss_list)
