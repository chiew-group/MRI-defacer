import json
import matplotlib.pyplot as plt
import numpy as np 
import os

from file_explorer import select_data
from ROVir import rovir
from ROVir import top_nv_sir
from ROVir import form_virtual_coil_data


def rsos(data):
    '''
    Computes the root sum of squares of the data across the channel dimension.
    
    Parameters
    ----------
        data -> np.ndarray: a 4D array of MRI data, shape (x, y, ch, z)

    Returns
    ----------
        np.ndarray: a 3D array of the root sum of squares of the data, shape (x, y, z)
    '''
    
    return np.sqrt(np.sum(np.abs(data)**2, axis=3))


with open('config.json', 'r') as config_file:
    inputs = json.load(config_file)

data = np.load(inputs["data_path"]) 
print("Data has been successfully loaded")
data = np.moveaxis(data, -1, 2) # move axis to shape (x, y, z, ch)
# data = np.transpose(data, (2, 1, 0, 3))
# data = np.flip(data, axis = 1)

# defining image slices
x_slice = int(inputs["x_slice"])
y_slice = int(inputs["y_slice"])
z_slice = int(inputs["z_slice"])

maskB = np.load(r"TotalSegmentator\mask.npy") # define face mask
maskA = np.ones((data.shape[0], data.shape[1], data.shape[2])) # define signal mask
maskA = maskA - maskB

# fourier transform and shift data 
image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))

V = rovir(image, maskA, maskB) # finding the eigenvectors for Av = λBv

all_coils = form_virtual_coil_data(V, data)

image_all = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(all_coils, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))
figure = plt.figure(figsize = (20, 20))
for coil in range(all_coils.shape[3]):
    plt.subplot(8, 6, coil+1)
    plt.imshow(np.abs(image_all[x_slice, :, :, coil]), cmap='gray')
    plt.imshow(maskB[x_slice, :, :], alpha = 0.4, cmap = 'Reds')
    plt.title(f'Virtual Coil {coil+1}')
plt.show()

sir_threshold = float(inputs["sir_threshold"])
i = top_nv_sir(V, image, maskA, maskB, sir_threshold) # finding top nv eigenvectors based on SIR
print(f'The top nv eigenvectors contain the first {i} eigenvectors')
V_retain = V[:,:i] # retain only the top nv eigenvectors 

virtual_coil_data = form_virtual_coil_data(V_retain, data) # forming virtual coils, with eigenvectors as linear combo weights

# shift, compute fourier transform, shift over x, y, z axes
image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(virtual_coil_data, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))
image_rss = rsos(image) # find root sum of squares of image over the channel dimension
kspace_rss = rsos(virtual_coil_data) # compute rsos of k-space data

# getting the axial, coronal, and sagittal kspace slices
kspace_axial = kspace_rss[:, :, z_slice] 
kspace_cor = kspace_rss[:, y_slice, :] 
kspace_sag = kspace_rss[x_slice, :, :]  

# getting the axial, coronal, and sagittal image slices
image_axial = image_rss[:, :, z_slice] 
image_cor = image_rss[:, y_slice, :]
image_sag = image_rss[x_slice, :, :]

fig = plt.figure(figsize=(10, 10))

plt.subplot(3,2,1)
plt.title(f'Axial K-space z = {z_slice}')
plt.imshow(np.log(np.abs(kspace_axial)+1e-9), cmap='gray')

plt.subplot(3,2,2)
plt.title(f'Axial View z = {z_slice}')
plt.imshow(image_axial, cmap='gray', vmin = 0, vmax = 0.3)
# plt.imshow(maskB[:, :, z_slice], alpha = 0.4, cmap = 'Reds')

plt.subplot(3,2,3)
plt.title(f'Coronal K-space y = {y_slice}')
plt.imshow(np.log(np.abs(kspace_cor)+1e-9), cmap='gray')

plt.subplot(3,2,4)
plt.title(f'Coronal View y = {y_slice}')
plt.imshow(image_cor, cmap='gray', vmin = 0, vmax = 0.3)
# plt.imshow(maskB[:, y_slice, :], alpha = 0.4, cmap = 'Reds')

plt.subplot(3,2,5) 
plt.title(f'Sagittal K-space x = {x_slice}')
plt.imshow(np.log(np.abs(kspace_sag)+1e-9), cmap='gray')

plt.subplot(3,2,6)
plt.title(f'Sagittal View x = {x_slice}')
plt.imshow(image_sag, cmap='gray', vmin = 0, vmax = 0.3)
# plt.imshow(maskB[x_slice, :, :], alpha = 0.4, cmap = 'Reds')

name = f'Reconstructed Image Axial {z_slice}, Coronal {y_slice}, Sagittal {x_slice}.png'
path = os.path.join(r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer", name)
plt.savefig(path)
plt.show()