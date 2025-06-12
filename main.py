import numpy as np 
import matplotlib.pyplot as plt
import os
from ROVir import rovir
from ROVir import top_nv_sir
from ROVir import form_virtual_coil_data
import time

# loading npy MRI data, shape (120, 120, 48, 126) = (x, y, channels, z) note: not always the case for shape
data = np.load(r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\reduced_data.npy") #loading MRI data, already centered

# defining image slices
x_slice = 60 # input("Enter x slice for sagittal view")
y_slice = 60 # input("Enter y slice for coronal view")
z_slice = 60 # input("Enter z slice for axial view"))

maskA = np.zeros((data.shape[0], data.shape[1], data.shape[3])) #define signal mask
maskB = np.ones((data.shape[0], data.shape[1], data.shape[3])) # define inteference mask

# margin = 50
# maskA[margin:-margin+20, margin:-margin+20, margin:-margin+20] = 1
# maskB[margin:-margin+20, margin:-margin+20, margin:-margin+20] = 0
# maskA[20:70, 20:70, :] = 1
# maskB[25:65, 25:65, :]  = 0
# maskA[:, 30:60, :] = 1
# maskB[:, 30:55, : ] = 0

V = rovir(data, maskA, maskB) #finding the eigenvectors for Av = λBv


i = top_nv_sir(V, data, maskA, maskB, sir_threshold = 3) # finding top nv eigenvectors based on SIR
print(f'The top nv eigenvectors contain the first {i} eigenvectors')
V_retain = V[:,:] # retain only the top nv eigenvectors 

virtual_coil_data = form_virtual_coil_data(V_retain, data) # forming virtual coils, with eigenvectors as linear combo weights

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
    
    return np.sqrt(np.sum(np.abs(data)**2, axis=2))

# shift, compute fourier transform, shift over x, y, z axes
image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(virtual_coil_data, axes = (0, 1, 3)), axes=(0, 1, 3)), axes = (0, 1, 3))
image_rss = rsos(image) # find root sum of squares of image over the channel dimension

figure = plt.figure(figsize = (20, 20))
for coil in range(virtual_coil_data.shape[2]):
    plt.subplot(8, 6, coil+1)
    plt.imshow(np.abs(np.rot90(image[x_slice, :, coil, :],3)), cmap='gray')
    # plt.imshow(maskA[x_slice, :, :], alpha = 0.2, cmap='Greens')
    # plt.imshow(maskB[x_slice, :, :], alpha = 0.2, cmap='Reds')
    plt.title(f'Virtual Coil {coil+1}')
plt.show()

data_slice_axial = virtual_coil_data[:, :, : , z_slice] # get an axial slice of k-space data
kspace_rss_axial = rsos(data_slice_axial) # compute rsos of k-space slice

image_axial = image_rss[:, :, z_slice] # get an axial slice from the reconstructed image
#image_shifted_axial = np.fft.ifftshift(image_axial, axes=(0,1)) # shift image to center 0 frequency

data_slice_cor = data[:, y_slice, :, :] # get a coronal slice from the reconstructed image
kspace_rss_cor = rsos(data_slice_cor) # compute rsos of k-space

image_cor = image_rss[:, y_slice, :] # get coronal slice of image
#image_shifted_cor = np.fft.ifftshift(image_cor, axes=(0,1)) # shift image to center

data_slice_sag = data[x_slice, :, :, :]  
kspace_rss_sag = np.sqrt(np.sum(np.abs(data_slice_sag)**2, axis=2)) 

image_sag = image_rss[x_slice, :, :]
#image_shifted_sag = np.fft.ifftshift(image_sag, axes=(0,1))

# visualizing reconstructed image
fig = plt.figure(figsize=(10, 10))

plt.subplot(3,2,1)
plt.title(f'Axial K-space z = {z_slice}')
plt.imshow(np.log(np.abs(kspace_rss_axial)+1e-9), cmap='gray')

plt.subplot(3,2,2)
plt.title(f'Axial View z = {z_slice}')
plt.imshow(np.rot90(image_axial,3), cmap='gray')

plt.subplot(3,2,3)
plt.title(f'Coronal K-space y = {y_slice}')
plt.imshow(np.log(np.abs(kspace_rss_cor)+1e-9), cmap='gray')

plt.subplot(3,2,4)
plt.title(f'Coronal View y = {y_slice}')
plt.imshow(np.rot90(image_cor,3), cmap='gray')

plt.subplot(3,2,5) 
plt.title(f'Sagittal K-space x = {x_slice}')
plt.imshow(np.log(np.abs(kspace_rss_sag)+1e-9), cmap='gray')

plt.subplot(3,2,6)
plt.title(f'Sagittal View x = {x_slice}')
plt.imshow(np.rot90(image_sag,3), cmap='gray')
# plt.imshow(maskA[x_slice, :, :], alpha = 0.2, cmap='Greens')
# plt.imshow(maskB[x_slice, :, :], alpha = 0.2, cmap='Reds')

name = f'Reconstructed Image Axial {z_slice}, Coronal {y_slice}, Sagittal {x_slice}.png'
path = os.path.join(r"C:\Users\selin\OneDrive\Desktop\SRI\SRI Code", name)
plt.savefig(path)
plt.show() 