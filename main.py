import numpy as np 
import matplotlib.pyplot as plt
import os
from ROVir import rovir
from ROVir import top_nv
from ROVir import form_virtual_coil_data
import time

#loading npy MRI data, shape (240, 240, 48, 252) = (x, y, channels, z) note: not always the case for shape
data = np.load(r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\reduced_data.npy") #loading MRI data, already centered

#downsize data by  2x in every aces 

x_slice = 60 # why are both ends visible, not central slices? need to shift before rss and slicing
y_slice = 60 
z_slice = 60 

# defining manual masks for ROVir algorithm
# mask A is the region of interest, mask B is the region of interference

maskA = np.zeros((data.shape[0], data.shape[1], data.shape[3]))
maskB = np.ones((data.shape[0], data.shape[1], data.shape[3])) # a 3D matrix of zeros
margin = 50
maskA[margin:-margin, margin:-margin, margin:-margin] = 1
maskB[margin:-margin, margin:-margin, margin:-margin] = 0

start = time.time()
V = rovir(data, maskA, maskB) #finding the eigenvectors 
end = time.time()
print(f"Time taken for ROVir: {end - start} seconds")

start = time.time()
# now choose the top nv eigencvectors
# calculate signal interference ratio for each channel, keep only those with sir > 1
# sir_threshold = input("Enter SIR thresshold, default is 1")
i = top_nv(V, data, maskA, maskB, sir_threshold = 1) #finding 
print(i)
V_retain = V[:,:i] # keep only the top nv eigenvectors 
end = time.time()
print(f"Time taken for top_nv: {end - start} seconds")

start = time.time()

# now, the eigenvectors make up the linear combo weights
# so we need to form the virtual coil data
virtual_coil_data = form_virtual_coil_data(V_retain, data)
end = time.time()
print(f"Time taken for form_virtual_coil_data: {end - start} seconds")

# fxn to compute the root sum of squares to combine channels
def rsos(data):
    return np.sqrt(np.sum(np.abs(data)**2, axis=2))

image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(virtual_coil_data, axes = (0, 1, 3)), axes=(0, 1, 3)), axes = (0, 1, 3)) # compute the Fourier transform of data over x, y, z axes
image_rss = rsos(image) 

data_slice_axial = data[:, :, : , z_slice] # get an axial slice of k-space data
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
plt.imshow(maskA[:, :, z_slice], alpha = 0.5, cmap='Greens')
plt.imshow(maskB[:, :, z_slice], alpha = 0.5, cmap='Reds')

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

name = f'Reconstructed Image Axial {z_slice}, Coronal {y_slice}, Sagittal {x_slice}.png'
path = os.path.join(r"C:\Users\selin\OneDrive\Desktop\SRI\SRI Code", name)
plt.savefig(path)
plt.show() 


#assume mrd format for input data 
