import numpy as np 
import matplotlib.pyplot as plt
import os
from ROVir import rovir
from ROVir import top_nv
from ROVir import form_virtual_coil_data

#loading npy MRI data, shape (240, 240, 48, 252) = (x, y, channels, z) note: not always the case for shape
data = np.load(r"C:\Users\selin\OneDrive\Desktop\SRI\SRI Code\numpy_data.npy") #loading MRI data, already centered

x_slice = 223 # why are both ends visible, not central slices?
y_slice = 230 
z_slice = 232 #180 to 250

#defining manual masks for ROVir algorithm
# want to define a matrix of ones (for ROI) and zeros (for interference) 
# mask A is the region of interest 
# mask B is the region of interferenc
x_cover = int(0.5*data.shape[0]) #does it have to be an int?
y_cover = int(0.5*data.shape[1])
z_cover = int(0.5*data.shape[3])
# x_show = data.shape[0] - x_cover
# y_show = data.shape[1] - y_cover
# z_show = data.shape[3] - z_cover

maskA = np.ones((data.shape[0], data.shape[1], data.shape[3]))
maskA[x_cover:, y_cover:, z_cover:] = 0
maskB = np.zeros((data.shape[0], data.shape[1], data.shape[3])) # a 3D matrix of zeros
maskB[x_cover:, y_cover:, z_cover:] = 1

# now want the eigenvectors
V = rovir(data, maskA, maskB)
# print(V)
# print(V.shape)

# now choose the top nv eigencvectors
# calculate signal interference ratio for each channel, keep only those with sir > 1
#sir_threshold = input("Enter SIR thresshold, default is 1")
i = top_nv(V, data, maskA, maskB, sir_threshold = 1)
print(i)
V_retain = V[:,:i] # keep only the top nv eigenvectors 

# now, the eigenvectors make up the linear combo weights
# so we need to form the virtual coil data
virtual_coil_data = form_virtual_coil_data(V_retain, data)

# fxn to compute the root sum of squares to combine channels
def rsos(data):
    return np.sqrt(np.sum(np.abs(data)**2, axis=2))

image = np.fft.ifftn(virtual_coil_data, axes=(0, 1, 3)) # compute the Fourier transform of data over x, y, z axes
image_rss = rsos(image) 

data_slice_axial = data[:, :, : , z_slice] # get an axial slice of k-space data
kspace_rss_axial = rsos(data_slice_axial) # compute rsos of k-space slice 

image_axial = image_rss[:, :, z_slice] # get an axial slice from the reconstructed image
image_shifted_axial = np.fft.ifftshift(image_axial, axes=(0,1)) # shift image to center 0 frequency

data_slice_cor = data[:, y_slice, :, :] # get a coronal slice from the reconstructed image
kspace_rss_cor = rsos(data_slice_cor) # compute rsos of k-space

image_cor = image_rss[:, y_slice, :] # get coronal slice of image
image_shifted_cor = np.fft.ifftshift(image_cor, axes=(0,1)) # shift image to center

data_slice_sag = data[x_slice, :, :, :]  
kspace_rss_sag = np.sqrt(np.sum(np.abs(data_slice_sag)**2, axis=2)) 

image_sag = image_rss[x_slice, :, :]
image_shifted_sag = np.fft.ifftshift(image_sag, axes=(0,1))

# visualizing reconstructed image
fig = plt.figure(figsize=(10, 10))

plt.subplot(3,2,1)
plt.title(f'Axial K-space z = {z_slice}')
plt.imshow(np.log(np.abs(kspace_rss_axial)+1e-9), cmap='gray')

plt.subplot(3,2,2)
plt.title(f'Axial View z = {z_slice}')
plt.imshow(np.rot90(image_shifted_axial,3), cmap='gray')

plt.subplot(3,2,3)
plt.title(f'Coronal K-space y = {y_slice}')
plt.imshow(np.log(np.abs(kspace_rss_cor)+1e-9), cmap='gray')

plt.subplot(3,2,4)
plt.title(f'Coronal View y = {y_slice}')
plt.imshow(np.rot90(image_shifted_cor,3), cmap='gray')

plt.subplot(3,2,5)
plt.title(f'Sagittal K-space x = {x_slice}')
plt.imshow(np.log(np.abs(kspace_rss_sag)+1e-9), cmap='gray')

plt.subplot(3,2,6)
plt.title(f'Sagittal View x = {x_slice}')
plt.imshow(np.rot90(image_shifted_sag,3), cmap='gray')

name = f'Reconstructed Image Axial {z_slice}, Coronal {y_slice}, Sagittal {x_slice}.png'
path = os.path.join(r"C:\Users\selin\OneDrive\Desktop\SRI\SRI Code", name)
plt.savefig(path)
plt.show()


#assume mrd format for input data 
