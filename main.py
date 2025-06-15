import numpy as np 
import matplotlib.pyplot as plt
import os
from ROVir import rovir
from ROVir import top_nv_sir
from ROVir import form_virtual_coil_data
from ROVir import top_nv_signal_retained
import matplotlib.patches as patches

# loading npy MRI data, shape (120, 120, 48, 126) = (x, y, channels, z) note: not always the case for shape
data = np.load(r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\reduced_data.npy") #loading MRI data, already centered
data = np.moveaxis(data, -1, 2)

# defining image slices
x_slice = 60 # input("Enter x slice for sagittal view")
y_slice = 60 # input("Enter y slice for coronal view")
z_slice = 60 # input("Enter z slice for axial view"))

maskA = np.zeros((data.shape[0], data.shape[1], data.shape[2])) #define signal mask
maskB = np.ones((data.shape[0], data.shape[1], data.shape[2])) # define inteference mask

# mask i used for sagittal view
# maskA[10:80, 20:70, 30:110] = 1
# maskB[10:80, 20:70, 30:110] = 0

x_start,x_end = 10, 80 #80,112
y_start, y_end = 15, 70
z_start, z_end = 30, 110

maskA[x_start:x_end, y_start:y_end, z_start:z_end] = 1
maskB[x_start:x_end, 0:y_end, z_start:z_end]= 0

image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))

V = rovir(image, maskA, maskB) # finding the eigenvectors for Av = λBv

# print(V)

all_coils = form_virtual_coil_data(V, data)

image_all = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(all_coils, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))
figure = plt.figure(figsize = (20, 20))
for coil in range(all_coils.shape[3]):
    plt.subplot(8, 6, coil+1)
    plt.imshow(np.abs(image_all[x_slice, :, :, coil]), cmap='gray')
    rect = patches.Rectangle((z_start, y_start), z_end-z_start, y_end-y_start,
                         linewidth=2, edgecolor='g', facecolor='none')
    plt.gca().add_patch(rect)
    plt.title(f'Virtual Coil {coil+1}')
plt.show()

i = top_nv_sir(V, image, maskA, maskB, 1.3) # finding top nv eigenvectors based on SIR
print(f'The top nv eigenvectors contain the first {i} eigenvectors')
V_retain = V[:,:i] # retain only the top nv eigenvectors 

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
    
    return np.sqrt(np.sum(np.abs(data)**2, axis=3))

# shift, compute fourier transform, shift over x, y, z axes
image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(virtual_coil_data, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))
image_rss = rsos(image) # find root sum of squares of image over the channel dimension
kspace_rss = rsos(virtual_coil_data) # compute rsos of k-space data

# figure = plt.figure(figsize = (20, 20))
# for coil in range(virtual_coil_data.shape[2]):
#     plt.subplot(8, 6, coil+1)
#     plt.imshow(np.abs(np.rot90(image[:, :, coil, z_slice],3)), cmap='gray')
#     plt.imshow(maskA[:, :, z_slice], alpha = 0.2, cmap='Greens')
#     plt.imshow(maskB[:, :, z_slice], alpha = 0.2, cmap='Reds')
#     plt.title(f'Virtual Coil {coil+1}')
# plt.show()

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
plt.imshow(image_axial, cmap='gray', vmin=0,vmax=0.1)
rect = patches.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                         linewidth=2, edgecolor='g', facecolor='none')
plt.gca().add_patch(rect)

# plt.imshow(maskA[x_slice, :, :], alpha = 0.1, cmap='Pastel1')
# plt.imshow(maskB[x_slice, :, :], alpha = 0.1, cmap='Greens')

plt.subplot(3,2,3)
plt.title(f'Coronal K-space y = {y_slice}')
plt.imshow(np.log(np.abs(kspace_cor)+1e-9), cmap='gray')

plt.subplot(3,2,4)
plt.title(f'Coronal View y = {y_slice}')
plt.imshow(np.rot90(image_cor,3), cmap='gray')

plt.subplot(3,2,5) 
plt.title(f'Sagittal K-space x = {x_slice}')
plt.imshow(np.log(np.abs(kspace_sag)+1e-9), cmap='gray')



plt.subplot(3,2,6)
plt.title(f'Sagittal View x = {x_slice}')
plt.imshow(image_sag, cmap='gray', vmin = 0, vmax = 0.1)
# plt.imshow(maskA[x_slice, :, :], alpha = 0.1, cmap='Pastel1')
# plt.imshow(maskB[x_slice, :, :], alpha = 0.5, cmap='Reds')
rect = patches.Rectangle((z_start, y_start), z_end-z_start, y_end-y_start,
                         linewidth=2, edgecolor='g', facecolor='none')
plt.gca().add_patch(rect)


# name = f'Reconstructed Image Axial {z_slice}, Coronal {y_slice}, Sagittal {x_slice}.png'
# path = os.path.join(r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer", name)
# plt.savefig(path)
plt.show() 