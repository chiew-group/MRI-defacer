import matplotlib.pyplot as plt
import numpy as np 
import os

def display(image_rss, kspace_rss, x_slice, y_slice, z_slice):

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
    plt.imshow(image_axial, cmap='gray')
    # plt.imshow(maskB[:, :, z_slice], alpha = 0.3, cmap = 'Reds')

    plt.subplot(3,2,3)
    plt.title(f'Coronal K-space y = {y_slice}')
    plt.imshow(np.log(np.abs(kspace_cor)+1e-9), cmap='gray')

    plt.subplot(3,2,4)
    plt.title(f'Coronal View y = {y_slice}')
    plt.imshow(image_cor, cmap='gray')
    # plt.imshow(maskB[:, y_slice, :], alpha = 0.3, cmap = 'Reds')

    plt.subplot(3,2,5) 
    plt.title(f'Sagittal K-space x = {x_slice}')
    plt.imshow(np.log(np.abs(kspace_sag)+1e-9), cmap='gray')

    plt.subplot(3,2,6)
    plt.title(f'Sagittal View x = {x_slice}')
    plt.imshow(image_sag, cmap='gray')
    # plt.imshow(maskB[x_slice, :, :], alpha = 0.3, cmap = 'Reds')

    # name = f'Reconstructed Image Axial {z_slice}, Coronal {y_slice}, Sagittal {x_slice}.png'
    # path = os.path.join(r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer", name)
    # plt.savefig(path)

    plt.show()