import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 

from scipy.linalg import orth
from scipy.linalg import norm

from global_ROVir import form_virtual_coil_data

from IPython import get_ipython

if 'ipykernel' in str(get_ipython()):
        matplotlib.use('module://matplotlib_inline.backend_inline')

else:
    matplotlib.use('Qt5Agg')         

def plot_metrics(xaxis, coil, nc, sirs, brain_signal, face_signal, brain_retain, face_retain, show_metric_graphs, save_metric_graphs, title =''):

    plt.figure(figsize=(16, 10), dpi=120)

    # visualize metrics as scatterplots
    plt.subplot(2,2,1)
    plt.scatter(xaxis, sirs, c = "blue")
    plt.axvline(x=coil+1, linestyle = "--", label = "Threshold")
    plt.title(f"SIR of Each Virtual Coil {title}")
    plt.xlabel("jth Coil")
    plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)

    plt.subplot(2,2,2)
    plt.scatter(xaxis, brain_signal, c = "green")
    plt.axvline(x=coil+1, linestyle = "--", label = "Threshold")
    plt.title(f"Signal Energy of Each Virtual Coil {title}")
    plt.xlabel("jth Coil")
    plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)

    plt.subplot(2,2,3)
    plt.scatter(xaxis, face_signal, c = "red")
    plt.axvline(x=coil+1, linestyle = "--", label = "Threshold")
    plt.title(f"Interference Energy of Each Virtual Coil {title}")
    plt.xlabel("jth Coil")
    plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    
    plt.subplot(2,2,4)
    plt.scatter(np.arange(1, nc+1), np.array(brain_retain), label = "Brain Retained", c= "green")
    plt.scatter(np.arange(1, nc+1), np.array(face_retain), label = "Face Retained", c= "red")
    plt.axvline(x=coil+1, linestyle = "--", label = "Threshold")
    plt.ylabel("Percentage Retained (%)")
    plt.xlabel("Total Channels Retained (N)")
    plt.legend()
    plt.title(f"Brain/Face Signal Retention vs Virtual Coils Retained {title}")
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.tight_layout(pad=3)

    if save_metric_graphs:
        plt.savefig(F'results/blah.png') # CHANGE THIS LATER

    if show_metric_graphs:
        plt.show()


def display_defaced(og_image_rsos, defaced_image_rsos, x_slice, y_slice, z_slice, 
            maskA, maskB, show_summary, save_summary):
    '''
    Visualize the x_slice, y_slice, and z_slice slices of the
    original image, defaced image, and masks overlayed. 

    Parameters
    ----------
        og_image_rsos -> np.ndarray: the original image data after applying rsos, shape (x, y, z)
        defaced_image_rsos -> np.ndarray: the defaced image data after applying rsos, shape (x, y, z)
        x_slice -> int: the slice taken in the x axis for the sagittal view
        y_slice -> int: the slice taken in the y axis for the coronal view
        z_slice -> int: the slice taken in the z axis for the axial view
        maskA -> np.ndarray: the mask defined for the brain region
        maskB -> np.ndarray: the mask defined for the face region
        brain_retain -> float: the percentage of signal from the original image left within maskA
        face_retain -> float: the percentage of signal from the original image left within maskB
    '''

    # getting the axial, coronal, and sagittal slices of the original image
    og_ax = np.rot90(og_image_rsos[:, :, z_slice]) 
    og_cor = np.rot90(og_image_rsos[:, y_slice, :]) 
    og_sag = np.rot90(og_image_rsos[x_slice, :, :]) 

    # getting the axial, coronal, and sagittal slices of the defaced image
    defaced_ax = np.rot90(defaced_image_rsos[:, :, z_slice]) 
    defaced_cor = np.rot90(defaced_image_rsos[:, y_slice, :])
    defaced_sag = np.rot90(defaced_image_rsos[x_slice, :, :])

    # getting the axial, coronal, and sagittal slices of the brain mask
    maskA_ax = np.rot90(maskA[:, :, z_slice]) 
    maskA_cor = np.rot90(maskA[:, y_slice, :])
    maskA_sag = np.rot90(maskA[x_slice, :, :])

    # getting the axial, coronal, and sagittal slices of the face mask
    maskB_ax = np.rot90(maskB[:, :, z_slice]) 
    maskB_cor = np.rot90(maskB[:, y_slice, :])
    maskB_sag = np.rot90(maskB[x_slice, :, :])

    # getting ratio image to deisplay loss/retention heatmap
    ratio = defaced_image_rsos/og_image_rsos 

    # get global max and min to set vmin data range 
    global_vmin = np.min([ratio[:, :, z_slice].min(), ratio[:, y_slice, :].min(), ratio[x_slice, :, :].min()])
    global_vmax = np.max([ratio[:, :, z_slice].max(), ratio[:, y_slice, :].max(), ratio[x_slice, :, :].max()])

    plt.subplot(3,4,1)
    plt.title(f'Original Images')
    plt.axis('off')
    plt.imshow(og_ax, cmap='gray')
    plt.subplot(3,4,5)
    plt.axis('off')
    plt.imshow(og_cor, cmap='gray')
    plt.subplot(3,4,9)
    plt.axis('off')
    plt.imshow(og_sag, cmap='gray')

    plt.subplot(3,4,2)
    plt.axis('off')
    plt.title(f'Defaced Images')
    plt.imshow(defaced_ax, cmap='gray')
    plt.subplot(3,4,6)
    plt.axis('off')
    plt.imshow(defaced_cor, cmap='gray') 
    plt.subplot(3,4,10)
    plt.axis('off')
    plt.imshow(defaced_sag, cmap='gray')

    plt.subplot(3,4,3)
    plt.axis('off')
    plt.title(f'Brain/Face Masks')
    plt.imshow(defaced_ax, cmap='gray')
    plt.imshow(maskB_ax, alpha = 0.3, cmap = 'Reds')
    plt.imshow(maskA_ax, alpha = 0.3, cmap = 'Greens')
    plt.subplot(3,4,7)
    plt.axis('off')
    plt.imshow(defaced_cor, cmap='gray')
    plt.imshow(maskB_cor, alpha = 0.3, cmap = 'Reds')
    plt.imshow(maskA_cor, alpha = 0.3, cmap = 'Greens')
    plt.subplot(3,4,11)
    plt.axis('off')
    plt.imshow(defaced_sag, cmap='gray')
    plt.imshow(maskB_sag, alpha = 0.3, cmap = 'Reds')
    plt.imshow(maskA_sag, alpha = 0.3, cmap = 'Greens')

    plt.subplot(3,4,4)
    plt.axis('off')
    plt.title(f'Ratio Heatmap')
    ratio_ax = plt.imshow(np.rot90(ratio[:, :, z_slice]), cmap='RdYlGn', vmin=global_vmin, vmax=global_vmax)
    plt.subplot(3,4,8)
    plt.axis('off')
    ratio_cor = plt.imshow(np.rot90(ratio[:, y_slice, :]), cmap='RdYlGn', vmin=global_vmin, vmax=global_vmax)
    plt.subplot(3,4,12)
    plt.axis('off')
    ratio_sag = plt.imshow(np.rot90(ratio[x_slice, :, :]), cmap='RdYlGn', vmin=global_vmin, vmax=global_vmax)

    ratio_colourbar_ax = plt.axes([0.92, 0.05, 0.02, 0.9])
    ratio_colourbar = plt.colorbar(ratio_sag, cax=ratio_colourbar_ax, orientation='vertical')
    ratio_colourbar.set_label('Retention Ratio', fontsize=10)

    # x_slices = og_image_rsos.shape[0]
    # (nrow, ncol) = closest_factors(x_slices)
    # for x_slice in range(0, x_slices):
   
    #     plt.subplot(nrow,ncol,x_slice+1)
    #     plt.axis('off')
    #     plt.text(0, 0, f'{x_slice}', fontsize=6, color='black')
    #     plt.imshow(np.rot90(ratio[x_slice, :, :]), cmap='RdYlGn', vmin=global_vmin, vmax=global_vmax)   
    # plt.show()

    # x_slices = og_image_rsos.shape[0]
    # (nrow, ncol) = closest_factors(x_slices)
    # for x_slice in range(0, x_slices):
   
    #     plt.subplot(nrow,ncol,x_slice+1)
    #     plt.axis('off')
    #     plt.text(0, 0, f'{x_slice}', fontsize=6, color='black')
    #     plt.imshow(np.rot90(defaced_image_rsos[x_slice, :, :]), cmap='gray')   
    #     plt.imshow(np.rot90(maskB[x_slice, :, :]), alpha = 0.3, cmap = 'Reds')
    #     plt.imshow(np.rot90(maskA[x_slice, :, :]), alpha = 0.3, cmap = 'Greens')

    if save_summary:
        plt.savefig(f'results/hello.png') #CHANGE THIS NAMING CONVENTION

    if show_summary:
        plt.show()


def compare_retention(data, eigenvec, brain_covar, face_covar, nc, x_slice, show_compare_retention, save_compare_retention):
    '''
    Visualizes the image retaining 1, 2, 3, ..., nc virtual coils to
    compare the defacing results depending on the number of top
    virtual coils retained.

    Parameters
    ----------
        data -> np.ndarray: the original k-space data
        eigenvec -> np.ndarray: the array with columns containing the ordered eigenvectors/optimal virtual coil weights
        brain_covar -> np.ndarray: the covariance matrix correlated with the brain region maskA
        face_covar -> np.ndarray: the covariance matrix correlated with the face region maskB
        nc -> int: the number of original coils 
        x_slice -> int: the slice number for the sagittal view desired for visualization 
    '''

    (nrow, ncol) = closest_factors(nc)

    # display rsos image for retaining 1, 2, 3, ..., nc virtual coils 
    for i in range(1, nc+1):
        eigenvec_retain = orth(eigenvec[:, :i]) # retain i eigenvectors
        virtual_coil_data = form_virtual_coil_data(eigenvec_retain, data) # form the virtual coil data with the top eigenvectors

        # compute the image data from the virtual coil data 
        image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(virtual_coil_data, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))
        image_rsos = np.sqrt(np.sum(np.abs(image)**2, axis=3))
        
        # calculate the percentage of signal retained from the brain and face regions
        orth_proj = eigenvec_retain @ eigenvec_retain.conj().T
        brain_retain = (norm((orth_proj @ brain_covar @ orth_proj), ord = 'fro') / norm (brain_covar, ord = 'fro'))*100
        face_retain = (norm((orth_proj @ face_covar @ orth_proj), ord = 'fro') / norm (face_covar, ord = 'fro'))*100

        
        plt.subplot(nrow,ncol,i)
        plt.axis('off')
        plt.text(image_rsos[x_slice, :, :].shape[0]//2, 5, f'Brain: {round(brain_retain, 2)}% \n Face: {round(face_retain, 2)}%', color="white", fontsize=8, ha='center', va='top')
        plt.text(image_rsos[x_slice, :, :].shape[0]-5, image_rsos[x_slice, :, :].shape[1], f'{i}', color="white", fontsize=15, ha='right', va='bottom')
        # plt.imshow(np.rot90(image_rss[x_slice, :, :]), cmap = 'gray', vmin = 0 , vmax = 0.0000008) # changed windowing for the 48ch dataset
        plt.imshow(np.rot90(image_rsos[x_slice, :, :]), cmap = 'gray', vmin = 0 , vmax = 800000000000)

    if save_compare_retention:
        plt.savefig(f'results/fdsal.png') # CHANGE THIS LATER
    if show_compare_retention:
        plt.show()

def closest_factors(n):
    return next((i, n // i) for i in range(int(np.sqrt(n)), 0, -1) if n % i == 0)

def show_virtual_coils(data, eigenvec, x_slice):
    '''
    Visualizes all virtual coils individually.

    Parameters
    ----------
        data -> np.ndarray: the original k-space data
        eigenvec -> np.ndarray: the array with columns containing the ordered eigenvectors/optimal virtual coil weights
        x_slice -> int: the slice number for the sagittal view desired for visualization 
    '''

    # compute virtual coil data using all eigenvectors and image data with all virtual coils, without rsos
    all_coils = form_virtual_coil_data(eigenvec, data) # form all virtual coils from all eigenvectors
    image_all_coils = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(all_coils, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))

    # display each virtual coil's data
    (nr, nc) = closest_factors(all_coils.shape[3])
    for coil in range(all_coils.shape[3]):
        plt.subplot(nr,nc, coil+1)
        # plt.imshow(np.abs(image_all[x_slice, :, :, coil]), cmap='gray',vmin = 0, vmax = 0.0000001)
        plt.imshow(np.abs(np.rot90(image_all_coils[x_slice, :, :, coil])), cmap='gray', vmin = 0, vmax = 200000000000)
        plt.axis('off')
        # plt.title(f'Virtual Coil {coil+1}')
    plt.show()



def display_virtual_coils(virtual_coil_data, slice_index, name, show_virtual_coil_images, save_virtual_coil_images, axis=0):
    '''
    Visualizes each individual virtual coil's reconstructed image at a single slice
    along the given spatial axis.

    Parameters
    ----------
        virtual_coil_data -> np.ndarray: the defaced k-space data with virtual coils, shape (x, y, z, nv)
        slice_index -> int: the slice index along `axis` to visualize
        axis -> int: which spatial axis to slice along (0 = x/sagittal, 1 = y/coronal, 2 = z/axial)
    '''
    import matplotlib.pyplot as plt

    view_names = {0: 'sagittal x', 1: 'coronal y', 2: 'axial z'}

    nv = virtual_coil_data.shape[3]
    (nrow, ncol) = closest_factors(nv)


    vmin = 0
    vmax = np.percentile(np.abs(np.concatenate([
        virtual_coil_data.ravel()
    ])), 99.5)

    plt.rcParams['savefig.dpi']=600
    plt.figure()
    plt.suptitle(f'{name} ({view_names[axis]} slice: {slice_index})')
    for coil in range(nv):
        # reconstruct one channel at a time and keep only the slice of interest, to avoid
        # holding every channel's full 3D image in memory at once

        plt.subplot(nrow, ncol, coil + 1)
        plt.axis('off')
        plt.text(0, 0, f'{coil}', color='red')

        if axis == 0 : 
            plt.imshow(np.rot90(np.abs(virtual_coil_data[slice_index,:,:,coil])), cmap='gray', vmin=vmin, vmax=vmax)
        elif axis == 2:
            plt.imshow(np.rot90(np.abs(virtual_coil_data[:,:,slice_index,coil])), cmap='gray', vmin=vmin, vmax=vmax)

    if save_virtual_coil_images:
        plt.savefig('results/CHAGNRE.png')# CHANGE THIS LATER
    if show_virtual_coil_images:
        plt.show()
