import numpy as np 
import nibabel as nib 

# # turning k-space data into image for segmentation
# data = np.load(r"C:\Users\selin\MRI defacer\numpy_data2.npy")
# data = np.moveaxis(data, -1, 2)

# # compute the fourier transform, get rss
# image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data, axes = (0, 1, 2)), axes = (0, 1, 2)), axes = (0, 1, 2))
# image =  np.sqrt(np.sum(np.abs(image)**2, axis=3))
# image = (image - image.min()) / (image.max() - image.min()) * 255 #min-max norm to 0-255 range

# image = image[:, :, :]

# print(f'Max value: {np.max(image)} Min value: {np.min(image)}') #checking image intensity range
# print(image.shape)

# # converting numpy data to nifti file 
# affine = np.array([[1.0, 0.0, 0.0, -image.shape[0]//2], 
#                    [0.0, 1.0, 0.0, -image.shape[1]//2],
#                    [0.0, 0.0, 1.0, -image.shape[2]//2],
#                    [0.0, 0.0, 0.0, 1.0]
                   
# ]) 
# nifti_img = nib.Nifti1Image(image.astype(np.uint8), affine)
# print(nifti_img.affine)

# nib.save(nifti_img, r"C:\Users\selin\MRI defacer\TotalSegmentator\input_img2.nii")
# print("Nifti file successfully saved")
GREEN = '\033[92m'
RESET = '\033[0m'

def niftify(image, a):
    
    image =  np.sqrt(np.sum(np.abs(image)**2, axis=3))
    image = (image - image.min()) / (image.max() - image.min()) * 255 
    
    # converting numpy data to nifti file 
    affine = np.array([[a, 0.0, 0.0, -image.shape[0]//2], 
                    [0.0, a, 0.0, -image.shape[1]//2],
                    [0.0, 0.0, a, -image.shape[2]//2],
                    [0.0, 0.0, 0.0, 1.0]
                    
    ]) 
    nifti_img = nib.Nifti1Image(image.astype(np.uint8), affine)
    print(nifti_img.affine)

    nib.save(nifti_img, r"results\input_image.nii")
    print(GREEN + "Nifti file successfully saved" + RESET)
