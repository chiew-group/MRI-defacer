import numpy as np 
import nibabel as nib 
import matplotlib.pyplot as plt
from totalsegmentator.python_api import totalsegmentator 

data = np.load(r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\reduced_data.npy")
data = np.moveaxis(data, -1, 2)

image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data, axes = (0, 1, 2)), axes = (0, 1, 2)), axes = (0, 1, 2))
image =  np.sqrt(np.sum(np.abs(image)**2, axis=3))


affine = np.diag([1, 1, 1, 1])
nifti_img = nib.Nifti1Image(image.astype(np.float32), affine)

inpath = r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\imgtoseg.nii"
nib.save(nifti_img, r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\imgtoseg.nii")


# if __name__ == "__main__":
#         out = totalsegmentator(
#                         input = r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\imgtoseg_ras.nii", 
#                         output = r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\totseg.nii",
#                         task = "face_mr"
#         )

# print("done")


# img = nib.load(r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\face.nii")
# img_np = img.get_fdata()
# print("Shape:", img.shape)
# print("Voxel size:", img.header.get_zooms())
# print("Orientation (axcodes):", nib.orientations.aff2axcodes(img.affine))


# print("Data min/max:", np.min(img_np), np.max(img_np))

# print(img_np.shape)
# print(img_np)
# print(np.count_nonzero(img_np))
# plt.subplot(3,2,1)
# plt.imshow(img_np[:, 60, :], cmap='Reds')
# plt.show() 
