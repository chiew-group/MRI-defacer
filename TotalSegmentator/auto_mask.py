import numpy as np 
import nibabel as nib 
import matplotlib.pyplot as plt
from totalsegmentator.python_api import totalsegmentator 
from scipy.ndimage import zoom

if __name__ == "__main__":

        # calling TotalSegmentator API
        totalsegmentator(
                        input = r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\TotalSegmentator\input_img.nii", 
                        output = r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\TotalSegmentator\output_img.nii",
                        task = "face_mr"
        ) # saved nifti mask to 

        # checking the mask
        img = nib.load(r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\TotalSegmentator\output_img.nii\face.nii.gz")
        img_np = img.get_fdata()

        # print(f'Max value: {np.min(img_np)} Min value: {np.max(img_np)}')
        # print(img_np)
        # print(np.count_nonzero(img_np))

        # resize the mask to fit the reduced data
        # does totseg always give mask shape that fits og image? should i do this step anyway ? 
        reduced_img = np.load(r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\reduced_data.npy")
        reduced_img = np.moveaxis(reduced_img, -1, 2)
        print(f'initial mask shape: {img_np.shape} desired shape: {reduced_img.shape[:3]}')
        zoom_factor = np.array(reduced_img.shape[:3])/np.array(img_np.shape)
        img_np = zoom(img_np, zoom = zoom_factor, order = 0) # what is the order?
        print(f'new mask shape: {img_np.shape} desired shape: {reduced_img.shape[:3]}')
        print(f'nonzero mask values: {np.count_nonzero(img_np)}')
        
        # saving mask as numpy file
        np.save(r"C:\Users\selin\OneDrive\Desktop\SRI\MRI defacer\TotalSegmentator\mask.npy", img_np)
        print("Mask saved successfully")

