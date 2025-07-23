import numpy as np 
import nibabel as nib 
from totalsegmentator.python_api import totalsegmentator 
import os

def gen_mask(input_fp, output_fp):
        # calling TotalSegmentator API
        totalsegmentator(
                        input = input_fp, 
                        output = output_fp,
                        task = "face_mr"
        )

        totalsegmentator(
                        input = input_fp , 
                        output = output_fp,
                        task = "tissue_types_mr"
        )
        
        totalsegmentator(
                        input = input_fp, 
                        output = output_fp, 
                        task = "total_mr", 
                        roi_subset = ["brain"]
        )

        face_mask = nib.load(os.path.join(output_fp, "face.nii.gz"))
        sub_fat_mask = nib.load(os.path.join(output_fp, "subcutaneous_fat.nii.gz"))
        muscle_mask = nib.load(os.path.join(output_fp, "skeletal_muscle.nii.gz"))
        brain_mask = nib.load(os.path.join(output_fp, "brain.nii.gz"))

        face_mask_np = face_mask.get_fdata()
        sub_fat_mask_np = sub_fat_mask.get_fdata()
        muscle_mask_np = muscle_mask.get_fdata()
        brain_mask_np = brain_mask.get_fdata()
        
        np.save(r"results\face_mask.npy", face_mask_np)
        np.save(r"results\fat_mask.npy", sub_fat_mask_np)
        np.save(r"results\muscle_mask.npy", muscle_mask_np)
        np.save(r"results\brain_mask.npy", brain_mask_np)

        print("Mask saved successfully")
