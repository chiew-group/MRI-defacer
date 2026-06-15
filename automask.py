import numpy as np
import nibabel as nib
import os
import subprocess
import shutil
# from scipy.ndimage import zoom

GREEN = '\033[92m'
RESET = '\033[0m'

'''
Wrapper for CHARM CLI tool.

'''
def gen_mask(input_fp, output_fp, saveID):
        '''
        To generate numpy binary masks for the brain and face regions using GRACE CLI. 
        
        Parameters
        ----------
        input_fp -> str: path to nifti file of initial reconstruction
        output_fq -> str: path to output nifti masks of face and brain
        saveID -> str: ID to save numpy masks 
        '''
        charm_path = shutil.which("charm.cmd") # user must have SimNIBS installed to path
        command = [charm_path,
                   saveID,
                   input_fp, "--noneck", "--forcesform", "--forcerun"]
        
        subprocess.run(command) # run CHARM CLI tool
        
        seg_img = nib.load(f'm2m_{saveID}/final_tissues.nii.gz') # load segmentation results 
        all_tissues = seg_img.get_fdata()
        
        lut_pf = f"m2m_{saveID}/final_tissues_LUT.txt" # file path for look up table

        name_key_map = {} # to store mapping for label name and key in segmentation file

        # use the look up table to find the segmentation labels 
        with open(lut_pf, 'r') as LUT:
                for line in LUT:
                        if not line.strip() or line.startswith('#'):
                                continue # skip empty lines and first line
                        
                        line_split = line.split() 

                        # add the tissue name and tissues code
                        name_key_map[line_split[1]] = int(line_split[0])

        brain_names = ['White-Matter', 'Gray-Matter', 'CSF']
        brain_labels = [name_key_map[name] for name in brain_names]
        
        
        face_names = ['Bone', 'Scalp', 
                      'Eye_balls', 'Muscle', 
                      'Cartilage', 'Fat']
        face_labels = [name_key_map[name] for name in face_names]
        
        brain_mask = np.isin(all_tissues, brain_labels).astype(np.uint8)
        face_mask = np.isin(all_tissues, face_labels).astype(np.uint8)

        os.makedirs(output_fp, exist_ok=True)

        brain_fp = os.path.join(output_fp, "brain.nii.gz")
        face_fp = os.path.join(output_fp, "face.nii.gz")
        nib.save(nib.Nifti1Image(brain_mask, seg_img.affine), brain_fp)
        nib.save(nib.Nifti1Image(face_mask, seg_img.affine), face_fp)

        print(GREEN + "Mask saved successfully" + RESET)

        print(GREEN + "Mask saved successfully" + RESET)
