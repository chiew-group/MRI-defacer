# MRI Defacer
{name of tool} is a defacing tool for anonymizing MRI data. It employs TotalSegmentator's face_mr task [2] to automatically generate a mask for the facial region, then uses region-optmized virtual coils (ROVir) [1] to deface the MRI data in k-space.

work in progress...!

## Installation and Usage 
Set up a Python environment for (name of tool)

```
conda create -n MRIDEFACER python=3.13
```

Install TotalSegmentator and the following required dependencies

```
pip install TotalSegmentator 
pip install matplotlib
pip install pyq5
pip install scipy
pip install ismrmrd
```

Clone this GitHub repository

```
git clone https://github.com/chiew-group/MRI-defacer.git
cd MRI-defacer
```
Edit the config.json file to specify the input data path, slice information, and SIR threshold. { name of tool} excepts raw MRI data in the MRD data format. Run main.py to get defaced images. 

## Expected Input

To change the input data required for defacing, navigate to the config.json file. Below are expected types and input descriptions to guide user input. 

| Parameter     | Expected Type | Description   | 
| ------------- | ------------- | ------------- | 
| data_path     | file path     | path to raw MRI data in MRD format  |
| x_slice       | int           | x slice number for the coronal view  | 
| y_slice       | int           | y slice number for the saggital view |
| z_slice       | int           | z slice number for the axial view  | 
| sir_threshold | int           | threshold to choose top signal coils  |
| gap           | int           | size of buffer between region of interest and inteference region  | 
| x_y_z_channel | list          | list[0] = shape index for sagittal plane <br/> list[1] = shape index for coronal plane <br/> list[2] = shape index for axial plane <br/> list[3] = shape index for channel axis | 
| voxel_space   | float         | image voxel resolution in millimeters | 

## Troubleshooting
1. Wrong affine or orientation: check direction, voxel spacing
2.  


## Citations 
[1] Kim D, Cauley SF, Nayak KS, Leahy RM, Haldar JP. Region- optimized virtual (ROVir) coils: Localization and/or suppression of 
spatial regions using sensor- domain beamforming. Magn Reson Med. 2021;86:197–212. 
https://doi.org/10.1002/mrm.28706 

[2] Wasserthal, J., Breit, H.-C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., Segeroth, M., 2023. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.230024