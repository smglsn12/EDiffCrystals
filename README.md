# EDiffCrystals
A python package allowing automated crystal structure analysis from electron diffraction patterns. 
This code base accompanies the manuscript "Random Forest Prediction of Crystal Structure from Electron Diffraction Patterns Incorporating Multiple Scattering" 
(https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.8.093802)

To run, first clone this repo and extract the 'data' folder from https://drive.google.com/drive/folders/1i1QaXYldgbIh9v46FMnPFRrA6nvFWetG?usp=drive_link and 
replace the empty data folder in this repo with the downloaded folder. Initialize a conda environment using the included .yml file or by installing the module 
in your local environment. 

To access the full simulated data containing ~36,600 materials that each have 10,000 unique electron diffraction patterns, go to the above google drive link
and navigate to the "Raw_Simulated_Bragg_Lists" folder. This containes a .pkl file for each material. This full simulated dataset is currently being incorperated 
into the materials project database for more efficient distribtion. Check back here for updates! 
