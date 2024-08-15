# Sadaf_Dargahi_DGA1032
These codes show experiments designed to evaluate our proposed deep learning model for estimating field maps used in correcting susceptibility artifacts in diffusion MRI (dMRI). We train our model using four different loss functions (L1 norm, L1 norm combined with Total Variation (TV), Normalized Cross-Correlation (NCC), and L2 norm) to compare the estimated field map (F) generated by the model with the ground truth field map F*, obtained using the TOPUP algorithm. The goal is to evaluate the impact of each loss function on the model's performance. As the next step, the two field maps are compared visually, and then we apply the field maps to the DWI data to correct b0 volumes (the first volume of each DWI data) and compare our method's results with those produced by TOPUP. 
