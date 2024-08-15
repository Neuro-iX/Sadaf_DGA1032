#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:38:10 2024

@author: at90180
"""



import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
from glob import glob
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import mutual_info_score
from scipy.ndimage import map_coordinates
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mutual_info_score

class NiftiDataset(Dataset):
    def __init__(self, b0_folder, t1_folder, ref_FM_topup_file, normalize=True, slice_orientation='axial', resize=None):
        self.b0_folder = b0_folder
        self.t1_folder = t1_folder
        self.ref_FM_topup_file = ref_FM_topup_file
        self.normalize = normalize
        self.slice_orientation = slice_orientation
        self.resize = resize

        self.b0_subject_ids = [self.extract_subject_id(filename) for filename in sorted(glob(os.path.join(b0_folder, 'sub-*_AP_b0_brain_mask.nii.gz')))]
        self.t1_subject_ids = [self.extract_subject_id(filename) for filename in sorted(glob(os.path.join(t1_folder, 'sub-*_T1-mask_registered.nii.gz')))]
        self.ref_FM_topup_subject_ids = [self.extract_subject_id(filename) for filename in sorted(glob(os.path.join(ref_FM_topup_file, '*_AP_PA_topup_fout.nii.gz')))]

        self.valid_subject_ids = set(self.b0_subject_ids) & set(self.t1_subject_ids) & set(self.ref_FM_topup_subject_ids)
        if not self.valid_subject_ids:
            raise ValueError("No matching subject IDs found between input and target files.")

        self.valid_files = [(os.path.join(b0_folder, f'sub-{subject_id}_AP_b0_brain_mask.nii.gz'),
                             os.path.join(t1_folder, f'sub-{subject_id}_T1-mask_registered.nii.gz'),
                             os.path.join(ref_FM_topup_file, f'{subject_id}_AP_PA_topup_fout.nii.gz'))
                            for subject_id in self.valid_subject_ids]

        self.b0_files = [item[0] for item in self.valid_files]
        self.t1_files = [item[1] for item in self.valid_files]
        self.ref_FM_topup_files = [item[2] for item in self.valid_files]

        if self.normalize:
            self.calculate_mean_std()

    def extract_subject_id(self, filename):
        basename = os.path.basename(filename)
        if 'sub-' in basename:
            return basename.split('_')[0][4:]
        elif 'Intensity_geometric_' in basename:
            return basename.split('_')[2]
        else:
            return basename.split('_')[0]

    def calculate_mean_std(self):
        b0_means, b0_stds, t1_means, t1_stds = [], [], [], []
        for b0_file, t1_file in zip(self.b0_files, self.t1_files):
            b0_img = nib.load(b0_file).get_fdata()
            t1_img = nib.load(t1_file).get_fdata()
            b0_means.append(np.mean(b0_img))
            b0_stds.append(np.std(b0_img))
            t1_means.append(np.mean(t1_img))
            t1_stds.append(np.std(t1_img))
        self.b0_mean = np.mean(b0_means)
        self.b0_std = np.mean(b0_stds)
        self.t1_mean = np.mean(t1_means)
        self.t1_std = np.mean(t1_stds)


    def __len__(self):
        return min(len(self.b0_files), len(self.t1_files))

    def __getitem__(self, idx):
        b0_img = nib.load(self.b0_files[idx]).get_fdata()
        t1_img = nib.load(self.t1_files[idx]).get_fdata()
        ref_FM_topup_img = nib.load(self.ref_FM_topup_files[idx]).get_fdata()

        if self.slice_orientation == 'axial':
            slices_b0 = [b0_img[:, :, i] for i in range(b0_img.shape[2])]
            slices_t1 = [t1_img[:, :, i] for i in range(t1_img.shape[2])]
            slices_ref_FM_topup = [ref_FM_topup_img[:, :, i] for i in range(ref_FM_topup_img.shape[2])]
        elif self.slice_orientation == 'sagittal':
            slices_b0 = [b0_img[i, :, :] for i in range(b0_img.shape[0])]
            slices_t1 = [t1_img[i, :, :] for i in range(t1_img.shape[0])]
            slices_ref_FM_topup = [ref_FM_topup_img[i, :, :] for i in range(ref_FM_topup_img.shape[0])]
        elif self.slice_orientation == 'coronal':
            slices_b0 = [b0_img[:, i, :] for i in range(b0_img.shape[1])]
            slices_t1 = [t1_img[:, i, :] for i in range(t1_img.shape[1])]
            slices_ref_FM_topup = [ref_FM_topup_img[:, i, :] for i in range(ref_FM_topup_img.shape[1])]
        else:
            raise ValueError("Invalid slice orientation. Choose from 'axial', 'sagittal', or 'coronal'.")

        slice_idx = b0_img.shape[2] // 2 
        slice_b0 = np.stack([slices_b0[slice_idx-1], slices_b0[slice_idx], slices_b0[slice_idx+1]], axis=0)
        slice_t1 = np.stack([slices_t1[slice_idx-1], slices_t1[slice_idx], slices_t1[slice_idx+1]], axis=0)
        slice_ref_FM_topup = slices_ref_FM_topup[slice_idx]  

        if self.normalize:
            slice_b0 = (slice_b0 - self.b0_mean) / self.b0_std
            slice_t1 = (slice_t1 - self.t1_mean) / self.t1_std

        if self.resize:
            resize_transform = transforms.Resize(self.resize)
            slice_b0 = resize_transform(slice_b0)
            slice_t1 = resize_transform(slice_t1)

        slice_b0 = torch.from_numpy(slice_b0).float()
        slice_t1 = torch.from_numpy(slice_t1).float()
        slice_ref_FM_topup = torch.from_numpy(slice_ref_FM_topup).float()

        return slice_b0, slice_t1, slice_ref_FM_topup

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.conv5 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x4 = self.relu(self.conv4(x))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class UNetWithASPPSE(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(UNetWithASPPSE, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.se1 = SEBlock(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.se2 = SEBlock(128)
        # Bottleneck
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.aspp = ASPP(256, 256)
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.upconv5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 64, 3, padding=1)
        # Output
        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        x1 = torch.relu(self.conv1(x))
        x1 = torch.relu(self.conv2(x1))
        x1 = self.se1(x1)
        x2 = self.pool1(x1)
        x2 = torch.relu(self.conv3(x2))
        x2 = torch.relu(self.conv4(x2))
        x2 = self.se2(x2)
        x3 = self.pool2(x2)
        # Bottleneck
        x3 = torch.relu(self.conv5(x3))
        x3 = torch.relu(self.conv6(x3))
        x3 = self.aspp(x3)
        # Decoder
        x4 = self.upconv4(x3)
        x4 = torch.cat((x2, x4), dim=1)
        x4 = torch.relu(self.conv7(x4))
        x4 = torch.relu(self.conv8(x4))
        x5 = self.upconv5(x4)
        x5 = torch.cat((x1, x5), dim=1)
        x5 = torch.relu(self.conv9(x5))
        x5 = torch.relu(self.conv10(x5))
        output = self.output(x5)
        return output

    
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    start_time = time.time()
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for b0_imgs, t1_imgs, ref_FM_topup_imgs in train_loader:
            # Move tensors to the appropriate device
            b0_imgs = b0_imgs.to(device)
            t1_imgs = t1_imgs.to(device)
            ref_FM_topup_imgs = ref_FM_topup_imgs.to(device)
            
            # Stack b0 and t1 slices along the channel dimension
            inputs = torch.cat([b0_imgs, t1_imgs], dim=1)

            # Forward pass
            outputs = model(inputs)
            
            # Interpolate if the output size does not match the reference FM size
            # if outputs.shape[-2:] != ref_FM_topup_imgs.shape[-2:]:
            #     outputs = F.interpolate(outputs, size=ref_FM_topup_imgs.shape[-2:], mode='bilinear', align_corners=False)
            
            # Compute loss
            loss = criterion(outputs, ref_FM_topup_imgs.unsqueeze(1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate the training loss
            train_loss += loss.item()
            torch.cuda.empty_cache()

        # Calculate average loss for the epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)  
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Save the model weights after training
    torch.save(model.state_dict(), 'last_epoch_weights.pth')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    return train_losses




def calculate_mutual_information(x, y, num_bins=256):
    x_binned = np.digitize(x.ravel(), bins=np.linspace(x.min(), x.max(), num_bins))
    y_binned = np.digitize(y.ravel(), bins=np.linspace(y.min(), y.max(), num_bins))
    mi = mutual_info_score(x_binned, y_binned)
    return mi




def calculate_jacobian(vdm_volume, affine, header):

    dy = np.gradient(vdm_volume, axis=1)
    JField = 1 + dy
    
    return JField


def save_test_results(model, test_loader, dataset, device):
    model.eval()
    os.makedirs('/home/at90180/Documents/phd/third_session/DGA1032/take-home-exam/original/L1/b0_AP_corrected_L1/', exist_ok=True)
    os.makedirs('/home/at90180/Documents/phd/third_session/DGA1032/take-home-exam/original/L1/b0_difference_images_L1/', exist_ok=True)
    os.makedirs('/home/at90180/Documents/phd/third_session/DGA1032/take-home-exam/original/L1/FM_difference_images_L1/', exist_ok=True)
    os.makedirs('/home/at90180/Documents/phd/third_session/DGA1032/take-home-exam/original/L1/predicted_FM_L1/', exist_ok=True)
    
    
    global_mse_FM = 0.0  
    global_mi_FM = 0.0 
    global_mse_b0 = 0.0  
    global_mi_b0 = 0.0  
    count_FM = 0 
    count_b0 = 0  

    read_out_time = 0.040356 
    
    subject_counter = 0  

    with torch.no_grad():
        for i, (b0_imgs, t1_imgs, ref_FM_topup_imgs) in enumerate(test_loader):
            b0_imgs, t1_imgs, ref_FM_topup_imgs = [img.to(device) for img in b0_imgs], [img.to(device) for img in t1_imgs], [img.to(device) for img in ref_FM_topup_imgs]
            
            predicted_slices = []
            difference_images = [] 
            
            # Process for each subject
            # 1. Generate VDM slices
            # 2. Construct the full VDM volume
            # 3. Calculate the Jacobian field
            # 4. Apply VDM correction to DWI data
            # 5. Multiply the Jacobian with the DWI-corrected data
            
            for b0_img, t1_img in zip(b0_imgs, t1_imgs):
                inputs = torch.cat([b0_img.unsqueeze(0), t1_img.unsqueeze(0)], dim=1).to(device)
                inputs = inputs.view(1, 6, inputs.shape[-2], inputs.shape[-1])
                output = model(inputs).cpu().squeeze().numpy()
    
                if output.shape != ref_FM_topup_imgs[0].shape[-2:]:
                    output = F.interpolate(torch.from_numpy(output).unsqueeze(0).unsqueeze(0), size=ref_FM_topup_imgs[0].shape[-2:], mode='bilinear', align_corners=False)
                    output = output.squeeze().numpy()
                
                #global metrics for FM
                mse_FM = mean_squared_error(ref_FM_topup_imgs[0].cpu().numpy(), output)
                mi_FM = calculate_mutual_information(output, ref_FM_topup_imgs[0].cpu().numpy())
                
                global_mse_FM += mse_FM
                global_mi_FM += mi_FM
                count_FM += 1
                
                # difference images
                difference_image = output - ref_FM_topup_imgs[0].cpu().numpy()
                difference_images.append(difference_image)

                predicted_slices.append(output)
                  
            #Correct the input b0 using FM   
                
            #predicted field map volume
            predicted_FM_volume = np.stack(predicted_slices, axis=2)
            
            subject_id = dataset.extract_subject_id(dataset.ref_FM_topup_files[i])
            nib.save(nib.Nifti1Image(predicted_FM_volume, affine=np.eye(4)), os.path.join('/home/at90180/Documents/phd/third_session/DGA1032/take-home-exam/original/L1/predicted_FM_L1/', f'{subject_id}_predicted_FM_L1.nii.gz'))
                        
            # VDM volume    
            VDM_data = predicted_FM_volume * read_out_time
            
            # Load DWI data for the subject
            dwi_path = os.path.join('/home/at90180/Documents/phd/dataset/NIMH', f'sub-{subject_id}', 'ses-01', 'dwi', f'sub-{subject_id}_ses-01_dir-flipped_dwi.nii.gz')

            if not os.path.exists(dwi_path):
                print(f"DWI file not found for subject {subject_id}. Skipping...")
                continue

            dwi_img = nib.load(dwi_path)
            dwi_data = dwi_img.get_fdata()
            dwi_data = dwi_data[:, :, :, 0]

            # Get the middle slice of the DWI data
            mid_slice_idx = dwi_data.shape[2] // 2
            # print(mid_slice_idx)
            dwi_middle_slice = dwi_data[:, :, mid_slice_idx]
            
            # print(dwi_data.shape)

            # print(dwi_middle_slice.shape)
            # print(VDM_data.shape)
 
            # Create displacement grid for the middle slice of VDM data
            x_displaced, y_displaced = create_displacement_grid(VDM_data[:, :, 0])

            dwi_corrected_slice = apply_vdm_to_middle_slice(dwi_middle_slice, x_displaced, y_displaced)
                
                
            # Intesity correction, final corrected DWI data (b0 volume)
            jacobian_field = calculate_jacobian(VDM_data[:, :, 0], dwi_img.affine, dwi_img.header)

            jacobian_tensor = torch.tensor(jacobian_field, dtype=torch.float32)
            # print(dwi_corrected_slice.shape)
            dwi_corrected_tensor = torch.tensor(dwi_corrected_slice, dtype=torch.float32)
            # print(dwi_corrected_tensor[..., 0].shape)

            final_corrected_data = jacobian_tensor * dwi_corrected_tensor
            final_corrected_data_np = final_corrected_data.numpy()

            output_path = os.path.join('/home/at90180/Documents/phd/third_session/DGA1032/take-home-exam/original/L1/b0_AP_corrected_L1/', f'{subject_id}_b0_AP_corrected_L1.nii.gz')
            final_corrected_img = nib.Nifti1Image(final_corrected_data_np, dwi_img.affine, dwi_img.header)
            nib.save(final_corrected_img, output_path)
            
            
            # b0_TU file
            b0TU_path = os.path.join('/home/at90180/Documents/phd/third_session/Intensity_variation_correction/forallsubjects', f'Intensity_geometric_{subject_id}_AP_corrected_allvolumes.nii.gz')
            b0TU_img = nib.load(b0TU_path)
            b0TU_data = b0TU_img.get_fdata()

            # Extract the middle slice of b0TU_data[..., 0]
            middle_slice_idx_b0 = b0TU_data.shape[2] // 2
            b0TU_middle_slice = b0TU_data[..., 0][:, :, middle_slice_idx_b0]
            
            # Ensure that final_corrected_data_np and b0TU_middle_slice have the same shape
            if final_corrected_data_np.shape != b0TU_middle_slice.shape:
                raise ValueError(f"Shape mismatch: final_corrected_data_np shape {final_corrected_data_np.shape} does not match b0TU_middle_slice shape {b0TU_middle_slice.shape}")
            
            # Calculate MSE using the middle slice of b0TU_data
            mse_b0 = mean_squared_error(final_corrected_data_np.flatten(), b0TU_middle_slice.flatten())


            mi_b0 = calculate_mutual_information(final_corrected_data_np, b0TU_middle_slice)
            
            global_mse_b0 += mse_b0
            global_mi_b0 += mi_b0
            count_b0 += 1

            
            #difference images of FMs for 3 slices from 3 different subjects
            if subject_counter < 3:
                mid_slice_idx = len(difference_images) // 2
            
                # Prepare inputs for the model
                inputs = torch.cat((b0_imgs[mid_slice_idx].unsqueeze(0), t1_imgs[mid_slice_idx].unsqueeze(0)), dim=1).to(device)
                inputs = inputs.view(1, 6, inputs.shape[-2], inputs.shape[-1])  # Reshape to 4D tensor: [batch_size, channels, height, width]
                
                with torch.no_grad():
                    predicted_output = model(inputs).cpu().squeeze().numpy()
            
                if predicted_output.shape != ref_FM_topup_imgs[mid_slice_idx].shape[-2:]:
                    predicted_output = F.interpolate(
                        torch.from_numpy(predicted_output).unsqueeze(0).unsqueeze(0),
                        size=ref_FM_topup_imgs[mid_slice_idx].shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze().numpy()
            
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(ref_FM_topup_imgs[mid_slice_idx].cpu().squeeze(), cmap='gray')
                axes[0].set_title(f'Reference Field Map TOPUP\nSubject ID: {subject_id}')
                axes[1].imshow(predicted_output, cmap='gray')
                axes[1].set_title(f'Predicted Field Map Deep Learning\nSubject ID: {subject_id}')
                axes[2].imshow(difference_images[mid_slice_idx].squeeze(), cmap='gray')
                axes[2].set_title(f'Difference Image Between FM TOPUP and FM Deep Learning\nSubject ID: {subject_id}')
                plt.savefig(f'/home/at90180/Documents/phd/third_session/DGA1032/take-home-exam/original/L1/FM_difference_images_L1/FM_diff_subject_{subject_id}_L1.png')
                plt.show()



                # print (b0TU_data.shape)
                mid_slice_idx = b0TU_data.shape[2] // 2
                b0TU_slice = b0TU_data[:, :, mid_slice_idx,0]  # shape (128, 128)
                # print ("final_corrected_data_np", final_corrected_data_np.shape)
                final_corrected_slice = final_corrected_data_np


                #difference images of b0s for 3 middle slices from 3 different subjects
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(b0TU_slice, cmap='gray')
                axes[0].set_title(f'b0_TU Corrected\nSubject ID: {subject_id}')
                axes[1].imshow(final_corrected_slice, cmap='gray')
                axes[1].set_title(f'b0_DL Corrected\nSubject ID: {subject_id}')
                axes[2].imshow(final_corrected_slice - b0TU_slice, cmap='gray')
                axes[2].set_title(f'Difference Image Between b0_DL and b0_TU for\nSubject ID: {subject_id}')
                plt.savefig(f'/home/at90180/Documents/phd/third_session/DGA1032/take-home-exam/original/L1/b0_difference_images_L1/b0_diff_subject_{subject_id}_L1.png')
                plt.show()

                subject_counter += 1

    global_mse_FM /= count_FM
    global_mi_FM /= count_FM
    
   
    if count_b0 > 0:
        global_mse_b0 /= count_b0
        global_mi_b0 /= count_b0
        print(f"Global Mean Squared Error for b0DL vs b0TU: {global_mse_b0:.4f}")
        print(f"Global Mutual Information for b0DL vs b0TU: {global_mi_b0:.4f}")
    else:
        print("No b0 data was processed. Ensure that the dataset and processing steps are correct.")
    
    print(f"Global Mean Squared Error for FM: {global_mse_FM:.4f}")
    print(f"Global Mutual Information for FM: {global_mi_FM:.4f}")




def create_displacement_grid(vdm_slice):
    if vdm_slice.ndim != 2:
        raise ValueError(f"Expected VDM_slice to have 2 dimensions, but got {vdm_slice.ndim}")

    x, y = np.meshgrid(
        np.arange(vdm_slice.shape[0]),  
        np.arange(vdm_slice.shape[1]),
        indexing='ij'
    )

    x_displaced = x
    y_displaced = y + vdm_slice  # Apply VDM slice data to y-displacement

    return x_displaced, y_displaced

def apply_vdm_to_middle_slice(dwi_middle_slice, x_displaced, y_displaced):

    if dwi_middle_slice.shape != x_displaced.shape:
        raise ValueError(f"Shape mismatch between DWI middle slice and displacement grids: {dwi_middle_slice.shape} vs {x_displaced.shape}")

    # Apply VDM correction to the middle slice
    dwi_corrected_slice = map_coordinates(
        dwi_middle_slice, 
        [x_displaced, y_displaced], 
        order=1
    )
    
    return dwi_corrected_slice

def visualize_sample(model, dataset, device, idx):
    model.eval()
    
    # Get the sample from the dataset
    b0_imgs, t1_imgs, ref_FM_topup_imgs = dataset[idx]
    
    # Move images to the device
    b0_imgs, t1_imgs, ref_FM_topup_imgs = [img.to(device) for img in [b0_imgs, t1_imgs, ref_FM_topup_imgs]]

    # Stack b0 and t1 slices along the channel dimension to form a 6-channel input
    inputs = torch.cat((b0_imgs, t1_imgs), dim=0).unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(inputs).cpu().squeeze().numpy()

    # If the output shape doesn't match the reference FM, interpolate
    if output.shape != ref_FM_topup_imgs.shape[-2:]:
        output = F.interpolate(torch.from_numpy(output).unsqueeze(0).unsqueeze(0), size=ref_FM_topup_imgs.shape[-2:], mode='bilinear', align_corners=False)
        output = output.squeeze().numpy()

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(b0_imgs[1].cpu(), cmap='gray')
    axes[0].set_title('B0 Image Slice')
    axes[1].imshow(t1_imgs[1].cpu(), cmap='gray')
    axes[1].set_title('T1 Image Slice')
    axes[2].imshow(ref_FM_topup_imgs.cpu(), cmap='gray')
    axes[2].set_title('Reference Field Map')
    axes[3].imshow(output, cmap='gray')
    axes[3].set_title('Predicted Field Map')
    plt.savefig('/home/at90180/Documents/phd/third_session/DGA1032/take-home-exam/original/L1/image-slices_L1.png')
    plt.show()


def plot_losses(train_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    b0_folder = '/home/at90180/Documents/phd/second_session/data/project-1/firstattempt/b0_masks'
    t1_folder = '/home/at90180/Documents/phd/second_session/data/project-1/firstattempt/T1mask_after_registration'
    ref_FM_topup_file = '/home/at90180/Documents/phd/third_session/topup_forallsubject/topupresults/topup_fout_FM'
    dataset = NiftiDataset(b0_folder, t1_folder, ref_FM_topup_file)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetWithASPPSE().to(device)
    

    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)

    num_epochs = 1500

    train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    plot_losses(train_losses)

    save_test_results(model, test_loader, dataset, device)

    visualize_sample(model, test_dataset, device, idx=0)
