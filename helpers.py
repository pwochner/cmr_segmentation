import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
import gryds

parent_path = os.path.dirname( os.getcwd() )
data_path = os.path.join(parent_path, "labelled/")

info_filename = "M&Ms Dataset Information.xlsx"
df_info = pd.read_excel( os.path.join(parent_path, info_filename) )
df_info.columns = df_info.columns.str.replace(' ', '') # remove space in column name


def load_images(the_files, img_dim, nr_slices):

    image_array = np.zeros( (img_dim, img_dim, nr_slices) )

    idx_z = 0
    for current_file in the_files:
        
        patient_img = nib.load(current_file).get_fdata()   

        # find patient information using filename
        filename = os.path.basename(current_file)
        pos = filename.find("_")
        patient_id = filename[: pos]
        nr_slices = patient_img.shape[2]
        patient_info = df_info[df_info['Externalcode'] == patient_id]
        idx_ED = patient_info['ED']
        idx_ES = patient_info['ES']

        # take central 192x192
        start_x = int ( np.floor( (patient_img.shape[0] - img_dim)/2 ) )
        start_y = int ( np.floor( (patient_img.shape[1] - img_dim)/2 ) )

        image_array[:,:,idx_z:idx_z+nr_slices] = np.copy( np.squeeze( patient_img[start_x:start_x+img_dim , start_y:start_y+img_dim,:,idx_ED] ) )
        image_array[:,:,idx_z+nr_slices:idx_z+2*nr_slices] = np.copy( np.squeeze( patient_img[start_x:start_x+img_dim , start_y:start_y+img_dim,:,idx_ES] ) )
    

        idx_z = idx_z + 2*nr_slices

    return image_array


def data_augmentation(images, labels, how_many):

    augmented_images = np.zeros( (images.shape[0], images.shape[1], images.shape[2]*how_many ) )
    augmented_labels = np.zeros( (labels.shape[0], labels.shape[1], labels.shape[2]*how_many ) )

    for i in range(images.shape[2]) :
        for j in range(how_many):
            img_sa = images[:,:,i] 
            # normalise data
            p5 = np.percentile(img_sa,5)
            p95 = np.percentile(img_sa,95)
            img_sa = (img_sa-p5) / (p95 - p5)
            # affine transformation
            affine_transformation = gryds.AffineTransformation(
            ndim = 2,
            angles = [ np.random.uniform(-np.pi/8., np.pi/8.) ], # the rotation angle
            scaling= [np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2)],# the anisotropic scaling
            # shear_matrix=[[1, 0.5], [0, 1]], # shearing matrix
            translation = [ np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)  ],# translation
            center=[0.5, 0.5] # center of rotation
            )
            # Define a random 3x3 B-spline grid for a 2D image:
            random_grid = np.random.rand(2, 3, 3)
            random_grid -= 0.5
            random_grid /= 5
            # Define a B-spline transformation object
            bspline = gryds.BSplineTransformation(random_grid)
            # Define an interpolator object for the image:
            interpolator_sa = gryds.Interpolator(img_sa)
            interpolator_gt = gryds.Interpolator(labels[:,:,i], order=0) # img_gt

            composed_trf = gryds.ComposedTransformation(bspline, affine_transformation)

            augmented_images[:,:, i*how_many + j] = np.clip(interpolator_sa.transform(composed_trf), 0, 1) 
            augmented_labels[:,:, i*how_many + j] = interpolator_gt.transform(composed_trf)

    augmented_images = augmented_images[np.newaxis,...]
    augmented_images = np.transpose(augmented_images, (3, 0, 1, 2))
    augmented_labels = np.transpose(augmented_labels, (2, 0, 1))
  

    return augmented_images, augmented_labels