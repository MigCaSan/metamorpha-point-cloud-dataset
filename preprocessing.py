import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted as ns
import pandas as pd
from utilities.utils import arreglar_imagen, tilt_correction, filter_image


# Define the root path where the point clouds are stored and the path for the resulting CSV file
root_path = 'path/to/point/clouds'
result_path = 'path/to/result.csv'

# List all point cloud files in the root directory
point_clouds_names = ns(os.listdir(root_path))

# Initialize an empty numpy array to store results
data = np.empty((len(point_clouds_names), 5), dtype=object)

# Iterate over all point cloud files
for n, name in enumerate(tqdm(point_clouds_names)):
    """
    The following preprocessing steps are performed:
    1. Correct NaNs in the dataset
    2. Perform tilt correction
    3. Filter the image for λ < 10 µm
    4. Calculate roughness values and average step height
    """

    # Step 1: Read the data, arrange it as an image, and correct NaN values
    img = np.genfromtxt(os.path.join(root_path, name), skip_header=14, delimiter=" ", comments="D", skip_footer=1)
    img_processed = arreglar_imagen(img)  # Custom function to fix the image

    # Step 2: Correct the tilt in the image
    img_corrected = tilt_correction(img_processed)  # Custom function for tilt correction
    img_corrected = img_corrected / np.max(img_corrected)

    # Step 3: Apply filtering for λ < 10 µm
    img_filtered = filter_image(img_corrected)  # Custom function for filtering

    # Step 4: Calculate roughness and Average Step Height
    # Calculate Sa and Sz (roughness parameters) on the tilt-corrected surface
    processed_surface = img_corrected[:, int(0.65 * img_corrected.shape[1]):]
    processed_surface = processed_surface - np.mean(processed_surface, axis=None)

    Sa = np.mean(np.abs(processed_surface), axis=None)  # Arithmetic mean of the absolute deviations
    Sz = np.abs(np.max(processed_surface) - np.min(processed_surface))  # Peak-to-valley height

    # Calculate Sa λ<10 µm (filtered surface)
    filtered_surface = img_filtered[:, int(0.65 * img_filtered.shape[1]):]
    filtered_surface = filtered_surface - np.mean(filtered_surface, axis=None)
    Sa_10um = np.mean(np.abs(filtered_surface), axis=None)

    # Calculate the Average Step Height (difference between reference and processed surfaces)
    reference_surface = img_corrected[:, :int(0.35 * img_corrected.shape[1])]
    processed_surface = img_corrected[:, int(0.65 * img_corrected.shape[1]):]
    average_step_height = np.abs(np.mean(reference_surface) - np.mean(processed_surface))

    # Store the calculated values in the results array
    data[n, :] = np.array([name, Sa, Sz, Sa_10um, average_step_height], dtype=object)

# Convert the results array into a DataFrame and save it as a CSV file
df = pd.DataFrame({
    'Name': data[:, 0], 
    'Sa': data[:, 1], 
    'Sz': data[:, 2], 
    'Sa λ<10um': data[:, 3], 
    'Average Step Height': data[:, 4]
})
df.to_csv(result_path, index=False)
