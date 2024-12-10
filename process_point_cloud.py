import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted as ns
import pandas as pd
from utilities.utils import arreglar_imagen, tilt_correction, filter_image


root_path = 'path/to/point/clouds'
result_path = 'path/to/result.csv'
point_clouds_names = ns(os.listdir(root_path))

data = np.empty((len(point_clouds_names), 5), dtype=object)

for n, name in enumerate(tqdm(point_clouds_names)):
    """"
    Se hace el siguiente preprocesado:
    1. Se corrigen los NaNs
    2. Se corrige la inclinación
    3. Se filtra la imagen lambda < 10 um
    4. Se calculan las rugosidades
    """

    "1.Empezamos leyendo la secuencia, ordenandola en una imagen y corrigiendo los valores NaN"
    img = np.genfromtxt(os.path.join(root_path, name), skip_header=14, delimiter=" ", comments="D", skip_footer=1)

    img_processed = arreglar_imagen(img)

    "2. Corregir la inclinación"
    img_corrected = tilt_correction(img_processed)
    img_corrected = img_corrected/np.max(img_corrected)

    "3. Filtrar lambda < 10 um"
    img_filtered = filter_image(img_corrected)

    "4. Calcular rugosdiades y Average step height"
    # Sa y Sz con corrección de la inclinación
    processed_surface = img_corrected[:, int(0.65 * img_corrected.shape[1]):]
    processed_surface = processed_surface - np.mean(processed_surface, axis=None)

    Sa = np.mean(np.abs(processed_surface), axis=None)
    Sz = np.abs(np.max(processed_surface) - np.min(processed_surface))

    # Sa con corrección de la inclinación y filtrada
    filtered_surface = img_filtered[:, int(0.65 * img_filtered.shape[1]):]
    filtered_surface = filtered_surface - np.mean(filtered_surface, axis=None)

    Sa_10um = np.mean(np.abs(filtered_surface), axis=None)

    reference_surface = img_corrected[:, :int(0.35 * img_corrected.shape[1])]
    processed_surface = img_corrected[:, int(0.65 * img_corrected.shape[1]):]
    average_step_height = np.abs(np.mean(reference_surface) - np.mean(processed_surface))

    data[n, :] = np.array([name, Sa, Sz, Sa_10um, average_step_height], dtype=object)

df = pd.DataFrame({'Name': data[:, 0], 'Sa': data[:, 1], 'Sz': data[:, 2], 'Sa λ<10um': data[:, 3], 'Average Step Height': data[:, 4]})
df.to_csv(result_path, index=False)

