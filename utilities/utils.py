import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy import stats
import torch


def view3Dmap(file_path, title):
    # Lists to store coordinates
    X = []
    Y = []
    Z = []

    # Read the file and store X, Y, Z coordinates
    with open(file_path, 'r') as file:
        for line in tqdm(file):
            x, y, z = map(float, line.strip().split(','))
            X.append(x)
            Y.append(y)
            Z.append(z)

    x_unique = np.unique(X)
    y_unique = np.unique(Y)

    img = np.zeros((len(x_unique), len(y_unique)))

    for global_idx, y in enumerate(tqdm(Y)):
        x = X[global_idx]
        x_idx = np.where(x_unique == x)[0][0]
        y_idx = np.where(y_unique == y)[0][0]

        img[x_idx, y_idx] = Z[global_idx]

    heatmap = plt.imshow(img)
    plt.colorbar(heatmap)
    plt.title(title)
    plt.show()


def arreglar_imagen(img):
    """
    :param img: Una lista con las coordenadas de la nube de puntos
    :return: img_processed: Un array con la imagen ordenada y procesada
    """
    width = np.max(img[:, 0])
    heigth = np.max(img[:, 1])

    img_2d = np.zeros((int(heigth) + 1, int(width) + 1))
    nan_values = []
    for i, z_value in enumerate(img[:, 2]):

        x = int(img[i, 0])
        y = int(img[i, 1])

        if np.isnan(z_value):
            nan_values.append((y, x))

        img_2d[y, x] = z_value

    alto, ancho = img_2d.shape
    img_processed = img_2d.copy()

    # Recorrer la imagen
    for (y, x) in nan_values:
        # Sumar todos los vecinos y contarlos
        suma = 0
        contador = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if 0 <= y + i < alto and 0 <= x + j < ancho and not (np.isnan(img_2d[y + i, x + j])):
                    suma += img_2d[y + i, x + j]
                    contador += 1

        if contador > 0:
            img_processed[y, x] = suma // contador

    tries = 0
    while np.any(np.isnan(img_processed)) and tries < 4:
        img_processed = corregir_nan_con_vecinos(img_processed)
        tries += 1

    return img_processed


def tilt_correction_slices(img, ref_width=0.35, proc_width=0.65, min_y=0, max_y=1):
    reference_surface = img[int(min_y * img.shape[0]):int(max_y * img.shape[0]), :int(ref_width * img.shape[1])]
    processed_surface = img[int(min_y * img.shape[0]):int(max_y * img.shape[0]), int(proc_width * img.shape[1]):]
    vertical_slice = reference_surface[:, int(ref_width / 2 * reference_surface.shape[0])]
    horizontal_slice = reference_surface[int(0.5 * reference_surface.shape[1]), :]
    # processed_horizontal_slice = processed_surface[int(0.5 * processed_surface.shape[1]), :]

    # El ajuste vertical es facil
    x_vertical = np.arange(0, reference_surface.shape[0])
    pendiente, interseccion, _, _, _ = stats.linregress(x_vertical, vertical_slice)
    x_vertical = np.arange(0, img.shape[0])  # Esto se hace para extender el ajuste a toda la imagen
    y_pred_vertical = pendiente * x_vertical + interseccion

    # El ajuste horizontal: ajustamos pendiente con referencia y la intersecion con la procesada, (???, esto es mentira)
    x_horizontal = np.arange(0, reference_surface.shape[1])
    pendiente, interseccion, _, _, _ = stats.linregress(x_horizontal, horizontal_slice)
    x_horizontal = np.arange(0, img.shape[1])  # Esto se hace para extender el ajuste a toda la imagen
    y_pred_horizontal = pendiente * x_horizontal + interseccion

    resultado = img - y_pred_vertical[:, np.newaxis] - y_pred_horizontal[np.newaxis, :]
    # Normalizar entre -1 y 1
    resultado = normalize_image(resultado)

    return resultado


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()

    # Shift the image to be in the range [0, max - min]
    image = image - image_min

    # Scale the image to be in the range [0, 1]
    image = image / (image_max - image_min)

    # Scale and shift the image to be in the range [-1, 1]
    image = image * 2 - 1

    return image


def tilt_correction_plane(img):
    fs = 1 / 0.434
    reference_surface = img[:, :int(img.shape[1] * 0.35)]

    corte_horizontal = reference_surface[reference_surface.shape[0] // 2, :]
    corte_vertical = reference_surface[:, reference_surface.shape[1] // 2]

    # Calcular la pendiente en cada corte
    pendiente_horizontal, interseccion_horizontal, _, _, _ = stats.linregress(np.arange(corte_horizontal.size),
                                                                              corte_horizontal)
    pendiente_vertical, interseccion_vertical, _, _, _ = stats.linregress(np.arange(corte_vertical.size),
                                                                          corte_vertical)

    # Calcular el término constante c usando el punto medio de los cortes
    x_medio = reference_surface.shape[1] // 2
    y_medio = reference_surface.shape[0] // 2
    z_medio = reference_surface[y_medio, x_medio]
    c = z_medio - (pendiente_horizontal * x_medio + pendiente_vertical * y_medio)

    # Calcular la distancia de cada punto al plano
    distancias = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x, y = j * fs, i * fs
            z = img[i, j]
            distancia = -(pendiente_horizontal * x + pendiente_vertical * y - z + c) / np.sqrt(
                pendiente_horizontal ** 2 + pendiente_vertical ** 2 + 1)
            distancias[i, j] = distancia

    return distancias


def filter_image(img):
    fs = 1 / 0.434
    alto, ancho = img.shape

    ejef_x = np.arange(0, ancho / 2) / ancho * fs
    fft_imagen = np.fft.fft2(img)
    fft_desplazada = np.fft.fftshift(fft_imagen)
    magnitud_espectro = np.log(np.abs(fft_desplazada) + 1)

    # Kernel gaussinao
    # Crear el kernel gaussiano
    x = np.linspace(-ancho // 2, ancho // 2, ancho) / ancho * fs
    y = np.linspace(-alto // 2, alto // 2, alto) / alto * fs
    x, y = np.meshgrid(x, y)
    sigma = 0.2 / (np.sqrt(8 * np.log(2)))
    kernel_gaussiano = 1 - np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Aplicar el filtro al FFT desplazado
    fft_filtrada = fft_desplazada * kernel_gaussiano
    magnitud_espectro_filtrado = np.log(np.abs(fft_filtrada) + 1)

    # Realizar la FFT inversa
    imagen_filtrada = np.fft.ifft2(np.fft.ifftshift(fft_filtrada))
    imagen_filtrada = np.abs(imagen_filtrada)

    return imagen_filtrada


def corregir_nan_con_vecinos(imagen):
    # Crear una copia de la imagen para no modificar la original
    imagen_corregida = np.copy(imagen)

    # Obtener las dimensiones de la imagen
    alto, ancho = imagen.shape

    # Iterar sobre cada elemento de la imagen
    for i in range(alto):
        for j in range(ancho):
            # Si el elemento es NaN, calcular la media de sus vecinos
            if np.isnan(imagen[i, j]):
                # Inicializar suma y contador de vecinos válidos
                suma, contador = 0, 0

                # Iterar sobre los vecinos
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        # Verificar si el vecino está dentro de los límites de la imagen
                        if 0 <= ni < alto and 0 <= nj < ancho and not np.isnan(imagen[ni, nj]):
                            suma += imagen[ni, nj]
                            contador += 1

                # Asignar la media de los vecinos al elemento NaN
                if contador > 0:
                    imagen_corregida[i, j] = suma / contador

    return imagen_corregida


def roughness(img):
    img = img - np.mean(img, axis=None)

    Sa = np.mean(np.abs(img), axis=None)

    return Sa


def desnorm_params(params, max_values, min_values):
    log_params = params * (max_values - min_values) + min_values
    new_params = torch.tensor(log_params)
    new_params[:, 3] = 10 ** log_params[:, 3]
    new_params[:, 4] = 10 ** log_params[:, 4]
    new_params[:, 8] = 10 ** log_params[:, 8]

    return new_params
