import numpy as np
from eigen_utils import meancenter, h_eigenvekt

def recogface(inp_img, dataset_img, labels, filenames, threshold=5000, return_pca=False, num_components=100):
    centered_data, mean_face = meancenter(dataset_img)

    eigenfaces, _ = h_eigenvekt(centered_data, num_components=num_components)

    projections = np.dot(centered_data, eigenfaces)

    input_centered = inp_img - mean_face
    input_projection = np.dot(input_centered, eigenfaces)

    jarak = [np.linalg.norm(projection - input_projection) for projection in projections]
    min_index = np.argmin(jarak)
    min_jarak = jarak[min_index]

    result = (filenames[min_index], labels[min_index], min_jarak) if min_jarak < threshold else (None, None, min_jarak)

    if return_pca:
        return *result, eigenfaces, mean_face
    else:
        return result
