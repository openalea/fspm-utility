'''IMPORTS'''
import os, shutil, importlib
import numpy as np
from keras.models import Model, load_model
import umap
import hdbscan

import src.analysis.time_series_projection
from src.analysis.time_series_projection import Preprocessing, DCAE

'''SCRIPT'''
input_type = "mtg"

# DCAE parameters
import_model = False
train_model = not import_model
dev = True
window = 24
EPOCHS = 10
BS = 100
test_prop = 0.2
# UMAP Parameters
umap_seed = 42  # Set if you want consistency (default None)
umap_dim = 12   # Number of dimensions to reduce (max is 24)
n_neighbors = 50  # We increase the number of neighbors to prevent focusing on too local data structure (default 15)
min_dist = 0.0  # We decrease this value to densely pack clusters and ease clustering (default 0)
# HDBSCAN Parameters
min_cluster_size = 5000  # Adjust this value if the number of produced similar clusters is to high (default 500)
min_samples = 10  # Number of necessary neighbors to consider a point is a cluster core point (default None)


def run_analysis(file, output_path, extract_props, input_type=input_type, import_model=import_model, train_model=train_model, dev=dev, window=window,
                 EPOCHS=EPOCHS, BS=BS, test_prop=test_prop, umap_seed=umap_seed, umap_dim=umap_dim, n_neighbors=n_neighbors, min_dist=min_dist,
                 min_cluster_size=min_cluster_size, min_samples=min_samples):
    """
    Description
    This function runs the main workflow for

    :param file: xarray datadset containing 't', 'vid and supplementary scenario parameters

    """

    # Import and preprocess data
    print("[INFO] Preprocessing saved file...")
    preprocess = Preprocessing(central_dataset=file, type=input_type, variables=extract_props, window=window)

    # stacked_dataset = [np.array(k) for k in preprocess.stacked_dataset]

    folder = os.path.dirname(__file__)
    if import_model:
        print("[INFO] Loading autoencoder...")
        autoencoder = load_model(folder + "/saved_model/autoencoder")
    else:
        # Build the convolutional autoencoder
        print("[INFO] Building autoencoder...")
        (encoder, decoder, autoencoder) = DCAE.build(height=1, width=window, depth=len(extract_props),
                                                     filters=((64, 10, 2), (32, 5, 2), (12, 5, 3)), latentDim=window)

    if train_model:
        plotting = False
        print("[INFO] Training autoencoder...")
        autoencoder = DCAE.train(stacked_dataset=preprocess.stacked_da, autoencoder=autoencoder, test_prop=test_prop,
                                 epochs=EPOCHS, batch_size=BS, plotting=False)
        shutil.rmtree(folder + '/saved_model/autoencoder')
        os.mkdir(folder + '/saved_model/autoencoder')
        autoencoder.save(folder + '/saved_model/autoencoder')

    # use the convolutional autoencoder to predict latent layer from trained encoder only
    # in autoencoder, -2 to retrieve encoder, -1 for decoder
    trained_encoder = Model(autoencoder.input, autoencoder.layers[-2].output)
    # print(trained_encoder.summary())

    # project on latent representation for all dataset, here time windows are still ordered in the obtained array
    print("[INFO] Encoding windows...")

    latent_windows = trained_encoder.predict(preprocess.stacked_da)
    # print(latent_windows)  # number of extracter windows x number of modalities x number of organs


    # Latent space projection on lower dimension
    print("[INFO] UMAP reducer processing latent windows...")
    umap_reducer_ND = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=umap_dim, random_state=umap_seed)
    windows_ND_embedding = umap_reducer_ND.fit_transform(latent_windows)

    print("[INFO] HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, leaf_size=40, cluster_selection_method='eom',
                                metric='euclidean', allow_single_cluster=False, min_cluster_size=min_cluster_size,
                                min_samples=min_samples)

    clusterer.fit(windows_ND_embedding)

    groups = list(np.unique(clusterer.labels_))
    hdbscan_clusters = [[] for k in groups]
    for c in range(len(clusterer.labels_)):
        hdbscan_clusters[groups.index(clusterer.labels_[c])] += [c]

    print(f"[INFO] Got {len(hdbscan_clusters)} clusters from HDBSCAN")

    # Plotting projection
    if umap_dim != 3:
        plot = False
    else:
        plot = True

    # If this is user call of the analysis
    if not dev:
        if len(hdbscan_clusters) > 0:
            from src.analysis.time_series_projection import MainMenu
            main_menu = MainMenu(windows_ND_projection=windows_ND_embedding, latent_windows=latent_windows,
                                 sliced_windows=preprocess.stacked_win, original_unorm_dataset=preprocess.unormalized_ds,
                                 original_dataset=preprocess.normalized_ds, coordinates=preprocess.labels,
                                 clusters=hdbscan_clusters, window=window, plot=plot, windows_time=preprocess.t_windows,
                                 output_path=output_path)
            main_menu.build_app()
        else:
            print('[ERROR] Analysis output without cluster')

    # Using this loop to be able to implement visualization without re-running UMAP each time
    if len(hdbscan_clusters) > 0:
        while dev:
            # If re-running, reload the local plotting library
            importlib.reload(src.analysis.time_series_projection)
            from src.analysis.time_series_projection import MainMenu

            main_menu = MainMenu(windows_ND_projection=windows_ND_embedding, latent_windows=latent_windows,
                                 sliced_windows=preprocess.stacked_win, original_unorm_dataset=preprocess.unormalized_ds,
                                 original_dataset=preprocess.normalized_ds, coordinates=preprocess.labels,
                                 clusters=hdbscan_clusters, window=window, plot=plot, windows_time=preprocess.t_windows, output_path=output_path)
            main_menu.build_app()
            again = input("reimport and replot? ([Y]/n)")
            if again not in ("y", "Y", ""):
                dev = False

    else:
        print('[ERROR] Analysis output without cluster')