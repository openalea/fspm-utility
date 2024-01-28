'''
Special requirements instructions:
__________________________________
conda install -c conda-forge xarray dask netCDF4 bottleneck
install matplotlib before tensorflow for compatibility issues
pandas, pickle
python -m pip install scikit-learn
conda install -c conda-forge umap-learn
python -m pip install tensorflow==2.12.0
conda install -c conda-forge hdbscan
'''

'''IMPORTS'''
# Data processing packages
import pandas as pd
import numpy as np
# Visual packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import CenteredNorm
import tkinter as tk
import xarray as xr

# Tensor management packages
from tensorflow import keras, reshape
from keras.layers import MaxPool2D, Conv2D, Conv2DTranspose, ReLU, UpSampling2D, Activation, Flatten, Dense, Reshape, Input
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Projection
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import f_oneway
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from src.tools_output import plot_xr

'''FUNCTIONS'''


class Preprocessing:
    def __init__(self, central_dataset, type='csv', variables={}, window=24, stride=12):
        self.unormalized_ds = central_dataset[list(variables.keys())]
        del central_dataset

        self.normalized_ds = self.normalization(self.unormalized_ds)

        # stacking to put every sliced window on the same learning slope then
        n_windows = int(1 + ((max(self.normalized_ds.coords["t"].values) - window + 1) / stride))
        times_in_window = [k * stride for k in range(n_windows)]

        # Slicing the dataset into windows
        roller = self.normalized_ds.rolling(dim=dict(t=window), center=True)
        rolled_ds = roller.construct(window_dim={"t": "window_time"}, stride=stride)
        # stacking to put every sliced window on the same learning slope then
        rolled_ds = rolled_ds.stack(window_id=[dim for dim in rolled_ds.dims if dim not in "window_time"]).fillna(0)

        self.labels, self.t_windows = [], []
        for group in rolled_ds.to_array().transpose("window_id", "t", "variable", "window_time"):
            self.t_windows += [group.t]
            self.labels += [np.array([group.vid for k in range(len(group.t))])]

        self.t_windows = list(np.concatenate(self.t_windows))
        self.labels = list(np.concatenate(self.labels))

        depth = len(variables)
        self.stacked_win = [reshape(win, shape=(1, window, depth)) for win in np.concatenate(rolled_ds.to_array().transpose("window_id", "t", "variable", "window_time"))]

        # If some windows are empty, they are thrown away from the training dataset.

        deleted = 0
        for index in range(len(self.stacked_win)):
            i = index - deleted
            test = self.stacked_win[i] == 0
            if False not in np.unique(test):
                # Then everything is null
                del self.stacked_win[i]
                del self.t_windows[i]
                del self.labels[i]
                deleted += 1

        # Convert to array as it is expected by the DCAE
        self.stacked_da = np.array(self.stacked_win)
        self.unormalized_ds = self.unormalized_ds.fillna(0.)

    def normalization(self, dataset):
        """
        Standard normalization technique
        NOTE : per organ normalization was fucking up the relative magnitude of the different organ comparison
        Now, the standardization is operated from min and max for all t, vid and scenario parameter.
        Still it remains essential to be able to compare the magnitude of the differences between clusters
        """

        return (dataset - dataset.min()) / (dataset.max() - dataset.min())


class DCAE:
    @staticmethod
    def build(height=1, width=60, depth=6, filters=((64, 10, 2), (32, 5, 2), (12, 5, 3)), latentDim=60):
        # initialize the input shape to be "channels last" along with
        # the channels dimension itself
        # channels dimension itself
        inputShape = (height, width, depth)

        # define the input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs
        # loop over the number of filters
        for f, s, p in filters:
            # apply a CONV => RELU => BN operation
            # Limites Relu non dérivable donc approximation limitée pour faible gradient, voir leaky relu
            x = Conv2D(f, (1, s), strides=1, padding="same")(x)
            x = ReLU()(x)
            x = MaxPool2D(pool_size=(1, p))(x)
        # flatten the network and then construct our latent vector
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim, activation='linear')(x)
        # build the encoder model
        encoder = Model(inputs, latent, name="encoder")

        # start building the decoder model which will accept the
        # output of the encoder as its inputs
        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        # loop over our number of filters again, but this time in
        # reverse order
        for f, s, p in filters[::-1]:
            # apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (1, s), strides=1, padding="same")(x)
            x = ReLU()(x)
            x = UpSampling2D(size=(1, p))(x)

        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        x = Conv2DTranspose(depth, (1, filters[0][0]), padding="same")(x)
        outputs = Activation("linear")(x)

        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")
        # our autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)),
                            name="autoencoder")

        # return a 3-tuple of the encoder, decoder, and autoencoder
        return (encoder, decoder, autoencoder)

    @staticmethod
    def train(stacked_dataset, autoencoder, test_prop=0.2, epochs=25, batch_size=100, plotting=False):
        trainX, testX = train_test_split(stacked_dataset, test_size=test_prop)
        opt = Adam(learning_rate=1e-3)
        autoencoder.compile(loss="mse", optimizer=opt)
        # train the convolutional autoencoder
        H = autoencoder.fit(
            trainX, trainX,
            validation_data=(testX, testX),
            epochs=epochs,
            batch_size=batch_size)

        if plotting:
            # construct a plot that plots and saves the training history
            N = np.arange(0, epochs)
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, H.history["loss"], label="train_loss")
            plt.plot(N, H.history["val_loss"], label="val_loss")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            plt.show()

        return autoencoder


class MainMenu:
    def __init__(self, windows_ND_projection, latent_windows, sliced_windows, original_unorm_dataset, original_dataset, coordinates, clusters, windows_time, window=60, plot=False, output_path=""):
        # Retrieving necessary dataset
        self.original_unorm_dataset = original_unorm_dataset
        self.original_dataset = original_dataset
        self.sliced_windows = np.array(sliced_windows)
        self.latent_windows = latent_windows
        self.windows_ND_projection = windows_ND_projection

        self.properties = list(self.original_unorm_dataset.keys())
        self.coordinates = coordinates
        self.window = window
        self.windows_time = windows_time
        self.vid_numbers = np.unique([index for index in self.coordinates])
        #self.sensitivity_coordinates = [index[1:] for index in self.coordinates]

        # Tk init
        self.root = tk.Tk()
        self.root.title('ND UMAP projection of extracted windows')
        self.output_path = output_path

        # Vid Listbox widget
        self.lb = tk.Listbox(self.root)
        scrollbar = tk.Scrollbar(self.root)
        for k in range(len(self.vid_numbers)):
            self.lb.insert(k, str(self.vid_numbers[k]))
        self.lb.grid(row=1, column=2, sticky='N')
        scrollbar.grid(row=1, column=1, sticky='NS')
        self.lb.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.lb.yview)

        # Variables Listbox widget
        self.lb2 = tk.Listbox(self.root)
        for k in range(len(self.properties)):
            self.lb2.insert(k, str(self.properties[k]))
        self.lb2.grid(row=1, column=3, sticky='N')

        # Buttons widget
        plot_button = tk.Button(self.root, text='Topo slice', command=self.flat_plot_instance)
        plot_button.grid(row=2, column=2, sticky='N')

        info_button = tk.Button(self.root, text='Clusters info', command=self.cluster_info)
        info_button.grid(row=3, column=4)

        svm_button = tk.Button(self.root, text='SVM comparison', command=self.svm_selection)
        svm_button.grid(row=2, column=4)

        prop_button = tk.Button(self.root, text='cluster contrib', command=self.cluster_contribution_proportion)
        prop_button.grid(row=2, column=3)

        # Label widget
        label = tk.Label(self.root, text='Organ ID :')
        label.grid(row=0, column=2, sticky='S')

        self.multiply_struct = tk.IntVar()
        check1 = tk.Checkbutton(self.root, text="* struct_mass", variable=self.multiply_struct, onvalue=1, offvalue=0)
        check1.grid(row=3, column=3, sticky='N')

        self.divide_struct = tk.IntVar()
        check2 = tk.Checkbutton(self.root, text="/ struct_mass", variable=self.divide_struct, onvalue=1, offvalue=0)
        check2.grid(row=4, column=3, sticky='N')

        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=10)
        self.root.rowconfigure(3, weight=1)
        # To ensure regular column spacing after graph
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.columnconfigure(3, weight=1)
        self.root.columnconfigure(4, weight=1)

        self.label = []
        self.clusters = clusters

    def svm_selection(self):
        print("[INFO] Testing clusters significativity...")
        classes = []
        selected_groups = []
        if len(self.clusters) == 0:
            print("[Error] : No selection")
        elif len(self.clusters) == 1:
            self.clusters += [[k for k in range(len(self.windows_ND_projection)) if k not in self.clusters[0]]]
            self.update_colors()
        # Retrieving latent windows corresponding to selected groups
        for k in range(len(self.clusters)):
            classes += [k for j in range(len(self.clusters[k]))]
            selected_groups += [list(i) for i in self.latent_windows[self.clusters[k]]]

        if len(self.clusters) > 1:
            # splitting data for svm training and test
            x_train, x_test, y_train, y_test = train_test_split(selected_groups, classes, test_size=0.2)
            clf = SVC(kernel='linear', C=100)
            # Here we use all data because we just perform analysis of model performance at segmentation,
            # we don't want it to be predictive
            clf.fit(selected_groups, classes)
            result = clf.predict(x_test)
            # Evaluating the accuracy of the model using the sklearn functions
            accuracy = accuracy_score(y_test, result) * 100
            confusion_mat = confusion_matrix(y_test, result)

            # Printing the results
            print("Accuracy for SVM is:", accuracy)
            print("Confusion Matrix")
            print(confusion_mat)
        else:
            print("Only one class")

    def compute_group_area_between_curves(self):
        # Check individual variables contributions to differences between clusters for each labelled cluster
        # Back to original data, grouping separated variable windows to use indexes selected by clusters

        clusters_windows = [self.sliced_windows[np.array(cluster)] for cluster in self.clusters]

        # Matrix to present main responsible for divergence between clusters through the Area Under the Curve (AUC)
        abcs = [[{} for k in range(len(self.clusters))] for i in range(len(self.clusters))]
        mean_diff_bcs = [[{} for k in range(len(self.clusters))] for i in range(len(self.clusters))]

        # for each row
        for k in range(len(self.clusters)):
            # for each element after diagonal in row
            for l in range(k+1, len(self.clusters)):
                # For a given variable, get the differences of means at each time-step
                # Then sum this to compute Area Under the curve for each variable and add it in the cross comparison matrix
                # for each variable, label the sum to a name (dict key) for readability
                # Axis 0 to mean among each window
                step_wise_differences = np.mean(clusters_windows[k], axis=0) - np.mean(clusters_windows[l], axis=0)

                # Axis 0 became for windows axis now among which we want to sum or mean now
                # Here we took indice [0] because the previous step output was shape (1, window, len(props))
                abcs[k][l] = dict(zip(self.properties, np.sum(np.abs(step_wise_differences[0]), axis=0)))
                mean_diff_bcs[k][l] = dict(zip(self.properties, np.mean(step_wise_differences[0], axis=0) * self.window))

        return abcs, mean_diff_bcs

    def cluster_info(self):
        abcs, mean_diff_between_clusters = self.compute_group_area_between_curves()

        print("[INFO] Plotting clusters")
        fig3 = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, len(self.clusters), height_ratios=[1, 2], figure=fig3)

        self.ax30 = [fig3.add_subplot(gs[0, k]) for k in range(len(self.clusters))]
        ax31 = fig3.add_subplot(gs[1, :])

        fig3.text(0.01, 0.95, "Space-Time repartition", fontweight='bold')
        fig3.text(0.01, 0.50, "window ABC between clusters", fontweight='bold')

        heatmap = []
        heatmap_values = []
        pair_labels = []

        # for each cluster combination
        for k in range(len(self.clusters)):
            times = [self.windows_time[index] for index in self.clusters[k]]
            # WARNING, here the 0 indice heavily depends on the way coordinates from xarray have been formated
            coords = [self.coordinates[index] for index in self.clusters[k]]
            unique_vids = np.unique(coords)
            maxs_index = [k for k, v in sorted(dict(zip(unique_vids, [coords.count(k) for k in unique_vids])).items(),
                                               key=lambda item: item[1], reverse=True)]

            self.ax30[k].set_title("C" + str(k) + " : " + str(int(len(self.clusters[k])/1000)) + "k / " + str(maxs_index[1:4])[1:-1])
            self.ax30[k].hist2d(times, coords, bins=10, cmap="Purples")

            for i in range(k+1, len(self.clusters)):
                heatmap += [list(abcs[k][i].values())]
                heatmap_values += [list(mean_diff_between_clusters[k][i].values())]
            pair_labels += ["{}-{}".format(k, i) for i in range(k + 1, len(self.clusters))]

        hm = ax31.imshow(np.transpose(heatmap), cmap="Greens", aspect="auto", vmin=0)
        fig3.colorbar(hm, orientation='horizontal', location='top')

        ax31.set_xticks(np.arange(len(pair_labels)), labels=pair_labels)
        ax31.set_yticks(np.arange(len(self.properties)), labels=self.properties)

        # Loop over data dimensions and create text annotations.
        for i in range(len(pair_labels)):
            for j in range(len(self.properties)):
                ax31.text(i, j, round(heatmap_values[i][j], 2), ha="center", va="center", color="w",
                               fontsize=10, fontweight='bold')

        fig3.set_size_inches(19, 10)
        fig3.savefig(self.output_path + "/clustering.png", dpi=400)
        fig3.show()

    def flat_plot_instance(self):
        layer = self.lb.curselection()
        if len(layer) > 0:
            layer = self.vid_numbers[layer[0]]
            plot_xr(datasets=self.original_unorm_dataset, vertice=[layer], selection=self.properties)

    def cluster_sensitivity_test(self, alpha=0.05):
        print("[INFO] Testing sensitivity to different scenarios")
        # Starting with multivariate anova assuming normality
        # Dataframe formating...
        classes = []
        selected_groups = []
        # Tuple is necessary here because this call is "Frozen"
        sensi_names = tuple(dim for dim in self.original_unorm_dataset.dims.keys() if dim not in ("t", "vid"))
        for c in range(len(self.clusters)):
            classes += [str(c) for j in range(len(self.clusters[c]))]
            selected_groups += [self.sensitivity_coordinates[k] for k in self.clusters[c]]

        cluster_sensi_values = pd.DataFrame(data=selected_groups, columns=sensi_names)
        cluster_sensi_values['cluster'] = classes

        # MANOVA for sensitivity factors across the cluster factor
        if len(sensi_names) > 1:
            # MANOVA for sensitivity factors across the cluster factor
            sensi_sum = ""
            for name in sensi_names:
                sensi_sum += f"{name} + "
            sensi_sum = sensi_sum[:-3]  # just to remove the + sign
            fit = MANOVA.from_formula(f'{sensi_sum} ~ cluster', data=cluster_sensi_values)
            manova_df = pd.DataFrame((fit.mv_test().results['cluster']['stat']))
            manova_pv = float(manova_df.loc[["Wilks' lambda"]]["Pr > F"])
        else:
            manova_pv = 0

        # If there is a significant difference between clusters regarding sensitivity variables...
        if manova_pv < alpha:
            # Perform pairwise tukey post-hoc test to identify which clusters are different
            meandiff_line = []
            significativity = []
            for sensi in sensi_names:
                tuckey_test = pairwise_tukeyhsd(cluster_sensi_values[sensi], cluster_sensi_values['cluster'], alpha=alpha).summary().data
                column_names = tuckey_test[0]
                pairwise_label = [line[column_names.index('group1')] + '-' + line[column_names.index('group2')] for line in tuckey_test[1:]]
                meandiff_line += [[line[column_names.index('meandiff')] for line in tuckey_test[1:]]]
                significativity += [[str(line[column_names.index('reject')]) for line in tuckey_test[1:]]]

            fig_tuckey, ax = plt.subplots()
            ax.set_xticks(np.arange(len(pairwise_label)), labels=pairwise_label)
            ax.set_yticks(np.arange(len(sensi_names)), labels=sensi_names)
            shifted_colormap = CenteredNorm()
            hm = ax.imshow(meandiff_line, cmap="PiYG", aspect="auto", norm=shifted_colormap)
            fig_tuckey.colorbar(hm, orientation='horizontal', location='top')
            # Loop over data dimensions and create text annotations.
            for i in range(len(sensi_names)):
                for j in range(len(pairwise_label)):
                    ax.text(j, i, significativity[i][j], ha="center", va="center", color="b",
                              fontsize=10, fontweight='bold')
            fig_tuckey.set_size_inches(19, 10)
            fig_tuckey.savefig(self.output_path + "/pairwise_tucker.png", dpi=400)
            fig_tuckey.show()

    def cluster_contribution_proportion(self):
        print("[INFO] Plot building pending...")
        variables = [self.lb2.get(self.lb2.curselection()[0])]

        fig_prop, ax = plt.subplots()
        label = 0
        len_simu = max(self.original_unorm_dataset.coords["t"])
        for cluster in self.clusters:
            # Retreive values from pre computed histogram
            times = [np.arange(self.windows_time[index], self.windows_time[index] + self.window, step=1) for index in cluster if self.windows_time[index] < len_simu-self.window/2]
            coords = [[self.coordinates[index]] * self.window for index in cluster if self.windows_time[index] < len_simu-self.window/2]
            #times = [self.windows_time[index] for index in cluster]
            #coords = [self.coordinates[index] for index in cluster]
            indexes = list(set(tuple(zip(np.concatenate(coords), np.concatenate(times)))))
            # indexes = list(set(tuple(zip(coords, times))))

            prop_tot = self.original_unorm_dataset.squeeze([dim for dim in self.original_unorm_dataset.dims if dim not in ("t", "vid")]).stack(in_clust=["vid", "t"])

            prop_cluster = prop_tot.sel(in_clust=indexes).unstack(dim="in_clust").sortby("t")
            prop_tot = prop_tot.unstack(dim="in_clust").sortby("t")

            title = "relative contribution of " + variables[0] + ' : ' + getattr(prop_cluster, variables[0]).attrs["unit"]
            for prop in variables:
                if self.divide_struct.get() == 1:
                    prop_ds = (getattr(prop_cluster, prop) / prop_cluster.struct_mass).mean(dim="vid") / (
                                getattr(prop_tot, prop) / prop_tot.struct_mass).mean(dim="vid")
                    title += ".g-1"
                elif self.multiply_struct.get() == 1:
                    prop_ds = (getattr(prop_cluster, prop) * prop_cluster.struct_mass).sum(dim="vid") / (
                                getattr(prop_tot, prop) * prop_tot.struct_mass).sum(dim="vid")
                    title += ".g"
                else:
                    prop_ds = getattr(prop_cluster, prop).sum(dim="vid") / getattr(self.original_unorm_dataset, prop).sum(dim="vid")

                prop_ds.plot.line(x="t", ax=ax, label="cluster " + str(label))
            label += 1
        ax.set_title(title)
        ax.legend()
        fig_prop.show()
        print("Done")

    def build_app(self):
        if len(self.clusters) > 1:
            print(f"[INFO] Comparing clusters...")
            self.cluster_info()
            # TODO : fix latter as it echoed bug after switching to the coupling between Root-CyNAPS and Rhizodep
            # self.cluster_sensitivity_test()
        self.root.mainloop()
