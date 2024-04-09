#!/usr/bin/env/python

import os
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from tqdm import tqdm

CELEBDFDIR = os.path.join(os.getcwd(), "..\\Deepfakes\\Celeb-DF-v2-faces")


class Eigenfaces_base:
    def __init__(self, celebdfdir=CELEBDFDIR, grayscale=True, spectral=False):
        self.pca = PCA()
        self.faces = []
        self.labels = []
        self.grayscale = grayscale
        self.spectral = spectral
        self.celebdfdir = celebdfdir

        self.eigenfaces = None
        self.weights = None

    def load_face(self, path) -> np.ndarray:
        """
        Load image from path and resize to 128x128. Optionally convert to grayscale or spectral domain using DCT.

        :param path: Path to image
        :param grayscale: Convert to grayscale
        :param spectral: Convert to spectral domain using DCT
        :return: Image as numpy array
        """
        img = Image.open(path)
        img = img.resize((128, 128))  # Resize to 128x128
        if self.grayscale:
            img = img.convert("L")  # Convert to grayscale
            img = np.asarray(img)
            return dct(img) if self.spectral else img
        else:
            img = np.asarray(img)
            if self.spectral:
                spectral_img = np.zeros((128, 128, 3))
                for i in range(3):
                    spectral_img[:, :, i] = dct(
                        dct(img[:, :, i], axis=0, norm="ortho"), axis=1, norm="ortho"
                    )
                return spectral_img
            return img

    def load_faces(self):
        """Load faces from Celeb-DF-v2-faces dataset from self.celebdfdir"""
        for subdir in os.listdir(self.celebdfdir):
            working_dir = os.path.join(self.celebdfdir, subdir)
            for file in os.listdir(working_dir):
                path = os.path.join(working_dir, file)
                face = self.load_face(path)
                self.faces.append(face.flatten())
                self.labels.append(0 if "real" in subdir else 1)

    def fit(self):
        """Fit PCA to self.faces"""
        raise NotImplementedError("Subclass must implement this method")

    def eval(self):
        """Evaluate model"""
        raise NotImplementedError("Subclass must implement this method")

    def plot(self):
        assert self.eigenfaces is not None, "Model has not been fitted before plotting"
        first_eigenface = self.eigenfaces[0]
        mean_eigenface = self.eigenfaces.mean(axis=0)
        sum_eigenface = self.eigenfaces.sum(axis=0)
        # Normalize to 0-255
        first_eigenface = (first_eigenface - first_eigenface.min()) / (
            first_eigenface.max() - first_eigenface.min()
        )
        mean_eigenface = (mean_eigenface - mean_eigenface.min()) / (
            mean_eigenface.max() - mean_eigenface.min()
        )
        sum_eigenface = (sum_eigenface - sum_eigenface.min()) / (sum_eigenface.max() - sum_eigenface.min())
        fig, ax = plt.subplots(1, 3)
        if self.grayscale:
            ax[0].imshow(first_eigenface.reshape(128, 128), cmap="gray")
            ax[1].imshow(mean_eigenface.reshape(128, 128), cmap="gray")
            ax[2].imshow(sum_eigenface.reshape(128, 128), cmap="gray")
        else:
            ax[0].imshow(first_eigenface.reshape(128, 128, 3))
            ax[1].imshow(mean_eigenface.reshape(128, 128, 3))
            ax[2].imshow(sum_eigenface.reshape(128, 128, 3))
        ax[0].set_title("First Eigenface")
        ax[1].set_title("Mean Eigenface")
        ax[2].set_title("Sum Eigenface")
        fig.suptitle(
            f"Eigenfaces - {'grayscale' if self.grayscale else 'RGB'} {'spectral' if self.spectral else 'image'}"
        )
        plt.show()


class Eigenfaces(Eigenfaces_base):
    def __init__(self, celebdfdir=CELEBDFDIR, grayscale=True, spectral=False):
        super().__init__(celebdfdir, grayscale, spectral)

    def fit(self, train_faces, train_labels, n_components=0):
        """
        Fit PCA to faces

        :param train_faces: Training faces
        :param n_components: Number of components to use in PCA. If 0, use all components
        """

        self.pca.fit(train_faces)
        if n_components == 0:  # Use all components if n_components is 0
            n_components = self.pca.n_components_

        assert (
            n_components <= self.pca.n_components_ and n_components > 0
        ), f"n_components must be less than or equal to the number of components in PCA ({self.pca.n_components_})"

        # Get the first n_components eigenfaces
        self.eigenfaces = self.pca.components_[:n_components]
        # Transform training data to eigenface space to get the trained weights
        # Equivalent to pca.transform(X_train) with n_components
        self.weights = self.eigenfaces @ (train_faces - self.pca.mean_).T
        self.train_labels = train_labels

    def eval(self, test_faces, test_labels) -> float:
        """
        Evaluate model using test_faces, return accuracy

        :param test_faces: Test faces
        :return: Accuracy
        """
        assert (
            self.eigenfaces is not None and self.weights is not None and self.train_labels is not None
        ), "Model has not been fitted before evaluating"

        # Transform test data to eigenface space using the n_components eigenfaces from fitting
        test_weights = self.eigenfaces @ (test_faces - self.pca.mean_).T

        # classify and calculate accuracy
        norm = euclidean_distances(self.weights.T, test_weights.T)
        matches = np.asarray(self.train_labels)[np.argmin(norm, axis=0)]
        accuracy = np.mean(matches == np.asarray(test_labels))
        return accuracy


if __name__ == "__main__":
    accuracies = [[], [], [], []]
    trials = 10
    for i in tqdm(range(trials)):
        for conf in range(4):
            g = conf < 2
            s = conf % 2 == 1
            ef = Eigenfaces(grayscale=g, spectral=s)
            ef.load_faces()
            X_train, X_test, y_train, y_test = train_test_split(ef.faces, ef.labels, test_size=0.2)
            ef.fit(X_train, y_train)
            accuracy = ef.eval(X_test, y_test)
            accuracies[conf].append(accuracy)
            print(f"Grayscale: {g}, Spectral: {s}, Accuracy: {accuracy*100:.2f}%")
            # ef.plot()
    print("=====================================")
    print(f"Avg accuracies in {trials} trials:")
    print(f"Grayscale image: {np.mean(accuracies[0])*100:.2f}%")
    print(f"Grayscale spectral: {np.mean(accuracies[1])*100:.2f}%")
    print(f"RGB image: {np.mean(accuracies[2])*100:.2f}%")
    print(f"RGB spectral: {np.mean(accuracies[3])*100:.2f}%")
