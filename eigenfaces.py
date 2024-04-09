#!/usr/bin/env/python

import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from sklearn.svm import SVC

CELEBDFDIR = os.path.join(os.getcwd(), "..\\Deepfakes\\Celeb-DF-v2-faces")
TRAIN_DIR = os.path.join(CELEBDFDIR, "train")
EVAL_DIR = os.path.join(CELEBDFDIR, "eval")


def load_face(path, grayscale=True, spectral=False) -> np.ndarray:
    """
    Load image from path and resize to 128x128. Optionally convert to grayscale or spectral domain using DCT.

    :param path: Path to image
    :param grayscale: Convert to grayscale
    :param spectral: Convert to spectral domain using DCT
    :return: Image as numpy array
    """
    img = Image.open(path)
    img = img.resize((128, 128))  # Resize to 128x128
    if grayscale:
        img = img.convert("L")  # Convert to grayscale
        img = np.asarray(img)
        return dct(img) if spectral else img
    else:
        img = np.asarray(img)
        if spectral:
            spectral_img = np.zeros((128, 128, 3))
            for i in range(3):
                spectral_img[:, :, i] = dct(dct(img[:, :, i], axis=0, norm="ortho"), axis=1, norm="ortho")
            return spectral_img
        return img


def load_faces(dir=TRAIN_DIR, grayscale=True, spectral=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Load faces from Celeb-DF-v2-faces dataset

    :param path: Path to dataset
    """
    faces = []
    labels = []
    for subdir in os.listdir(dir):
        working_dir = os.path.join(dir, subdir)
        for file in os.listdir(working_dir):
            path = os.path.join(working_dir, file)
            face = load_face(path, grayscale, spectral)
            faces.append(face.flatten())
            labels.append(0 if "real" in subdir else 1)
    return np.array(faces), np.array(labels)


class Eigenfaces_base:
    def __init__(self, grayscale=True, spectral=False):
        self.pca = PCA()
        self.grayscale = grayscale
        self.spectral = spectral

        self.eigenfaces = None
        self.weights = None

    def fit(self):
        """Fit PCA to self.faces"""
        raise NotImplementedError("Subclass must implement this method")

    def eval(self):
        """Evaluate model"""
        raise NotImplementedError("Subclass must implement this method")

    def plot(self):
        assert self.eigenfaces is not None, "Model has not been fitted before plotting"
        pca_mean = self.pca.mean_
        first_eigenface = self.eigenfaces[0]
        mean_eigenface = self.eigenfaces.mean(axis=0)
        sum_eigenface = self.eigenfaces.sum(axis=0)
        # Normalize to 0-255
        pca_mean = (pca_mean - pca_mean.min()) / (pca_mean.max() - pca_mean.min())
        first_eigenface = (first_eigenface - first_eigenface.min()) / (
            first_eigenface.max() - first_eigenface.min()
        )
        mean_eigenface = (mean_eigenface - mean_eigenface.min()) / (
            mean_eigenface.max() - mean_eigenface.min()
        )
        sum_eigenface = (sum_eigenface - sum_eigenface.min()) / (sum_eigenface.max() - sum_eigenface.min())
        fig, ax = plt.subplots(1, 4)
        if self.grayscale:
            ax[0].imshow(first_eigenface.reshape(128, 128), cmap="gray")
            ax[1].imshow(mean_eigenface.reshape(128, 128), cmap="gray")
            ax[2].imshow(sum_eigenface.reshape(128, 128), cmap="gray")
            ax[3].imshow(pca_mean.reshape(128, 128), cmap="gray")
        else:
            ax[0].imshow(first_eigenface.reshape(128, 128, 3))
            ax[1].imshow(mean_eigenface.reshape(128, 128, 3))
            ax[2].imshow(sum_eigenface.reshape(128, 128, 3))
            ax[3].imshow(pca_mean.reshape(128, 128, 3))
        ax[0].set_title("First Eigenface")
        ax[1].set_title("Mean Eigenface")
        ax[2].set_title("Sum Eigenface")
        ax[3].set_title("PCA Mean")
        fig.suptitle(
            f"Eigenfaces - {'grayscale' if self.grayscale else 'RGB'} {'spectrum' if self.spectral else 'image'}"
        )
        plt.show()


class Eigenfaces(Eigenfaces_base):
    def __init__(self, grayscale=True, spectral=False):
        super().__init__(grayscale, spectral)

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


class EF_SVM:
    def __init__(self, grayscale=True, spectral=False):
        self.ef = Eigenfaces(grayscale, spectral)
        self.clf = SVC()

    def fit(self, train_faces, train_labels, n_components=0):
        self.ef.fit(train_faces, train_labels, n_components)
        self.clf.fit(self.ef.weights.T, train_labels)

    def eval(self, test_faces, test_labels) -> float:
        # Transform test data to eigenface space
        test_weights = self.ef.eigenfaces @ (test_faces - self.ef.pca.mean_).T
        preds = self.clf.predict(test_weights.T)
        accuracy = np.mean(preds == test_labels)
        return accuracy


def main(variant: Literal["eigenfaces", "ef_svm"]):
    for conf in range(4):
        g = conf < 2
        s = conf % 2 == 1

        if variant == "eigenfaces":
            ef = Eigenfaces(grayscale=g, spectral=s)
        elif variant == "ef_svm":
            ef = EF_SVM(grayscale=g, spectral=s)
        else:
            raise ValueError("Invalid variant")

        train_faces, train_labels = load_faces(dir=TRAIN_DIR)
        ef.fit(train_faces, train_labels)
        eval_faces, eval_labels = load_faces(dir=EVAL_DIR)
        accuracy = ef.eval(eval_faces, eval_labels)
        print(
            f"{'Grayscale' if g else 'RGB'} {'spectrum' if s else 'image'} accuracy: {accuracy*100:.2f}%"
        )
        # ef.plot()


if __name__ == "__main__":
    print("Eigenfaces:")
    main("eigenfaces")
    print("=====================================")
    print("EF_SVM:")
    main("ef_svm")
