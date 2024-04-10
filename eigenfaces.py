#!/usr/bin/env/python

import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
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


def get_real_fake_face(faces, labels) -> tuple[np.ndarray, np.ndarray]:
    """
    Get a pair of real and fake faces

    :param faces: List or array of faces
    :param labels: List or array of corresponding labels
    :return: Face
    """
    real_index = np.random.choice(np.where(labels == 0)[0])
    fake_index = np.random.choice(np.where(labels == 1)[0])
    return faces[real_index], faces[fake_index]


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
        # Normalize
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

    def plot_comparison(self, real_face, fake_face, subtitle: str = ""):
        assert self.eigenfaces is not None, "Model has not been fitted before plotting"
        # real_weight = self.eigenfaces @ (real_face - self.pca.mean_)
        # fake_weight = self.eigenfaces @ (fake_face - self.pca.mean_)

        # print(f"shapes: real face {real_face.shape}, fake face {fake_face.shape}")
        # print(f"shapes: real weight {real_weight.shape}, fake weight {fake_weight.shape}")
        # print(f"shapes: eigenfaces {self.eigenfaces.shape}, pca.mean_ {self.pca.mean_.shape}")

        # real_weight = real_weight + self.pca.mean_
        # fake_weight = fake_weight + self.pca.mean_
        # real_weight = (real_weight - real_weight.min()) / (real_weight.max() - real_weight.min())
        # fake_weight = (fake_weight - fake_weight.min()) / (fake_weight.max() - fake_weight.min())

        real_face = real_face - self.pca.mean_
        fake_face = fake_face - self.pca.mean_

        plt.figure(figsize=(10, 5))
        fig, ax = plt.subplots(1, 2)
        if self.grayscale:
            ax[0].imshow(real_face.reshape(128, 128), cmap="gray")
            ax[1].imshow(fake_face.reshape(128, 128), cmap="gray")
        else:
            ax[0].imshow(real_face.reshape(128, 128, 3))
            ax[1].imshow(fake_face.reshape(128, 128, 3))
        ax[0].set_title("Real face in eigenface space")
        ax[1].set_title("Fake face in eigenface space")
        color = "grayscale" if self.grayscale else "RGB"
        space = "spectrum" if self.spectral else "image"
        fig.suptitle(f"Comparison of {subtitle} - {color} {space}")
        plt.savefig(f"comparison_{subtitle}_{color}_{space}.png")
        plt.close()


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


class EF_base:
    def __init__(self, grayscale=True, spectral=False):
        self.ef = Eigenfaces(grayscale, spectral)

    def fit(self):
        raise NotImplementedError("Subclass must implement this method")

    def eval(self):
        raise NotImplementedError("Subclass must implement this method")

    def plot(self):
        self.ef.plot()

    def plot_comparison(self, real_face, fake_face, subtitle: str = ""):
        self.ef.plot_comparison(real_face, fake_face, subtitle)


class EF_SVM(EF_base):
    def __init__(self, grayscale=True, spectral=False):
        super().__init__(grayscale, spectral)
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


class EF_DiscriminantAnalysis_Gaussian(EF_base):
    def __init__(self, grayscale=True, spectral=False, variant: Literal["lda", "qda"] = "lda"):
        super().__init__(grayscale, spectral)
        self.da = LinearDiscriminantAnalysis() if variant == "lda" else QuadraticDiscriminantAnalysis()

    def fit(self, train_faces, train_labels, n_components=0):
        self.ef.fit(train_faces, train_labels, n_components)
        self.da.fit(self.ef.weights.T, train_labels)

    def eval(self, test_faces, test_labels) -> float:
        # Transform test data to eigenface space
        test_weights = self.ef.eigenfaces @ (test_faces - self.ef.pca.mean_).T
        preds = self.da.predict(test_weights.T)
        accuracy = np.mean(preds == test_labels)
        return accuracy


def main(variant: Literal["eigenfaces", "ef_svm", "ef_lda", "ef_qda"]):
    for conf in range(4):
        g = conf < 2
        s = conf % 2 == 1

        if variant == "eigenfaces":
            ef = Eigenfaces(grayscale=g, spectral=s)
        elif variant == "ef_svm":
            ef = EF_SVM(grayscale=g, spectral=s)
        elif variant == "ef_lda":
            ef = EF_DiscriminantAnalysis_Gaussian(grayscale=g, spectral=s, variant="lda")
        elif variant == "ef_qda":
            ef = EF_DiscriminantAnalysis_Gaussian(grayscale=g, spectral=s, variant="qda")
        else:
            raise ValueError("Invalid variant")

        train_faces, train_labels = load_faces(dir=TRAIN_DIR, grayscale=g, spectral=s)
        ef.fit(train_faces, train_labels, n_components=50)
        eval_faces, eval_labels = load_faces(dir=EVAL_DIR, grayscale=g, spectral=s)
        accuracy = ef.eval(eval_faces, eval_labels)
        print(f"{'Grayscale' if g else 'RGB'} {'spectrum' if s else 'image'} accuracy: {accuracy*100:.2f}%")

        real_face, fake_face = get_real_fake_face(eval_faces, eval_labels)
        plt.figure(figsize=(10, 5))
        fig, ax = plt.subplots(1, 2)
        if g:
            ax[0].imshow(real_face.reshape(128, 128), cmap="gray")
            ax[1].imshow(fake_face.reshape(128, 128), cmap="gray")
        else:
            ax[0].imshow(real_face.reshape(128, 128, 3))
            ax[1].imshow(fake_face.reshape(128, 128, 3))
        ax[0].set_title("Real face")
        ax[1].set_title("Fake face")
        fig.suptitle(f"Real vs Fake - {'grayscale' if g else 'RGB'} {'spectrum' if s else 'image'}")
        plt.savefig(f"rf_{variant}_{'grayscale' if g else 'RGB'}_{'spectrum' if s else 'image'}.png")
        ef.plot_comparison(real_face, fake_face, subtitle=variant)
        # ef.plot()


def train_for_interactive(n_components=0):
    """
    Train all models for interactive testing and plotting. Takes a few minutes.

    Returns a list of lists of trained models. You can access the models as follows:
    - first index: variant, i.e. 0 - Eigenfaces, 1 - EF_SVM, 2 - EF_LDA, 3 - EF_QDA
    - second index: configuration, i.e. 0 - grayscale, 1 - grayscale spectral, 2 - RGB, 3 - RGB spectral
    Examples: models[0][0] - Eigenfaces grayscale, models[1][4] - EF_SVM RGB spectral

    :param n_components: Number of components to use in PCA. If 0, use all components
    :return: List of trained models - [[Eigenfaces], [EF_SVM], [EF_LDA], [EF_QDA]]
    """

    models = [[], [], [], []]
    for conf in range(4):
        g = conf < 2
        s = conf % 2 == 1
        train_faces, train_labels = load_faces(dir=TRAIN_DIR, grayscale=g, spectral=s)
        for i, variant in enumerate(["eigenfaces", "ef_svm", "ef_lda", "ef_qda"]):
            if variant == "eigenfaces":
                ef = Eigenfaces(grayscale=g, spectral=s)
            elif variant == "ef_svm":
                ef = EF_SVM(grayscale=g, spectral=s)
            elif variant == "ef_lda":
                ef = EF_DiscriminantAnalysis_Gaussian(grayscale=g, spectral=s, variant="lda")
            elif variant == "ef_qda":
                ef = EF_DiscriminantAnalysis_Gaussian(grayscale=g, spectral=s, variant="qda")
            else:
                raise ValueError("Invalid variant")

            ef.fit(train_faces, train_labels, n_components)
            models[i].append(ef)
    return models


if __name__ == "__main__":
    for variant in ["eigenfaces", "ef_svm", "ef_lda", "ef_qda"]:
        print(f"Running {variant}")
        main(variant)  # type: ignore | run all variants
        print()
