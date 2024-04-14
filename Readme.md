# BZA 2023/2024 project - Deepfake detection using EigenFaces and PCA

Used configuration:
- **Python 3.10**
- **OpenCV** for extracting faces from videos
- **Scikit-learn** for PCA implementation, also used for LDA, QDA and SVM classifiers
- [**Celeb-DF-v2** dataset](https://github.com/yuezunli/celeb-deepfakeforensics) custom split into train/eval subsets (see [Dataset](#dataset))

All neccessary requirements are present in `requirements.txt` file and can be easily installed with 

```pip install -r requirements.txt```

## Repository structure

- *img/* - folder with plotted images
- *acc.txt* - measured accuracies of all variants on the created train/eval split
- *extract_faces.py* - script for extracting faces from videos using OpenCV
- *eigenfaces.py* - script for fitting and evaluating eigenfaces deepfake detection on the dataset (see [Implementation](#implementation)) using scipy and scikit-learn
- *PBS_\*.sh* - scripts for running a batch job on [metacentrum](https://metavo.metacentrum.cz/)
- *Readme.md* - this readme with information
- *requirements.txt* - Python modules requirements

## Dataset

Extracted faces from **Celeb-DF-v2** using the `extract_faces.py` script, custom split into train/eval subsets are available for [download from gDrive](https://drive.google.com/file/d/1vGzPvH1nHLRtv2PoRjMMAvqmd35pCE8R/view?usp=sharing). 

⚠️ **The same [terms and conditions](https://docs.google.com/forms/d/e/1FAIpQLScoXint8ndZXyJi2Rcy4MvDHkkZLyBFKN43lTeyiG88wrG0rA/viewform) apply for this cropped subset as for the original dataset!** ⚠️

The split was produced as follows:
1. Split the data roughly 50-50 as training and evaluation
2. Make sure that identities in the evaluation data are not present in the training data. That means, that the whole evaluation subset can be considered *unseen samples*.

The training subset contains 307 real faces of celebrities and 101 real faces extracted from random YouTube videos (see the [original dataset](https://github.com/yuezunli/celeb-deepfakeforensics) for details), together making 408 real faces. As in the original dataset, there are significantly more synthetic faces - 3.313 samples.

Similarly, the evaluation subset contains 287 real celebrity faces and 204 real faces from YouTube, together 491 real faces, and 2411 synthetic faces.

## Implementation

There are two main files in the repository - `extract_faces.py` for extracting faces from videos and `eigenfaces.py` containing the logic for fitting and evaluating various eigenface models.

### Face extraction

The python script `extract_faces.py` is currently tuned for processing Celeb-DF-v2 dataset, but it should be rather easy to modify the code to work with any structure and videos. The code itself is inspired by [freearhey's github implementation](https://github.com/freearhey/face-extractor/tree/master). 

**Reccomendation:** It is highly reccomended to use CUDA-enabled OpenCV and tensorflow installations, as those are the libraries used behind the scenes to do the demanding computations. At the moment, the script is restricted to extract only faces from the first frame of the videos, which is feasible even on ordinary multi-core CPU. To surpass this restriction, simply remove the `break` statement on [line 57 on the script](https://github.com/vojteskas/EigenFaces/blob/main/extract_faces.py#L57).

The inner workings are simple: first, a video stream is open using OpenCV `VideoCapture`, which allows for reading frames on demand. `FaceDetector` from `facedetector-py` is used to detect faces in the frame and based on the bounded boxes it provides, the faces are extracted and cropped from the frame and saved as jpg files.

### Eigenfaces

There are numerous models implemented in the script. The foundation is the **Eigenfaces** class, housing PCA from sklearn, which can be fitted using the `fit` method to provided vectors, calculating principal components also known as eigenfaces. The `eval` method transforms the test vectors into the eigenface space and using the euclidean distance, matches the test vector to a one from the training vectors. True labels are compared with the predicted ones (of the real/fake classes) to calculate accuracy of correct prediction.

On top of Eigenfaces, the **EF_\*** classes employ additional classifiers in the transformed eigenface space. Those are **EF_SVM** using Support Vector Machines and **EF_DiscriminantAnalysis_Gaussian** using Gaussian classifier with Linear or Quadratic Discriminant Analysis based on the constructor parameter. By default, no parameters are modified (so for example SVM implicitely uses radial basis kernel function), but can be modified in the code.

The classes are split into base classes (*Eigenfaces_base* and *EF_base*) containing the common code. This allows for further expansion and modularization of the project. It should be very easy to extend this repository with more classifiers and do further analysis.

The `eigenfaces.py` script also contains a number of helper functions, namely:
- `load_faces` to load faces from a specific directory, expects the directory to have the structure of the custom split dataset discussed in section [Dataset](#dataset). Parameters control the following:
    - *grayscale* - whether to convert the input image to grayscale or leave it as is (presumably RGB)
    - *spectral* - whether to transform the loading images using **DCT**. Useful for experimenting in spectral domain.
    - *type* - what images to load. Considering the dataset structure, 'fake' loads only synthesized faces, 'real' only real faces and 'all' loads all faces from the dataset.

    Returned is an array of the *flattened* face/spectrum vectors and corresponding labels (0 for real, 1 for fake)
- `get_real_fake_face` samples a random fake and random real face from the provided arrays of faces and corresponding labels.
- **`fit_ef_model`** is the versatile core of the project to fit a specific configuration of a PCA model. The parameters allow for customizable setup:
    - *variant*: Model variant (Eigenfaces, EF_SVM, EF_LDA, EF_QDA)
    - *grayscale*: Convert to grayscale
    - *spectral*: Convert to spectral domain using DCT
    - *n_components*: Number of components to use in PCA. If 0, use all components
    - *type*: Type of faces to load (all, real, fake)
  Returned is the fitted model.
- `train_for_interactive` is a helper function to train all 16 currently implemented variants for experiments in interactive python shell. It returns a list of lists of trained models. The models can be accessed as follows:
    - first index: variant, i.e. 0 - Eigenfaces, 1 - EF_SVM, 2 - EF_LDA, 3 - EF_QDA
    - second index: configuration, i.e. 0 - grayscale, 1 - grayscale spectral, 2 - RGB, 3 - RGB spectral

    For example: `models[0][0]` - Eigenfaces grayscale, `models[1][4]` - EF_SVM RGB spectral
- The `main` function fits a specific variant in all configurations and evaluates accuracy on test faces.
