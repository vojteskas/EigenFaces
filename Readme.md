# BZA 2023/2024 project - Deepfake detection using EigenFaces and PCA

Used configuration:
- **Python 3.10**
- **OpenCV** for extracting faces from videos
- **Scikit-learn** for PCA implementation, also used for LDA, QDA and SVM classifiers
- [**Celeb-DF-v2** dataset](https://github.com/yuezunli/celeb-deepfakeforensics) custom split into train/eval subsets (see [Dataset](#dataset))

## Repository structure

- *img/* - folder with plotted images
- *acc.txt* - measured accuracies of all variants on the created train/eval split
- *extract_faces.py* - script for extracting faces from videos
- *eigenfaces.py* - script for fitting and evaluating eigenfaces deepfake detection on the dataset (see [Implementation](#implementation))
- *PBS_\*.sh* - scripts for running a batch job on [metacentrum](https://metavo.metacentrum.cz/)
- *Readme.md* - this readme with information
- *requirements.txt* - Python modules requirements, use with `pip install -r requirements.txt`

## Dataset

Extracted faces from **Celeb-DF-v2** using the `extract_faces.py` script, custom split into train/eval subsets are available for [download from gDrive](https://drive.google.com/file/d/1vGzPvH1nHLRtv2PoRjMMAvqmd35pCE8R/view?usp=sharing).

The split was produced as follows:
1. Split the data roughly 50-50 as training and evaluation
2. Make sure that identities in the evaluation data are not present in the training data. That means, that the whole evaluation subset can be considered *unseen samples*.

The training subset contains 307 real faces of celebrities and 101 real faces extracted from random YouTube videos (see the [original dataset](https://github.com/yuezunli/celeb-deepfakeforensics) for details), together making 408 real faces. As in the original dataset, there are significantly more synthetic faces - 3.313 samples.

Similarly, the evaluation subset contains 287 real celebrity faces and 204 real faces from YouTube, together 491 real faces, and 2411 synthetic faces.

## Implementation
