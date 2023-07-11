import os
import git

ROOT = os.path.abspath(os.path.dirname(__file__))


try:
    repo = git.Repo(path=ROOT, search_parent_directories=True)
    GIT_CURRENT_HEAD_SHA = repo.head.object.hexsha
    repo.close()
    del(repo)
except:
    GIT_CURRENT_HEAD_SHA = "NA"


# Dataset Paths
MNIST_PATH = os.path.join(ROOT, "..", "datasets")
CIFAR_PATH = os.path.join(ROOT, "..", "datasets")

# SET THE PATH TO THE IMAGENET AND PROJECTED IMAGENET BEFORE USE!
IMAGENET_AO_PATH = os.path.abspath('./')

IMAGENET_PATH = os.path.abspath('./')
PROJECTED_IMAGENET_PATH = os.path.abspath('./')

# Verbosity Settings
# SILENT < PROGRESS < SCORES < INTERMEDIATES
VERBOSITY_SETTING_SILENT = 0
VERBOSITY_SETTING_PROGRESS = 8
VERBOSITY_SETTING_SCORES = 16
VERBOSITY_SETTING_INTERMEDIATES = 32
