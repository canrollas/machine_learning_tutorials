import numpy as np
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings("ignore")

cancer = load_breast_cancer()

# Printing the Keys of the cancer.
print("Cancer dataset keys:{}".format(list(cancer.keys())))
# Printing the description
print("Description of the dataset:{}\n".format(cancer.get("DESCR")))
print("\n\n")

# Printing the shape
print("Cancer data shape:{}".format(cancer.get("data").shape))
print("Cancer data {} entry".format(cancer.get("data").shape[0]))
print("Cancer data {} column".format(cancer.get("data").shape[1]))
print("\n\n")

# What is Target ? Target is basically malignant and bening.
# Malignant => Kötü huylu
# Bening => İyi Huylu
print("Sample count per class:\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("\n\n")
# Printing Feature names.
print("Feature names:\n{}".format(cancer.feature_names))