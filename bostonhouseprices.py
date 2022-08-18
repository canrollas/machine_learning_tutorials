from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings("ignore")

boston = load_boston()

# Printing the Description of the dataset.

print("Dataset Description:\n{}".format(boston.get("DESCR")))
print("\n\n")

# Printing the keys.

print("These are keys of dataset:{}".format(list(boston.keys())))
# Printing feature names.
print("These are keys of data:{}".format(boston.feature_names))
print("Shape of the data:{}".format(boston.data.shape))
print("{} Entry".format(boston.data.shape[0]))
print("{} Column".format(boston.data.shape[1]))






