import matplotlib.pyplot as plt
import mglearn
import warnings
warnings.filterwarnings("ignore")


mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

mglearn.plots.plot_knn_classification(n_neighbors=6)
plt.show()