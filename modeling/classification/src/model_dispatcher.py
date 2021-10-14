from sklearn import tree
from sklearn import ensemble
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


models = {
    'decision_tree_gini': tree.DecisionTreeClassifier(
        criterion='gini'
    ),
    'decision_tree_entropy': tree.DecisionTreeClassifier(
        criterion='entropy'
    ),
    'rf': ensemble.RandomForestClassifier(),
    'ada_boost': ensemble.AdaBoostClassifier(),
    'gp': GaussianProcessClassifier(
        1.0 * RBF(1.0)
    ),
    'knn': KNeighborsClassifier(10),
    'svc_linear': SVC(kernel='linear', C=0.025),
    'svc_none_linear': SVC(gamma=2, C=1)
}