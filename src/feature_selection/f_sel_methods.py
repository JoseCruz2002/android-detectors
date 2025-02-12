from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import (GenericUnivariateSelect, chi2,
                                       mutual_info_classif, f_classif)
from sklearn.feature_selection import (RFE, RFECV)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import (SelectFromModel)
from sklearn.feature_selection import (SequentialFeatureSelector)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier

__all__ = ["feature_selection"]

## -------------------------------------------------------------------------------- ##
# Removing Features with Low Variance
## -------------------------------------------------------------------------------- ##

def feat_selection_variance(input, input_features, p=0.8):
    """
    Perform feature selection using Variance Threshold method.
    Arguments:
        input: The encoded input before feature selection.
        input_features: The corresponding names of the input features.
        p: The value of p.
    Returns:
        (The input with only the selected features, The names of these last features)
    """
    print("Starting feature selection: Variance threshold")
    selector = VarianceThreshold(threshold=(p * (1 - p)))
    new_input = selector.fit_transform(input)
    new_input_features = selector.get_feature_names_out(input_features)
    return (new_input, new_input_features)

## -------------------------------------------------------------------------------- ##
# Univariate Feature Selection
## -------------------------------------------------------------------------------- ##

selection_function_map = {
    "chi2": chi2,
    "mutual_info_classif": mutual_info_classif,
    "f_classif": f_classif
}

def feat_selection_univariate(input, input_features, labels, selection_type,
                              selection_function, param):
    """
    Perform feature selection using Univariate methods.
    Arguments:
        input: The encoded input before feature selection.
        input_features: The corresponding names of the input features.
        labels: The corresponding labels for the input.
        selection_type: either SelectKBest or SelectPercentile.
        selection_function: the function used to compute the statisitical scores.
        param: For the chosen selection_type, the parameter, wether K or the percentile.
    Returns:
        (The input with only the selected features, The names of these last features)
    """
    print("Starting feature selection: Univariate")
    if selection_type not in ('percentile', 'k_best', 'fpr', 'fdr', 'fwe') or \
       selection_function not in ('chi2', 'mutual_info_classif', 'f_classif'):
        raise ValueError("Selection type or function were wrong!!")
    
    param = int(param) if (selection_type == "k_best") else param
    selector = GenericUnivariateSelect(selection_function_map[selection_function],
                                       mode=selection_type, param=param)
    new_input = selector.fit_transform(input, labels)
    new_input_features = selector.get_feature_names_out(input_features)
    return (new_input, new_input_features)

## -------------------------------------------------------------------------------- ##
# Recursive Feature Selection
## -------------------------------------------------------------------------------- ##

estimator_map = {
    "SVR": SVR(kernel="linear"),
}

def feat_selection_recursive(input, input_features, labels, estimator_str,
                             n_features_to_select):
    print("Starting feature selection: Recursive")
    estimator = estimator_map[estimator_str]
    selector = RFE(estimator, n_features_to_select=int(n_features_to_select))
    new_input = selector.fit_transform(input, labels)
    new_input_features = selector.get_feature_names_out(input_features)
    return (new_input, new_input_features)

def feat_selection_recursiveCV(input, input_features, labels, estimator_str,
                               min_features_to_select):
    print("Starting feature selection: Recursive with Cross-Validation")
    selector = RFECV(
        estimator=estimator_map[estimator_str],
        step=1,
        cv=StratifiedKFold(5),
        scoring="accuracy",
        min_features_to_select=int(min_features_to_select),
        n_jobs=4,
    )
    new_input = selector.fit_transform(input, labels)
    new_input_features = selector.get_feature_names_out(input_features)
    return (new_input, new_input_features)

## -------------------------------------------------------------------------------- ##
# Using SelectFromModel Feature Selection
## -------------------------------------------------------------------------------- ##

# L1-based Feature Selection

# Tree-based Feature Selection

## -------------------------------------------------------------------------------- ##
# Sequential Feature Selection
## -------------------------------------------------------------------------------- ##

def feat_selection_sequential(input, input_features, labels, estimator_str,
                              direction, n_features_to_select):
    print("Starting feature selection: Sequential")
    estimator = estimator_map[estimator_str]
    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=int(n_features_to_select),
        direction=direction
    )
    new_input = selector.fit_transform(input, labels)
    new_input_features = selector.get_feature_names_out(input_features)
    return (new_input, new_input_features)


## -------------------------------------------------------------------------------- ##
# Callable
## -------------------------------------------------------------------------------- ##

def feature_selection(input, input_features, labels, args):
    if args["feat_selection"] == "Variance":
        return feat_selection_variance(input, input_features, args["param"])
    elif args["feat_selection"] == "Univariate":
        return feat_selection_univariate(input, input_features, labels,
                                         args["selection_type"],
                                         args["selection_function"],
                                         args["param"])
    elif args["feat_selection"] == "Recursive":
        return feat_selection_recursive(input, input_features, labels,
                                        args["estimator"], args["param"])
    elif args["feat_selection"] == "RecursiveCV":
        return feat_selection_recursiveCV(input, input_features, labels,
                                        args["estimator"], args["param"])
    elif args["feat_selection"] == "Sequential":
        return feat_selection_sequential(input, input_features, labels,
                                         args["estimator"], args["direction"],
                                         args["param"])
    else:
        raise ValueError (f"No fs method with the name: {args['feat_selection']}")
