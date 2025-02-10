from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import (GenericUnivariateSelect, chi2,
                                       mutual_info_classif, f_classif)
from sklearn.feature_selection import (RFE, RFECV)
from sklearn.feature_selection import (SelectFromModel)
from sklearn.feature_selection import (SequentialFeatureSelector)

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

## -------------------------------------------------------------------------------- ##
# Using SelectFromModel Feature Selection
## -------------------------------------------------------------------------------- ##

# L1-based Feature Selection

# Tree-based Feature Selection

## -------------------------------------------------------------------------------- ##
# Sequential Feature Selection
## -------------------------------------------------------------------------------- ##

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
    else:
        raise ValueError (f"No fs method with the name: {args['feat_selection']}")
