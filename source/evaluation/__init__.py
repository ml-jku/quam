from .uncertainty import *
from .ood import *
from .missclass import *

uncertainty_fns_map = {
    'uncertainty_our': calculate_uncertainty_setting_b,
    'uncertainty_gal': calculate_uncertainty_setting_a,
}

evaluation_fns_map = {
    'ood': evaluate_ood,
    'retention': evaluate_missclass,
}

filter_sample_fns_map = {
    'best': combine_sample_best,
    'last': combine_sample_last,
    'all': combine_sample_all,
    'tempered_exp': None,
    'softmax': combine_sample_softmax,
}
