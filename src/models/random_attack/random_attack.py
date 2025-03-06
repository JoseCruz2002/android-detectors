import os
import numpy
import itertools
import random

class Feature:
    def __init__(self, add, remove):
        self.add = add
        self.remove = remove


DREBIN_FEATURES = {
    "req_permissions": Feature(True, False),
    "activities": Feature(True, True),
    "services": Feature(True, True),
    "providers": Feature(True, True),
    "receivers": Feature(True, True),
    "features": Feature(True, False),
    "intent_filters": Feature(True, False),
    "used_permissions": Feature(True, False),
    "api_calls": Feature(True, True),
    "suspicious_calls": Feature(True, True),
    "urls": Feature(True, True)
}

class RandomAttack:

    def __init__(self):
        return

    def RS_sample_modification(self, sample, noise, input_features):
        if len(sample) == 0:
            print("Sample's length is zero.")
            return sample
        n_modifications = 0
        while n_modifications != noise:
            decision = random.randint(1, 2)
            if decision == 1:
                # Remove a feature
                idx = random.randint(0, len(sample)-1)
                feature = sample[idx]
                removable = DREBIN_FEATURES[feature.split("::")[0]].remove
                if removable:
                    sample.remove(feature)
                    n_modifications += 1
            else:
                # Add a feature
                idx = random.randint(0, len(input_features)-1)
                feature = input_features[idx]
                addable = DREBIN_FEATURES[feature.split("::")[0]].add
                if feature not in sample and addable:
                    sample.append(feature)
                    n_modifications += 1
        return sample