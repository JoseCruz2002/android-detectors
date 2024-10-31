from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
import models
from models import DREBIN
import os

model_base_path = os.path.join(os.path.dirname(models.__file__), "../..")
base_path = os.path.join(os.path.dirname(__file__))

clf_path = os.path.join(model_base_path, "pretrained/drebin_classifier.pkl")
vect_path = os.path.join(model_base_path, "pretrained/drebin_vectorizer.pkl")

# Assuming `drebin_model` is your trained LinearSVC model
classifier = DREBIN.load(vect_path, clf_path)
classifier = classifier.get_svc_model()
classifier.classes_ = np.array([0, 1])
# drebin_model.fit(X_train, y_train)  # Ensure it’s already trained

# Define the initial type for the model input
# Replace `num_features` with the actual number of features in your data
num_features = 1461078  # for example purposes
initial_type = [("input", FloatTensorType([None, num_features]))]
#options={id(classifier): {'nocl': True}}
# Convert to ONNX format
onnx_model = convert_sklearn(classifier, initial_types=initial_type)

# Save the ONNX model
with open(os.path.join(model_base_path, "pretrained/drebin_classifier.onnx"), "wb") as f:
    f.write(onnx_model.SerializeToString())