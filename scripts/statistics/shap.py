import matplotlib.pyplot as plt
from scripts import settings
import xgboost as xgb
import pandas as pd
import matplotlib
import pickle
import shap

participant = 'aliceblackwood123@gmail.com'

X = pd.read_csv(f'{settings.not_filtered_model_dir}/{participant}_X.csv')

booster = xgb.Booster({'nthread': 4})
booster.load_model(f'{settings.not_filtered_model_dir}/{participant}.txt')
explainer = shap.TreeExplainer(model=booster)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
