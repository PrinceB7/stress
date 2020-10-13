import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scripts import settings
from scripts import utils
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib
import pickle
import shap

train_dir = settings.combined_filtered_dataset_dir
test_numbers = settings.combined_best_tests

for participant in settings.participants:
    if not participant.startswith('azizsambo'):
        continue
    models, Xs, conf_mtx = utils.participant_train_for_model(participant=participant, train_dir=train_dir, test_number=test_numbers[participant])

    # region plot confusion matrix
    fig = go.Figure(
        go.Heatmap(
            x=['Pred. Label = 0', 'Pred Label = 1'],
            y=['True. Label = 1', 'True. Label = 0'],
            z=np.flip(conf_mtx, axis=0),
        )
    )
    fig.update_layout(
        title_text=f'Confusion matrix for XGBoost evaluation ({participant})',
        xaxis_title_text='Prediction',
        yaxis_title_text='True (GT)'
    )
    fig.show()
    # endregion

    # region plot feature importance
    feature_importances = {}
    for model in models:
        imp = model.get_fscore()
        for k, v in imp.items():
            if k in feature_importances:
                feature_importances[k] += v
            else:
                feature_importances[k] = v
    feature_importances = [(k, float(v) / 5) for k, v in feature_importances.items()]
    df = pd.DataFrame(feature_importances, columns=['name', 'importance']).sort_values('importance', ascending=False).head(50)
    fig = go.Figure(
        go.Bar(
            x=df.loc[:, 'name'],
            y=df.loc[:, 'importance']
        )
    )
    fig.update_layout(
        title_text='Feature importance',
        yaxis_title_text='Feature importances'
    )
    fig.show()
    # endregion

    # region shap
    shap_values = []
    test_features = []
    for model, X in zip(models, Xs):
        explainer = shap.TreeExplainer(model)
        shap_value = explainer.shap_values(X, tree_limit=model.best_ntree_limit)
        shap_values += [shap_value]
        test_features += [X]
    shap.summary_plot(shap_values, test_features, title=f'SHAP for {participant}')
    # endregion

# explainer = shap.TreeExplainer(model=model)
# shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
