import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scripts import settings
from scripts import utils
import xgboost as xgb
import pandas as pd
import matplotlib
import pickle
import shap

train_dir = settings.combined_filtered_dataset_dir
test_numbers = settings.combined_best_tests

for participant in settings.participants:
    model, X, Y = utils.participant_train_for_model(participant=participant, train_dir=train_dir, test_number=test_numbers[participant])
    feature_importances = {}
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

    explainer = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(X, tree_limit=model.best_ntree_limit)
    shap.summary_plot(shap_value, X, title=participant)

# explainer = shap.TreeExplainer(model=model)
# shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
