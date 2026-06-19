# Solubility Prediction using Linear Regression and Random Forest

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
df = pd.read_csv('delaney_solubility_with_descriptors.csv')

print('Dataset preview:')
print(df.head())

print('\nDataset shape:')
print(df.shape)


# Separate features and target
X = df.drop('logS', axis=1)
y = df['logS']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=100
)

print('\nTrain/Test shapes:')
print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)


# ==============================
# Linear Regression Model
# ==============================

lr = LinearRegression()
lr.fit(X_train, y_train)

y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_results = pd.DataFrame([
    ['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]
])

lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']


# ==============================
# Random Forest Model
# ==============================

rf = RandomForestRegressor(max_depth=10, random_state=100)
rf.fit(X_train, y_train)

y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame([
    ['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]
])

rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']


# Compare models
df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)

print('\nModel Comparison:')
print(df_models.round(4))


# ==============================
# Predict a New Molecule
# ==============================

# Default molecule values
DEFAULT_MOLLOGP = 2.5
DEFAULT_MOLWT = 180
DEFAULT_NUM_ROTATABLE_BONDS = 3
DEFAULT_AROMATIC_PROPORTION = 0.4


def get_user_value(feature_name, default_value):
    """Get a numeric value from the user. Pressing Enter keeps the default value."""
    user_input = input(f'{feature_name} [{default_value}]: ')

    if user_input.strip() == '':
        return default_value

    try:
        return float(user_input)
    except ValueError:
        print(f'Invalid input. Default value {default_value} will be used for {feature_name}.')
        return default_value


print('\nEnter new molecule values.')
print('Press Enter without typing anything to use the default values.\n')

mol_logp = get_user_value('MolLogP', DEFAULT_MOLLOGP)
mol_wt = get_user_value('MolWt', DEFAULT_MOLWT)
num_rotatable_bonds = get_user_value('NumRotatableBonds', DEFAULT_NUM_ROTATABLE_BONDS)
aromatic_proportion = get_user_value('AromaticProportion', DEFAULT_AROMATIC_PROPORTION)

new_molecule = pd.DataFrame({
    'MolLogP': [mol_logp],
    'MolWt': [mol_wt],
    'NumRotatableBonds': [num_rotatable_bonds],
    'AromaticProportion': [aromatic_proportion]
})

new_lr_prediction = lr.predict(new_molecule)[0]

print('\nNew Molecule Input:')
print(new_molecule.to_string(index=False))

print('\nLinear Regression Prediction for the New Molecule:')
print('Predicted logS:', round(new_lr_prediction, 3))
print('Note: Higher logS means better solubility.')


# Text for the chart annotation
new_molecule_label = (
    f'MolLogP: {mol_logp}\n'
    f'MolWt: {mol_wt}\n'
    f'Rotatable Bonds: {num_rotatable_bonds}\n'
    f'Aromatic Prop.: {aromatic_proportion}\n'
    f'Predicted logS: {new_lr_prediction:.2f}'
)


# ==============================
# Plot Linear Regression Results
# ==============================

plt.figure(figsize=(8, 6))

# Test data predictions
plt.scatter(
    y_test,
    y_lr_test_pred,
    color='#10B981',
    edgecolor='#065F46',
    alpha=0.65,
    s=55,
    label='Test molecules'
)

# The new molecule does not have an experimental logS value.
# So it is displayed on the reference line using its predicted logS value.
plt.scatter(
    new_lr_prediction,
    new_lr_prediction,
    color='#2563EB',
    edgecolor='#1E3A8A',
    marker='D',
    s=140,
    label='New molecule prediction'
)

min_value = min(y_test.min(), y_lr_test_pred.min(), new_lr_prediction)
max_value = max(y_test.max(), y_lr_test_pred.max(), new_lr_prediction)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color='#EF4444',
    linestyle='--',
    linewidth=2,
    label='Perfect prediction line'
)

# Add a readable label for the new molecule point
plt.annotate(
    new_molecule_label,
    xy=(new_lr_prediction, new_lr_prediction),
    xytext=(25, -35),
    textcoords='offset points',
    fontsize=9,
    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#1E3A8A', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='#1E3A8A', lw=1.5)
)

plt.xlabel('Experimental logS')
plt.ylabel('Predicted logS')
plt.title('Linear Regression: Experimental vs Predicted logS')
plt.grid(True, linestyle='--', alpha=0.25)
plt.legend(frameon=True)
plt.tight_layout()
plt.show()
