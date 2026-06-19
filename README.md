# Molecular Solubility Prediction using Machine Learning

A machine learning regression project for predicting molecular water solubility (`logS`) using numerical molecular descriptors.

This project demonstrates a complete beginner-friendly machine learning workflow, including data loading, feature selection, model training, evaluation, visualization, and prediction for a new molecule entered by the user.
---
<img width="1672" height="941" alt="git02" src="https://github.com/user-attachments/assets/e098f38b-cb24-4adf-99de-eda7db81586d" />
---

## Project Overview

Molecular solubility is an important chemical property in areas such as chemistry, pharmacy, drug discovery, and material science. In many real-world applications, it is useful to estimate whether a molecule can dissolve well in water before running expensive laboratory experiments.

In this project, machine learning models are trained to predict the water solubility of molecules based on a small set of molecular descriptors.

The main objective is to predict the `logS` value of a molecule.

---

## What This Project Does

The project takes molecular descriptor values as input and predicts the molecule's water solubility.

```text
Molecular Descriptors  →  Machine Learning Model  →  Predicted logS
```

The workflow includes:

- Loading a molecular solubility dataset
- Separating input features and target values
- Splitting the dataset into training and testing sets
- Training regression models
- Comparing model performance
- Accepting custom user input for a new molecule
- Predicting the `logS` value for the new molecule
- Visualizing actual vs predicted solubility values

---

## Dataset

The dataset used in this project contains molecular descriptors and experimentally measured solubility values.

Each row represents one molecule.

The target column is `logS`, and the remaining columns are used as input features.

---

## Input Features

| Feature | Description | Unit |
|---|---|---|
| `MolLogP` | Lipophilicity / hydrophobicity of the molecule | Unitless |
| `MolWt` | Molecular weight of the molecule | g/mol |
| `NumRotatableBonds` | Number of rotatable bonds in the molecule | Count |
| `AromaticProportion` | Proportion of aromatic atoms or aromatic structure in the molecule | Ratio, unitless |

### 1. MolLogP

`MolLogP` describes how lipophilic or hydrophobic a molecule is.

In simple terms, it shows how much a molecule prefers fat-like environments compared to water. A higher `MolLogP` value often means the molecule is less water-soluble.

### 2. MolWt

`MolWt` stands for molecular weight.

It represents the mass of the molecule and is measured in grams per mole (`g/mol`).

### 3. NumRotatableBonds

`NumRotatableBonds` represents the number of rotatable bonds in the molecule.

This is a count-based descriptor and can give information about the flexibility of the molecule.

### 4. AromaticProportion

`AromaticProportion` represents the fraction of the molecule that contains aromatic structure.

It is a ratio and does not have a physical unit.

---

## Target Variable

| Target | Description |
|---|---|
| `logS` | Logarithmic water solubility of the molecule |

The model predicts `logS`, which is the logarithmic value of molecular solubility in water.

In simple terms:

- Higher `logS` means better water solubility
- Lower or more negative `logS` means lower water solubility

Example:

```text
logS = -2  → relatively more soluble
logS = -5  → less soluble
```

Because `logS` is logarithmic, a difference of 1 unit can represent a large difference in actual solubility.

---

## Machine Learning Models

This project uses two regression models:

### Linear Regression

Linear Regression is used as a baseline model. It tries to learn a linear relationship between the molecular descriptors and the target value `logS`.

### Random Forest Regressor

Random Forest Regressor is an ensemble learning model based on multiple decision trees. It can capture more complex and non-linear relationships between molecular descriptors and solubility.

---

## Model Evaluation

The models are evaluated using the following metrics:

| Metric | Description |
|---|---|
| `MSE` | Mean Squared Error. Lower values indicate better performance. |
| `R² Score` | Shows how well the model explains the target values. Values closer to 1 are better. |

The project compares both models on training and testing data.

---

## New Molecule Prediction

The script allows the user to enter custom values for a new molecule:

- `MolLogP`
- `MolWt`
- `NumRotatableBonds`
- `AromaticProportion`

If the user presses Enter without typing a value, the program automatically uses default sample values.

Default input values:

```text
MolLogP: 2.5
MolWt: 180
NumRotatableBonds: 3
AromaticProportion: 0.4
```

The Linear Regression model then predicts the `logS` value for the new molecule.

---

## Visualization

The project includes a regression plot that compares:

- Experimental `logS` values
- Predicted `logS` values

The plot also highlights the new molecule prediction using a different marker.

The red dashed line represents the perfect prediction line. Points closer to this line indicate better predictions.

---

## Project Workflow

```text
Dataset
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Evaluation
   ↓
Visualization
   ↓
New Molecule Prediction
```

---

## Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib

---


## Project Structure

```text
.
├── main.py
├── delaney_solubility_with_descriptors.csv
└── README.md
```

---

## Example Input

```text
MolLogP: 2.5
MolWt: 180
NumRotatableBonds: 3
AromaticProportion: 0.4
```

## Example Output

```text
Predicted logS: -2.97
```

This means the model predicts the molecule has a moderate level of water solubility.

---

## Limitations

This project is designed as an introductory machine learning project.

Some limitations include:

- The dataset is relatively small
- Only four molecular descriptors are used
- The model does not use molecular structures directly
- The prediction should be considered an estimate, not an exact laboratory measurement
- More advanced molecular representations such as SMILES are not used in this version

---

## Future Improvements

Possible improvements for future versions:

- Add more molecular descriptors
- Use cross-validation
- Perform hyperparameter tuning
- Add more regression models
- Save the trained model using `joblib`
- Build a simple web interface using Streamlit
- Use molecular SMILES representations
- Compare more advanced machine learning algorithms

---

## Conclusion

This project demonstrates how machine learning can be used to predict a real chemical property from numerical molecular descriptors.

Although the project is simple, it covers the main steps of a machine learning workflow:

- Data loading
- Feature selection
- Model training
- Model evaluation
- Prediction
- Visualization

This is my first machine learning project and a starting point for building more advanced AI and chemistry-related projects.

---

## Repository Topics

Suggested GitHub topics:

```text
machine-learning
python
scikit-learn
regression
data-science
chemoinformatics
solubility-prediction
random-forest
linear-regression
matplotlib
```
