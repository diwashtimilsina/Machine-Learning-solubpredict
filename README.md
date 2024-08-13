**Solubility Testing Project
**_Overview_****
This project aims to predict the solubility of compounds based on various molecular descriptors using linear regression. The key parameters used for model building are:

**MolLogP**: Logarithm of the partition coefficient between octanol and water.
**MolWt**: Molecular weight of the compound.
**NumRotatableBonds**: Number of rotatable bonds in the molecule.
**AromaticProportion**: Proportion of aromatic atoms in the molecule.
**logS**: Logarithm of the solubility in water.
**Dataset**
The dataset used for this project includes:

MolLogP: LogP values for each compound.
MolWt: Molecular weights of the compounds.
NumRotatableBonds: Count of rotatable bonds in each compound.
AromaticProportion: Ratio of aromatic atoms.
logS: Log of solubility in water (target variable).
**Model**
Linear Regression
Linear regression is used to model the relationship between the molecular descriptors and the solubility of the compounds. The model is trained on the provided dataset and evaluated to predict solubility values based on the given parameters.

**Evaluation Metrics
**The performance of the model is evaluated using the following metrics:

Mean Squared Error (MSE): Measures the average of the squares of errors.
R-squared (R2): Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
Accuracy: For regression tasks, this is often evaluated through MSE and R2.
