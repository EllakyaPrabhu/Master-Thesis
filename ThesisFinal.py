#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd

# Load the datasets
files = {
    "Immunization": "Immunization.csv",
    "Effective_Literacy_Rate": "Effective_Literacy_Rate.csv",
    "Childhood_Diseases": "Childhood_Diseases.csv",
    "Mortality": "Mortality.csv",
    "Ante_Natal_Care": "Ante_Natal_Care.csv",
    "Post_Natal_Care": "Post_Natal_Care.csv",
    "Delivery_Care": "Delivery_Care.csv",
    "Work_Status": "Work_Status.csv",
    "Household_Characteristics": "Household_Characteristics.csv",
    "Breastfeeding_And_Supplementation": "Breastfeeding_And_Supplementation.csv",
}

# Read all datasets into pandas DataFrames
dataframes = {name: pd.read_csv(path) for name, path in files.items()}

# Merge all datasets on 'State_Name' and 'State_District_Name'
merged_data = None
for name, df in dataframes.items():
    if merged_data is None:
        merged_data = df
    else:
        merged_data = pd.merge(merged_data, df, on=["State_Name", "State_District_Name"], how="outer")
merged_data.to_csv("Merged_Categorized_Data.csv", index=False)



# In[51]:


# Define the bins and labels for categorization
bins = [0, 50, 80, 100]  # Adjust these based on your data
labels = ["Low", "Medium", "High"]

# Identify percentage columns
percentage_columns = [
    col for col in merged_data.columns if merged_data[col].dtype in ['float64', 'int64'] and merged_data[col].max() <= 100
]

# Convert percentage columns into categories
for col in percentage_columns:
    merged_data[col + "_Category"] = pd.cut(merged_data[col], bins=bins, labels=labels, include_lowest=True)

# Save the merged and categorized data to a new file
merged_data.to_csv("Merged_binned_Data.csv", index=False)

print("Data has been merged and categorized. The output is saved as 'Merged_Categorized_Data.csv'.")


# In[53]:


import pandas as pd

# Load the dataset
data = pd.read_csv('Merged_binned_Data.csv')

# List of columns to check for imbalances
columns_to_plot = [
    'Children_Aged_12_23_Months_Who_Have_Received_Bcg',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Polio_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Dpt_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_Measles_Vaccine', 
    'Children_Aged_12_23_Months_Fully_Immunized',
    'Children_Who_Did_Not_Receive_Any_Vaccination',
    'Effective_Literacy_Rate_Male',
    'Effective_Literacy_Rate_Female',
    'Children_Suffering_From_Diarrhoea', 
    'Children_Suffering_From_Acute_Respiratory',
    'Children_Suffering_From_Fever',
    'Infant_Mortality_Rate',
    'Neo_Natal_Mortality_Rate', 
    'Post_Neo_Natal_Mortality_Rate',
    'Currently_Married_Pregnant_Women_Aged_15_49_Years_Registered_For_Anc',
    'Mothers_Who_Received_3_Or_More_Antenatal_Care',
    'Mothers_Who_Received_At_Least_One_Tetanus_Toxoid_Tt_Injection',
    'Mothers_Who_Consumed_Ifa_For_100_Days_Or_More',
    'Mothers_Who_Received_Post_Natal_Check_Up_Within_48_Hrs_Of_Delivery',
    'New_Borns_Who_Were_Checked_Up_Within_24_Hrs_Of_Birth',
    'Institutional_Delivery',
    'Delivery_At_Home',
    'Work_Participation_Rate_15_Years_And_Above_Male',
    'Work_Participation_Rate_15_Years_And_Above_Female',
    'Currently_Married_Illiterate_Women_Aged_15_49_Years',
    'Children_Breastfed_Within_One_Hour_Of_Birth',
    'Children_Aged_6_35_Months_Exclusively_Breastfed_For_At_Least_Six_Months',
    'Children_Who_Received_Foods_Other_Than_Breast_Milk_During_First_6_Months_Water',
    'Children_Who_Received_Foods_Other_Than_Breast_Milk_During_First_6_Months_Animal_Formula_Milk'
]

# Printing the count values for each category in each specified column
for column in columns_to_plot:
    print(f"Counts for {column}:")
    print(data[column].value_counts())
    print("\n")
    


# In[54]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Merged_binned_Data.csv')

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Plotting the distribution of 'Children_Aged_12_23_Months_Fully_Immunized'
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=data, x='Children_Aged_12_23_Months_Fully_Immunized', palette='viridis')
plt.title('Distribution of Children Aged 12-23 Months Fully Immunized')
plt.xlabel('Immunization Status')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate labels if necessary

# Save the plot to a file
plt.savefig('fully_immunized_distribution.png', dpi=300, bbox_inches='tight')

plt.show()


# In[55]:


import pandas as pd

# Load the dataset
data = pd.read_csv('Merged_binned_Data.csv')

# List of categorical columns to encode, excluding 'State_Name' and 'State_District_Name' if they exist
categorical_columns = [col for col in data.columns if col not in ['State_Name', 'State_District_Name']]

# Encoding 'High', 'Medium', 'Low' to 3, 2, 1 respectively
encoding_map = {'High': 3, 'Medium': 2, 'Low': 1}
for column in categorical_columns:
    data[column] = data[column].map(encoding_map)

# Show the new DataFrame structure
print(data.head())

# Optionally, save the new DataFrame to a CSV file
data.to_csv('Encoded_Complete_Data.csv', index=False)


# In[60]:


import pandas as pd

# Load the dataset
data = pd.read_csv('Encoded_Complete_Data.csv')



# Correlation matrix
correlation_matrix = data[['Children_Aged_12_23_Months_Fully_Immunized', 'Infant_Mortality_Rate', 'Neo_Natal_Mortality_Rate']].corr()
print(correlation_matrix)


# In[71]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Encoded_Complete_Data.csv')

# Selecting relevant columns
columns = ['Children_Aged_12_23_Months_Fully_Immunized', 'Infant_Mortality_Rate', 'Neo_Natal_Mortality_Rate']
relevant_data = data[columns]

# Compute the correlation matrix
corr_matrix = relevant_data.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap of Immunization and Mortality Rates')

# Save the plot to a file
plt.savefig('Heatmap of Immunization and Mortality Rates.png', dpi=300, bbox_inches='tight')
plt.show()


# In[65]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Load the dataset
data = pd.read_csv('Encoded_Complete_Data.csv')

# Prepare response variable
y = pd.Categorical(data['Infant_Mortality_Rate'], categories=[1, 2, 3], ordered=True)

# Model setup and fit for each vaccine
vaccines = ['Children_Aged_12_23_Months_Who_Have_Received_Bcg',
            'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Polio_Vaccine',
            'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Dpt_Vaccine',
            'Children_Aged_12_23_Months_Who_Have_Received_Measles_Vaccine']

results = {}
for vaccine in vaccines:
    X = data[[vaccine]]  # Predictor - one vaccine at a time
    mod = OrderedModel(y, X, distr='logit')
    res = mod.fit(method='bfgs', disp=False)  # disp=False to avoid output during fit
    results[vaccine] = res

# Print the summary of each model
for vaccine, result in results.items():
    print(f"Results for {vaccine}:")
    print(result.summary())
    print("\n")


# In[162]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Encoded_Complete_Data.csv')

# Select relevant columns
columns = [
    'Children_Aged_12_23_Months_Who_Have_Received_Bcg',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Polio_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Dpt_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_Measles_Vaccine',
    'Infant_Mortality_Rate',
    'Neo_Natal_Mortality_Rate'
]

# Create a smaller DataFrame with the relevant columns
relevant_data = data[columns]

# Calculate the correlation matrix
correlation_matrix = relevant_data.corr()

# Generate a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap for Individual Vaccine Coverage and Mortality Rates')
plt.savefig('Correlation Heatmap for Individual Vaccine Coverage and Mortality Rates.png', dpi=300, bbox_inches='tight')
plt.show()


# In[75]:


pip install imbalanced-learn


# In[94]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load the dataset
data = pd.read_csv('Encoded_Complete_Data.csv')

# Adjusting labels to start from 0 for all models' compatibility, especially XGBoost
data['Children_Aged_12_23_Months_Fully_Immunized'] -= 1

# Define features and target
features = [
    'Effective_Literacy_Rate_Male', 'Effective_Literacy_Rate_Female',
    'Currently_Married_Pregnant_Women_Aged_15_49_Years_Registered_For_Anc',
    'Mothers_Who_Received_3_Or_More_Antenatal_Care',
    'Mothers_Who_Received_At_Least_One_Tetanus_Toxoid_Tt_Injection',
    'Mothers_Who_Consumed_Ifa_For_100_Days_Or_More',
    'Mothers_Who_Received_Post_Natal_Check_Up_Within_48_Hrs_Of_Delivery',
    'New_Borns_Who_Were_Checked_Up_Within_24_Hrs_Of_Birth',
    'Institutional_Delivery', 'Delivery_At_Home',
    'Work_Participation_Rate_15_Years_And_Above_Male',
    'Work_Participation_Rate_15_Years_And_Above_Female'
]
target = 'Children_Aged_12_23_Months_Fully_Immunized'

X = data[features]
y = data[target]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Model training and evaluation
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Generate and print classification report for each model
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

    # Compute ROC-AUC if the model supports probability predictions
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test_scaled)
        roc_auc = roc_auc_score(y_test, probabilities, multi_class='ovr', average='weighted')
        


# In[137]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('Encoded_Complete_Data.csv')

# Adjusting labels to start from 0 for all models' compatibility, especially XGBoost
data['Children_Aged_12_23_Months_Fully_Immunized'] -= 1

# Define features and target
features = [
    'Effective_Literacy_Rate_Male', 'Effective_Literacy_Rate_Female',
    'Currently_Married_Pregnant_Women_Aged_15_49_Years_Registered_For_Anc',
    'Mothers_Who_Received_3_Or_More_Antenatal_Care',
    'Mothers_Who_Received_At_Least_One_Tetanus_Toxoid_Tt_Injection',
    'Mothers_Who_Consumed_Ifa_For_100_Days_Or_More',
    'Mothers_Who_Received_Post_Natal_Check_Up_Within_48_Hrs_Of_Delivery',
    'New_Borns_Who_Were_Checked_Up_Within_24_Hrs_Of_Birth',
    'Institutional_Delivery', 'Delivery_At_Home',
    'Work_Participation_Rate_15_Years_And_Above_Male',
    'Work_Participation_Rate_15_Years_And_Above_Female'
]
target = 'Children_Aged_12_23_Months_Fully_Immunized'

X = data[features]
y = data[target]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Model definitions with pipelines
models = {
    "Decision Tree": Pipeline([('scaler', scaler), ('smote', smote), ('classifier', DecisionTreeClassifier())]),
    "Random Forest": Pipeline([('scaler', scaler), ('smote', smote), ('classifier', RandomForestClassifier())]),
    "SVM": Pipeline([('scaler', scaler), ('smote', smote), ('classifier', SVC(probability=True))]),
    "XGBoost": Pipeline([('scaler', scaler), ('smote', smote), ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))])
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Print classification report for each model
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

    # Compute ROC-AUC if the model supports probability predictions
    if hasattr(model.named_steps['classifier'], "predict_proba"):
        probabilities = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, probabilities, multi_class='ovr', average='weighted')
        
        
        


# In[147]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Load and prepare your data
data = pd.read_csv('Encoded_Complete_Data.csv')
data = data.drop(['State_Name', 'State_District_Name'], axis=1)

X = data.drop('Children_Aged_12_23_Months_Fully_Immunized', axis=1)
y = data['Children_Aged_12_23_Months_Fully_Immunized']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Standardizing features
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)

# Setting up the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create a RandomForestClassifier object and GridSearchCV object
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train_smote, y_train_smote)

# Best parameters and model
best_grid = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

# Predict on the test data using the best model
y_pred = best_grid.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

print("Random Forest Classification Report:")
print(classification_report(y_test, best_grid.predict(X_test), target_names=['Low', 'Medium', 'High']))
cm_rf = confusion_matrix(y_test, best_grid.predict(X_test))
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Low', 'Medium', 'High'])
disp_rf.plot(cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.show()

from scipy.stats import iqr

# Assume classifiers have been imported and set up as 'best_xgb', 'best_svm', 'best_tree', 'best_grid'

# Setup for cross-validation
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
f1_scorer = make_scorer(f1_score, average='weighted')
scores_rf = cross_val_score(best_grid, X, y, scoring=f1_scorer, cv=cv)
se_rf = np.std(scores_rf) / np.sqrt(len(scores_rf))


# In[148]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# Load and prepare your data
data = pd.read_csv('Encoded_Complete_Data.csv')
data = data.drop(['State_Name', 'State_District_Name'], axis=1)

X = data.drop('Children_Aged_12_23_Months_Fully_Immunized', axis=1)
y = data['Children_Aged_12_23_Months_Fully_Immunized']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline with SMOTE and DecisionTreeClassifier
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Parameter grid for Decision Tree
param_grid = {
    'classifier__max_depth': [None, 10, 20, 30, 40],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# GridSearchCV setup
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Best model
print("Best parameters:", grid_search.best_params_)

# Evaluation
best_tree = grid_search.best_estimator_
y_pred = best_tree.predict(X_test)
print(classification_report(y_test, y_pred))

# Decision Tree
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, best_tree.predict(X_test), target_names=['Low', 'Medium', 'High']))
cm_dt = confusion_matrix(y_test, best_tree.predict(X_test))
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=['Low', 'Medium', 'High'])
disp_dt.plot(cmap=plt.cm.Blues)
plt.title("Decision Tree Confusion Matrix")
plt.show()

from scipy.stats import iqr

# Assume classifiers have been imported and set up as 'best_xgb', 'best_svm', 'best_tree', 'best_grid'

# Setup for cross-validation
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
f1_scorer = make_scorer(f1_score, average='weighted')
scores_tree = cross_val_score(best_tree, X, y, scoring=f1_scorer, cv=cv)
se_tree = np.std(scores_tree) / np.sqrt(len(scores_tree))


# In[149]:


from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import iqr

# Load and prepare your data
data = pd.read_csv('Encoded_Complete_Data.csv')
data = data.drop(['State_Name', 'State_District_Name'], axis=1)

X = data.drop('Children_Aged_12_23_Months_Fully_Immunized', axis=1)
y = data['Children_Aged_12_23_Months_Fully_Immunized']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Creating a pipeline with SMOTE and SVM
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', SVC(random_state=42))
])

# Parameter grid for SVM
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']
}

# GridSearchCV setup
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Best model
print("Best parameters:", grid_search.best_params_)

# Evaluation
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)
print(classification_report(y_test, y_pred))
print("\nSVM Classification Report:")
print(classification_report(y_test, best_svm.predict(X_test), target_names=['Low', 'Medium', 'High']))
cm_svm = confusion_matrix(y_test, best_svm.predict(X_test))
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Low', 'Medium', 'High'])
disp_svm.plot(cmap=plt.cm.Blues)
plt.title("SVM Confusion Matrix")
plt.show()

cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
f1_scorer = make_scorer(f1_score, average='weighted')
scores_svm = cross_val_score(best_svm, X, y, scoring=f1_scorer, cv=cv)
se_svm = np.std(scores_svm) / np.sqrt(len(scores_svm))


# In[150]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Load and prepare your data
data = pd.read_csv('Encoded_Complete_Data.csv')
data = data.drop(['State_Name', 'State_District_Name'], axis=1)
X = data.drop('Children_Aged_12_23_Months_Fully_Immunized', axis=1)
y = data['Children_Aged_12_23_Months_Fully_Immunized'] - 1  # Adjusting labels to start from 0

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline with SMOTE and XGBoost
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
])

# Parameter grid for XGBoost
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1]
}

# GridSearchCV setup
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Best model
print("Best parameters:", grid_search.best_params_)

# Evaluation
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# XGBoost
print("\nXGBoost Classification Report:")
print(classification_report(y_test, best_xgb.predict(X_test), target_names=['Low', 'Medium', 'High']))
cm_xgb = confusion_matrix(y_test, best_xgb.predict(X_test))
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['Low', 'Medium', 'High'])
disp_xgb.plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix")
plt.show()


from scipy.stats import iqr

# Assume classifiers have been imported and set up as 'best_xgb', 'best_svm', 'best_tree', 'best_grid'

# Setup for cross-validation
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
f1_scorer = make_scorer(f1_score, average='weighted')

# Collect F1 scores for each model
scores_xgb = cross_val_score(best_xgb, X, y, scoring=f1_scorer, cv=cv)
se_xgb = np.std(scores_xgb) / np.sqrt(len(scores_xgb))


# In[163]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# Load and prepare your data
data = pd.read_csv('Encoded_Complete_Data.csv')
data = data.drop(['State_Name', 'State_District_Name'], axis=1)
X = data.drop('Children_Aged_12_23_Months_Fully_Immunized', axis=1)
y = data['Children_Aged_12_23_Months_Fully_Immunized'] - 1  # Adjusting labels to start from 0

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline with SMOTE and Decision Tree Classifier
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Parameter grid for Decision Tree
param_grid = {
    'classifier__max_depth': [None, 10, 20, 30, 40],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# GridSearchCV setup
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Best model
print("Best parameters:", grid_search.best_params_)

# Evaluation
best_tree = grid_search.best_estimator_
y_pred = best_tree.predict(X_test)
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# Visualize the confusion matrix
cm_tree = confusion_matrix(y_test, best_tree.predict(X_test))
disp_tree = ConfusionMatrixDisplay(confusion_matrix=cm_tree, display_labels=['Low', 'Medium', 'High'])
disp_tree.plot(cmap=plt.cm.Blues)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Setup for cross-validation
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
f1_scorer = make_scorer(f1_score, average='weighted')

# Collect F1 scores for Decision Tree model
scores_tree = cross_val_score(best_tree, X, y, scoring=f1_scorer, cv=cv)
se_tree = np.std(scores_tree) / np.sqrt(len(scores_tree))
print(f"Standard Error for Decision Tree: {se_tree}")


# In[217]:


from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Assuming you have already loaded your data and split it into training and validation sets
data = pd.read_csv('Encoded_Complete_Data.csv')
X = data.drop(['Children_Aged_12_23_Months_Fully_Immunized', 'State_Name', 'State_District_Name'], axis=1)
y = data['Children_Aged_12_23_Months_Fully_Immunized']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Apply SMOTE to handle class imbalance for training and validation sets
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_val_smote, y_val_smote = smote.fit_resample(X_val, y_val)

# Display the number of instances in each class after SMOTE application
train_class_distribution = Counter(y_train_smote)
val_class_distribution = Counter(y_val_smote)
test_class_distribution

print("Class distribution in training set after SMOTE:", train_class_distribution)
print("Class distribution in validation set after SMOTE:", val_class_distribution)


# In[ ]:





# In[138]:


import numpy as np
from sklearn.metrics import classification_report, f1_score

# Assuming y_train and y_test are your training and test target vectors
minority_class = y_train.value_counts().idxmin()  # Find the minority class

# Create baseline predictions for the test set
baseline_predictions = np.full(shape=y_test.shape, fill_value=minority_class)

# Calculate true positives, false positives
tp = (y_test == minority_class).sum()
fp = (y_test != minority_class).sum()
tn = fn = 0  # No true negatives or false negatives in this baseline scenario

# Calculate baseline precision, recall, and F1 score for the minority class
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = 1  # Recall is 1 as all actual minority class instances are 'detected'
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Display the calculated metrics and counts
print("Baseline Metrics for Minority Class:")
print(f"Class: {minority_class}")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")


# In[140]:


import numpy as np
import matplotlib.pyplot as plt

# Presuming scores_xgb, scores_svm, scores_tree, scores_rf are defined from previous messages
# Collect scores for XGB from your previous setup
scores_xgb = cross_val_score(best_xgb, X, y, scoring=f1_scorer, cv=cv)
se_xgb = np.std(scores_xgb) / np.sqrt(len(scores_xgb))

# Function to calculate DBM and OVS
def dbm_ovs(score1, score2):
    median1, median2 = np.median(score1), np.median(score2)
    dbm = abs(median1 - median2)
    ovs = max(np.max(score1), np.max(score2)) - min(np.min(score1), np.min(score2))
    return dbm / ovs if ovs != 0 else np.nan

# Comparing two models as an example, you can compare more models similarly
dbm_ovs_xgb_svm = dbm_ovs(scores_xgb, scores_svm)
dbm_ovs_xgb_tree = dbm_ovs(scores_xgb, scores_tree)
dbm_ovs_xgb_rf = dbm_ovs(scores_xgb, scores_rf)

# Plotting the boxplots
plt.boxplot([scores_xgb, scores_svm, scores_tree, scores_rf], labels=['XGBoost', 'SVM', 'Decision Tree', 'Random Forest'])
plt.ylabel('F1 Score')
plt.title('F1 Scores Distribution Across Models')
plt.savefig('F1_Scores_Distribution.png')  # Save the plot to a file
plt.show()

# Print standard errors and DBM/OVS ratio
print(f"Standard Errors: XGB: {se_xgb}, SVM: {se_svm}, Tree: {se_tree}, RF: {se_rf}")
print(f"DBM/OVS ratios: XGB vs. SVM: {dbm_ovs_xgb_svm}, XGB vs. Tree: {dbm_ovs_xgb_tree}, XGB vs. RF: {dbm_ovs_xgb_rf}")


# In[165]:


import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer
import numpy as np
import matplotlib.pyplot as plt

# Load and prepare your data
data = pd.read_csv('Encoded_Complete_Data.csv')
data = data.drop(['State_Name', 'State_District_Name'], axis=1)
X = data.drop('Children_Aged_12_23_Months_Fully_Immunized', axis=1)
y = data['Children_Aged_12_23_Months_Fully_Immunized'] - 1  # Adjusting labels to start from 0

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline and parameter grid for each model
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Setup cross-validation and scoring
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='weighted')

# Perform cross-validation and plot results for each model
fig, ax = plt.subplots(figsize=(12, 6))
positions = range(len(models))
for i, (name, model) in enumerate(models.items()):
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('standardize', StandardScaler()),
        ('classifier', model)
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=f1_scorer)
    se = np.std(scores) / np.sqrt(len(scores))  # Standard Error
    
    # Plotting
    ax.boxplot(scores, positions=[i], widths=0.6)
    ax.scatter([i] * len(scores), scores, alpha=0.5)
    print(f"{name} - Mean F1 Score: {np.mean(scores):.3f}, SE: {se:.3f}")

ax.set_xticks(positions)
ax.set_xticklabels(models.keys())
ax.set_ylabel('F1 Score')
ax.set_title('F1 Scores Distribution Across Models')
plt.grid(True)
plt.tight_layout()
plt.savefig('cross-validation_F1_Scores_Distribution.png')  # Save the plot to a file
plt.show()



# In[154]:


import pandas as pd

# Load the dataset
data = pd.read_csv('Encoded_Complete_Data.csv')

# Calculate value counts for each column and print the results
for column in data.columns:
    print(f"Value counts for {column}:")
    print(data[column].value_counts())
    print()  # Print a newline for better readability between columns


# In[156]:


import pandas as pd

# Load your data
data = pd.read_csv('Immunization.csv')

# Convert vaccination-related columns to numeric, coercing errors
vaccine_cols = [
    'Children_Aged_12_23_Months_Who_Have_Received_Bcg',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Polio_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Dpt_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_Measles_Vaccine',
    'Children_Aged_12_23_Months_Fully_Immunized',
    'Children_Who_Did_Not_Receive_Any_Vaccination'
]
for col in vaccine_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing numeric data with the mean of each column
data[vaccine_cols] = data[vaccine_cols].fillna(data[vaccine_cols].mean())

# Calculate the total average percentage of each vaccine
average_vaccination = data[vaccine_cols].mean()

# Calculate the average percentage of each vaccine for cities under each state
average_vaccination_by_state = data.groupby('State_Name')[vaccine_cols].mean()

# Total average of children who did not receive any vaccination in each state
average_no_vaccination_by_state = data.groupby('State_Name')['Children_Who_Did_Not_Receive_Any_Vaccination'].mean()

# State with the highest percentage of children who did not receive any vaccination
state_highest_no_vaccination = average_no_vaccination_by_state.idxmax()

# State with the best vaccination rate (assuming fully immunized percentage is the metric)
best_vaccination_rate_state = data.groupby('State_Name')['Children_Aged_12_23_Months_Fully_Immunized'].mean().idxmax()

# Printing the results
print("Average Vaccination Rate Across All Vaccines:\n", average_vaccination)
print("\nAverage Vaccination Rate by State:\n", average_vaccination_by_state)
print("\nAverage Percentage of Unvaccinated Children by State:\n", average_no_vaccination_by_state)
print("\nState with the Highest No-Vaccination Rate:", state_highest_no_vaccination)
print("\nState with the Best Vaccination Rate:", best_vaccination_rate_state)


# In[161]:


import pandas as pd

# Load your data
data = pd.read_csv('Mortality.csv')

# Convert mortality rate columns to numeric, coercing errors
mortality_cols = [
    'Infant_Mortality_Rate',
    'Neo_Natal_Mortality_Rate',
    'Post_Neo_Natal_Mortality_Rate'
]
for col in mortality_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing numeric data with the mean of each column
data[mortality_cols] = data[mortality_cols].fillna(data[mortality_cols].mean())

# Calculate the average mortality rates for cities under each state
average_mortality_by_state = data.groupby('State_Name')[mortality_cols].mean()

# Calculate the total mortality rate by summing all the specific mortality rates
data['Total_Mortality_Rate'] = data[mortality_cols].sum(axis=1)

# State with the highest total mortality rate
state_highest_mortality = data.groupby('State_Name')['Total_Mortality_Rate'].mean().idxmax()
state_lowest_mortality = data.groupby('State_Name')['Total_Mortality_Rate'].mean().idxmin()

# Printing the results
print("Average Mortality Rates by State:\n", average_mortality_by_state)
print("\nState with the Highest Mortality Rate:", state_highest_mortality)
print("\nState with the Lowest Mortality Rate:", state_lowest_mortality)


# In[159]:


import pandas as pd
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('Mortality.csv')

# Calculate the average mortality rates for each state
average_mortality = data.groupby('State_Name')[['Infant_Mortality_Rate', 'Neo_Natal_Mortality_Rate', 'Post_Neo_Natal_Mortality_Rate']].mean()

# Find the state with the highest overall mortality rate
# Summing the three types of mortality rates to get an overall rate for comparison
data['Total_Mortality_Rate'] = data['Infant_Mortality_Rate'] + data['Neo_Natal_Mortality_Rate'] + data['Post_Neo_Natal_Mortality_Rate']
highest_mortality_state = data.groupby('State_Name')['Total_Mortality_Rate'].mean().idxmax()

# Print the state with the highest mortality rate
print("State with the highest mortality rate:", highest_mortality_state)

# Visualization of the mortality rates across states
fig, ax = plt.subplots(figsize=(10, 8))
average_mortality.sort_values('Infant_Mortality_Rate', ascending=False).plot(kind='bar', ax=ax)
plt.title('Average Mortality Rates by State')
plt.ylabel('Mortality Rate')
plt.xlabel('State')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('State_Mortality_Rates.png')
plt.show()


# In[160]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the immunization data
data = pd.read_csv('Immunization.csv')

# Calculate the average vaccination rates for each vaccine by state
average_vaccination_rates = data.groupby('State_Name')[[
    'Children_Aged_12_23_Months_Who_Have_Received_Bcg',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Polio_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Dpt_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_Measles_Vaccine',
    'Children_Aged_12_23_Months_Fully_Immunized'
]].mean()

# Visualization of the vaccination rates across states
fig, ax = plt.subplots(figsize=(14, 10))
average_vaccination_rates.sort_values('Children_Aged_12_23_Months_Fully_Immunized', ascending=False).plot(kind='bar', ax=ax)
plt.title('Average Vaccination Rates by State')
plt.ylabel('Average Rate (%)')
plt.xlabel('State')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('State_Vaccination_Rates.png')
plt.show()


# In[166]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# Load and prepare your data
data = pd.read_csv('Encoded_Complete_Data.csv')
data = data.drop(['State_Name', 'State_District_Name'], axis=1)
X = data.drop('Children_Aged_12_23_Months_Fully_Immunized', axis=1)
y = data['Children_Aged_12_23_Months_Fully_Immunized'] - 1  # Adjusting labels to start from 0

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline with SMOTE and RandomForestClassifier
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid for RandomForest
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30, 40],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

# GridSearchCV setup
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Best model
print("Best parameters:", grid_search.best_params_)

# Evaluation
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# Visualize the confusion matrix
cm_rf = confusion_matrix(y_test, best_rf.predict(X_test))
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Low', 'Medium', 'High'])
disp_rf.plot(cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.show()

# Setup for cross-validation
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
f1_scorer = make_scorer(f1_score, average='weighted')

# Collect F1 scores for Random Forest model
scores_rf = cross_val_score(best_rf, X, y, scoring=f1_scorer, cv=cv)
se_rf = np.std(scores_rf) / np.sqrt(len(scores_rf))
print(f"Standard Error for Random Forest: {se_rf}")


# In[169]:


# Feature importance
feature_importances = best_rf.feature_importances_
feature_names = X.columns

# Create a DataFrame to view the features and their importance
features_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# List of features to exclude from the importance graph
features_df = [
    'Children_Aged_12_23_Months_Who_Have_Received_Bcg',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Polio_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Dpt_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_Measles_Vaccine',
    'Infant_Mortality_Rate',
    'Neo_Natal_Mortality_Rate',
    'Post_Neo_Natal_Mortality_Rate',
    'Children_Breastfed_Within_One_Hour_Of_Birth_Total',
    'Children_Aged_6_35_Months_Exclusively_Breastfed_For_At_Least_Six_Months_Total',
    'Children_Who_Received_Foods_Other_Than_Breast_Milk_During_First_6_Months_Water_Total',
    'Children_Who_Received_Foods_Other_Than_Breast_Milk_During_First_6_Months_Animal_Formula_Milk_Total',
    'Children_Who_Did_Not_Receive_Any_Vaccination',
    'Children_Suffering_From_Diarrhoea',
    'Children_Suffering_From_Fever',
    'Children_Suffering_From_Acute_Respiratory',
    'Children_Aged_6_35_Months_Exclusively_Breastfed_For_At_least_Six_Months',
    'Children_Breastfed_Within_One_Hour_Of_Birth',
    'Children_Aged_6_35_Months_Exclusively_Breastfed_For_At_Least_Six_Months',
    'Children_Who_Received_Foods_Other_Than_Breast_Milk_During_First_6_Months_Water', 
    'Children_Who_Received_Foods_Other_Than_Breast_Milk_During_First_6_Months_Animal_Formula_Milk'
]

# Filter out excluded features
features_df = features_df.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 8))
plt.barh(features_df['Feature'], features_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances (Selective)')
plt.gca().invert_yaxis()
plt.show()

# Print the sorted feature importance
print(features_df)


# In[176]:


# Feature importance
feature_importances = best_rf.feature_importances_
feature_names = X.columns

# Create a DataFrame to view the features and their importance
features_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# List of features to exclude from the importance graph
exclude_features = [
    'Children_Aged_12_23_Months_Who_Have_Received_Bcg',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Polio_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Dpt_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_Measles_Vaccine',
    'Infant_Mortality_Rate',
    'Neo_Natal_Mortality_Rate',
    'Post_Neo_Natal_Mortality_Rate',
    'Children_Breastfed_Within_One_Hour_Of_Birth_Total',
    'Children_Aged_6_35_Months_Exclusively_Breastfed_For_At_Least_Six_Months_Total',
    'Children_Who_Received_Foods_Other_Than_Breast_Milk_During_First_6_Months_Water_Total',
    'Children_Who_Received_Foods_Other_Than_Breast_Milk_During_First_6_Months_Animal_Formula_Milk_Total',
    'Children_Who_Did_Not_Receive_Any_Vaccination',
    'Children_Suffering_From_Diarrhoea',
    'Children_Suffering_From_Fever',
    'Children_Suffering_From_Acute_Respiratory',
    'Children_Aged_6_35_Months_Exclusively_Breastfed_For_At_least_Six_Months',
    'Children_Breastfed_Within_One_Hour_Of_Birth',
    'Children_Aged_6_35_Months_Exclusively_Breastfed_For_At_Least_Six_Months',
    'Children_Who_Received_Foods_Other_Than_Breast_Milk_During_First_6_Months_Water', 
    'Children_Who_Received_Foods_Other_Than_Breast_Milk_During_First_6_Months_Animal_Formula_Milk',
    'Children_Who_Did_Not_Receive_Any_Vaccination',
    'Mothers_Who_Consumed_Ifa_For_100_Days_Or_More ',
    'Mothers_Who_Received_At_Least_One_Tetanus_Toxoid_Tt_Injection',
    'Mothers_Who_Consumed_Ifa_For_100_Days_Or_More'
    
]

# Filter out excluded features
features_df = features_df[~features_df['Feature'].isin(exclude_features)]
features_df = features_df.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 8))
plt.barh(features_df['Feature'], features_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.savefig('Feature Importance.png')
plt.show()

# Print the sorted feature importance
print(features_df)


# In[207]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
file_path = 'Updated_Merged_binned_Data.csv'  # Update the file path if necessary
data = pd.read_csv(file_path)

# Preprocess the data: Drop unnecessary columns
columns_to_drop = ['State_Name', 'State_District_Name','Children_Aged_12_23_Months_Who_Have_Received_Bcg',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Polio_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Dpt_Vaccine',
    'Children_Aged_12_23_Months_Who_Have_Received_Measles_Vaccine',
    'Infant_Mortality_Rate',
    'Neo_Natal_Mortality_Rate',
    'Post_Neo_Natal_Mortality_Rate',
    'Children_Who_Did_Not_Receive_Any_Vaccination',
    'Children_Suffering_From_Diarrhoea',
    'Children_Suffering_From_Fever',
    'Children_Suffering_From_Acute_Respiratory',
    'Children_Aged_6_35_Months_Exclusively_Breastfed_For_At_least_Six_Months',
    'Children_Breastfed_Within_One_Hour_Of_Birth',
    'Children_Aged_6_35_Months_Exclusively_Breastfed_For_At_Least_Six_Months',
    'Children_Who_Received_Foods_Other_Than_Breast_Milk_During_First_6_Months_Water', 
    'Children_Who_Received_Foods_Other_Than_Breast_Milk_During_First_6_Months_Animal_Formula_Milk',
    'Children_Who_Did_Not_Receive_Any_Vaccination',
    'Mothers_Who_Consumed_Ifa_For_100_Days_Or_More ',
    'Mothers_Who_Received_At_Least_One_Tetanus_Toxoid_Tt_Injection',
    'Mothers_Who_Consumed_Ifa_For_100_Days_Or_More',
    'Work_Participation_Rate_15_Years_And_Above_Female']
data_processed = data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1)

# Convert categorical columns to one-hot encoding
data_encoded = pd.get_dummies(data_processed)

# Apply Apriori to find frequent itemsets
frequent_itemsets = apriori(data_encoded, min_support=0.05, use_colnames=True)
print(frequent_itemsets)


# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4, num_itemsets= 2)

# Filter rules where the consequent is 'Immunization_Category_High'
rules_high = rules[rules['consequents'] == frozenset({'Children_Aged_12_23_Months_Fully_Immunized_High'})]



# Sort rules by lift for better interpretability
rules_high_sorted = rules_high.sort_values(by='support', ascending=False)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

output_file = "all_association_rules_high.csv"
rules_high_sorted.to_csv(output_file, index=False)


# Select the top 10 rules
top_10_rules = rules_high_sorted.head(10)

# Display the top 10 rules in the console
print("Top 10 Association Rules Leading to Immunization_Category_High:")
print(top_10_rules)

# Save the top 10 rules to a CSV file
output_file = "top_10_association_rules_high.csv"
top_10_rules.to_csv(output_file, index=False)
print(f"Top 10 rules saved to {output_file}")


# In[213]:


import matplotlib.pyplot as plt
import networkx as nx
from textwrap import fill

# Create a graph
G = nx.DiGraph()

# Add nodes and edges (replace with actual rules)
rules = [
    ('Currently Married Illiterate Women Low + ANC High', 'Fully Immunized High'),
    ('ANC High + Male Literacy High', 'Fully Immunized High'),
    ('Illiterate Women Low + ANC High + Male Literacy High', 'Fully Immunized High'),
    ('Illiterate Women Low + ANC High + Male Work Medium', 'Fully Immunized High'),
    ('Illiterate Women Low + ANC High + Male Literacy High + Male Work Medium', 'Fully Immunized High')
]

for antecedent, consequent in rules:
    # Wrap text for better readability
    wrapped_antecedent = fill(antecedent, width=30)
    wrapped_consequent = fill(consequent, width=30)
    G.add_edge(wrapped_antecedent, wrapped_consequent)

# Plot the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  # Adjust layout for better spacing
nx.draw_networkx_nodes(G, pos, node_size=7000, node_color='lightblue')
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', verticalalignment="center")
plt.title("Network Graph of Association Rules", fontsize=14)
plt.savefig('Network Graph of Association Rules.png')
plt.show()


# In[214]:


import pandas as pd

# Load the dataset
data = pd.read_csv('Immunization.csv')

# Calculate the average percentage for each specified vaccine type
average_bcg = data['Children_Aged_12_23_Months_Who_Have_Received_Bcg'].mean()
average_polio = data['Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Polio_Vaccine'].mean()
average_dpt = data['Children_Aged_12_23_Months_Who_Have_Received_3_Doses_Of_Dpt_Vaccine'].mean()
average_measles = data['Children_Aged_12_23_Months_Who_Have_Received_Measles_Vaccine'].mean()

# Print the results
print(f"Average percentage of BCG vaccination: {average_bcg:.2f}%")
print(f"Average percentage of Polio vaccination (3 doses): {average_polio:.2f}%")
print(f"Average percentage of DPT vaccination (3 doses): {average_dpt:.2f}%")
print(f"Average percentage of Measles vaccination: {average_measles:.2f}%")


# In[215]:


import sys
print("Python version")
print(sys.version)
print("Version info.")
print(sys.version_info)


# In[222]:


import pandas as pd
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('Merged_Categorized_Data.csv')

# Selecting the columns of interest
columns_of_interest = [
    'Children_Aged_12_23_Months_Fully_Immunized', 
    'Effective_Literacy_Rate_Male', 
    'Effective_Literacy_Rate_Female', 
    'Currently_Married_Pregnant_Women_Aged_15_49_Years_Registered_For_Anc'
]

# Grouping by state and calculating the mean for the selected columns
state_averages = data.groupby('State_Name')[columns_of_interest].mean()

# Plotting the averages for each state
fig, ax = plt.subplots(figsize=(10, 8))
state_averages.plot(kind='bar', ax=ax)
plt.title('Socioeconomic factors average by State')
plt.ylabel('Average')
plt.xlabel('State')
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
plt.savefig('Socioeconomic factors by state.png')
plt.show()


# In[221]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Merged_Categorized_Data.csv')

# Columns of interest
columns_of_interest = [
    'Children_Aged_12_23_Months_Fully_Immunized',
    'Effective_Literacy_Rate_Male',
    'Effective_Literacy_Rate_Female',
    'Currently_Married_Pregnant_Women_Aged_15_49_Years_Registered_For_Anc'
]

# Group by state and calculate the mean for the specified columns
state_averages = data.groupby('State_Name')[columns_of_interest].mean()

# Print the state averages
print(state_averages)

# Find the states with the highest and lowest average for each column
for column in columns_of_interest:
    highest = state_averages[column].idxmax()
    lowest = state_averages[column].idxmin()
    print(f'{column}: Highest average in {highest}, Lowest average in {lowest}')

# Plotting
fig, axes = plt.subplots(len(columns_of_interest), 1, figsize=(10, 5 * len(columns_of_interest)))
for i, column in enumerate(columns_of_interest):
    state_averages[column].sort_values().plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'Average {column} per State')
    axes[i].set_ylabel(column)
    axes[i].set_xlabel('State')

plt.tight_layout()
plt.show()


# In[ ]:




