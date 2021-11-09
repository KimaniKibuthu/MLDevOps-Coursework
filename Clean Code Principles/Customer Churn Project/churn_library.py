"""
Module containing helper functions.

Author: Kimani Kibuthu

"""

# import libraries
import os
import joblib
import shap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    base_path = './images/eda'
    # Set the figure size
    plt.figure(figsize=(20, 10))

    # Check out the churn
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    df['Churn'].hist()
    plt.xlabel('Churn')
    plt.ylabel('Number of Customers')
    plt.title('Churn Rate')
    plt.savefig(os.path.join(base_path, 'churn_distribution.png'))

    # Customer age distribution
    df['Customer_Age'].hist()
    plt.xlabel('Customer Age')
    plt.ylabel('Number of Customers')
    plt.title('Customer Age Distribution')
    plt.savefig(os.path.join(base_path, 'customer_age_distribution.png'))

    # Marital Status
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.xlabel('Marital Status')
    plt.ylabel('Number of Customers')
    plt.title('Customer Marital Status')
    plt.savefig(os.path.join(base_path, 'marital_status_distribution.png'))

    # Total Trans Count 
    sns.histplot(df['Total_Trans_Ct'], kde=True).set(title='Total Transaction Cost')
    plt.savefig(os.path.join(base_path, 'total_transaction_distribution.png'))

    # Heatmap
    sns.heatmap(df.corr(), annot=True, cmap='Dark2_r', linewidths=2).set(title='Heatmap')
    plt.savefig(os.path.join(base_path, 'heatmap.png'))


def encoder_helper(df, category_list, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    # iterate through columns and create the new columns

    for column in category_list:
        temp_lst = []
        temp_groups = df.groupby(column).mean()[response]
        for val in df[column]:
            temp_lst.append(temp_groups.loc[val])

        df[f'{column}_{response}'] = temp_lst

    return df


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    # Columns to keep from the dataframe
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    # Split the data
    X = df[keep_cols]
    y = df[response]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_predictions_lr,
                                y_train_predictions_rf,
                                y_test_predictions_lr,
                                y_test_predictions_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_predictions_lr: training predictions from logistic regression
            y_train_predictions_rf: training predictions from random forest
            y_test_predictions_lr: test predictions from logistic regression
            y_test_predictions_rf: test predictions from random forest

    output:
             None
    """
    # Random forest results
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'size': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_predictions_rf)), {'size': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'size': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_predictions_rf)), {'size': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')

    # Linear regression results
    plt.text(0.01, 1.25, str('Linear Regression Train'), {'size': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_predictions_lr)), {'size': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6, str('Linear Regression Test'), {'size': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_predictions_lr)), {'size': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importance in pth
    input:
            model: model object containing feature_importance
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_data)
    shap.summary_plot(values, x_data, plot_type="bar", show=False)
    plt.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=10000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_predictions_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_predictions_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_predictions_lr = lrc.predict(x_train)
    y_test_predictions_lr = lrc.predict(x_test)

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=ax, alpha=0.8)
    plot_roc_curve(lrc, x_test, y_test, ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # scores & feature importance
    classification_report_image(y_train,
                                y_test,
                                y_train_predictions_lr,
                                y_train_predictions_rf,
                                y_test_predictions_lr,
                                y_test_predictions_rf)

    feature_importance_plot(cv_rfc.best_estimator_, x_train, './images/results/feature_importances.png')
