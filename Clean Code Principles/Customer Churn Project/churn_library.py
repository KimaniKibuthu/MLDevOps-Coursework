# library doc string
'''
Module containing helper functions.

Author: Kimani Kibuthu

'''

# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    data = pd.read_csv(pth)
    return data


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Check out the churn
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist();
    plt.xlabel('Churn')
    plt.ylabel('Number of Customers')
    plt.title('Churn Rate')
    plt.savefig('./images/churn_rate.png')

    # Customer age distribution
    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist()
    plt.xlabel('Customer Age')
    plt.ylabel('Number of Customers')
    plt.title('Customer Age Distribution')
    plt.savefig('./images/customer_age_distribution.png')

    # Marital Status
    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar');
    plt.xlabel('Marital Status')
    plt.ylabel('Number of Customers')
    plt.title('Customer Marital Status')
    plt.savefig('./images/marital_status.png')

    # Total Trans Count
    plt.figure(figsize=(20,10)) 
    sns.histplot(df['Total_Trans_Ct'], kde=True).set(title='Total Transaction Cost')
    plt.savefig('./images/total_trans_ct.png')

    # Heatmap
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=True, cmap='Dark2_r', linewidths = 2).set(title='Heatmap')
    plt.savefig('./images/heatmap.png')
    





def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass