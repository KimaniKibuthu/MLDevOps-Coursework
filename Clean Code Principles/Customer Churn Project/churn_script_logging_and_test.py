import os
import numpy as np
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """

    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    """
    test perform eda function
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        cls.perform_eda(df)
        logging.info("Performing EDA: SUCCESS")
    except KeyError as err:
        logging.error("Performing EDA: A column stated was not found in the list of df columns")
        raise err

    try:
        required_images = ['churn_distribution.png',
                           'customer_age_distribution.png',
                           'heatmap.png',
                           'marital_status_distribution.png',
                           'total_transaction_distribution.png']

        list_of_images = os.listdir('./images/eda')
        is_image = []
        for image in list_of_images:
            is_image.append(image in required_images)

        assert len(set(is_image)) == 1
        assert len(list_of_images) >= 5

    except AssertionError as err:
        logging.error("Performing EDA: Not all plots were saved")
        raise err


def test_encoder_helper():
    """
    test encoder helper
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        cls.perform_eda(df)
        cat_columns = ['Gender',
                       'Education_Level',
                       'Marital_Status',
                       'Income_Category',
                       'Card_Category']
        new_df = cls.encoder_helper(df, cat_columns, 'Churn')
        logging.info("Testing encoder_helper: SUCCESS")

    except KeyError as err:
        logging.error("Testing encoder_helper: The columns not found in df")
        raise err

    try:
        assert sum(df.isnull().sum()) == 0
        assert len(new_df) > int(np.floor(0.7 * df.shape[0]))

    except AssertionError as err:
        logging.error("Testing encoder_helper: The file doesn't input the column data correctly")
        raise err


def test_perform_feature_engineering():
    """
    test perform_feature_engineering
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        cls.perform_eda(df)
        cat_columns = ['Gender',
                       'Education_Level',
                       'Marital_Status',
                       'Income_Category',
                       'Card_Category']
        new_df = cls.encoder_helper(df, cat_columns, 'Churn')
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(new_df, 'Churn')
        logging.info("Testing perform_feature_engineering: SUCCESS")

    except KeyError as err:
        logging.error("Testing perform_feature_engineering: The columns not found in df")
        raise err

    try:
        assert x_train.shape[0] >= int(np.floor(0.7 * df.shape[0]))
        assert x_test.shape[0] >= int(np.floor(0.3 * df.shape[0]))
        assert x_train.shape[1] == 19
        assert x_test.shape[1] == 19
        assert y_train.shape[0] >= int(np.floor(0.7 * df.shape[0]))
        assert y_test.shape[0] >= int(np.floor(0.3 * df.shape[0]))

    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: The data is not split correctly")
        raise err


def test_train_models():
    """
    test train_models
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        df = cls.import_data("./data/bank_data.csv")
        cls.perform_eda(df)
        cat_columns = ['Gender', 
                       'Education_Level',
                       'Marital_Status',
                       'Income_Category',
                       'Card_Category']
        
        new_df = cls.encoder_helper(df, cat_columns, 'Churn')
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(new_df, 'Churn')
        cls.train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")

    except (TypeError, NameError) as err:
        logging.error("Testing train_models: The parameters not correctly defined.")
        raise err

    try:
        required_images = ['feature_importances.png',
                           'logistic_results.png',
                           'rf_results.png',
                           'roc_curve_result.png']

        required_models = ['logistic_model.pkl', 'rfc_model.pkl']

        list_of_images = os.listdir('./images/results')
        list_of_models = os.listdir('./models')

        is_images = []
        is_models = []

        for image in list_of_images:
            is_images.append(image in required_images)

        for model in list_of_models:
            is_models.append(model in required_models)

        assert len(set(is_images)) == 1
        assert len(set(is_models)) == 1

    except AssertionError as err:
        logging.error("Testing train_models: The data is not saved properly")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
