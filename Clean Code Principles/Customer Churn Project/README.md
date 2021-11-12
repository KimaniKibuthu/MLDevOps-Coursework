# Predict Customer Churn

## Introduction

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The project entails predicting the customer churn of a bank. 

The repository contains several major files to achieve this goal. They include:
* data folder - Contains the data on customer churn.

* images folder - It contains the eda and results folders. Once the notebook is run, it will generate plots during data exploration which will be saved in the eda folder. During the modelling section, it will generate images of feature importance and model performance plots that will be saved in the results folder.

* models folder - It contains the two models used in modeling the data.

* logs folder - It contains the log created during the testing of the helper functions

* churn_library.py - It contains helper functions that are used in the notebook.

* churn_notebook.py - It contains the processes of data processing and modelling on the customer churn data.

* churn_script_logging_and_tests.py - It contains the tests on the helper functions in churn_library.py. Once run, it will fill the churn_library.log in the logs.


## Running the project
In order to successfully run the project and obtain the intended results, follow the steps as shown below:
* Run the churn_script_logging_and_tests.py. This is by running the command `pytest churn_script_logging_and_tests.py` in the terminal of an IDE such as pycharm or VS Code.
* Once all the tests are passed, it means therefore that the functions in the churn_library.py are good to go. The data generated during the process of running the tests will be populated in the images folder, logs folder, and models folder.
* To utilize the helper functions found in churn_library.py, you need to import the module with an alias cls. That is `import churn_library as cls`. Once imported, you can access the helper functions from the module. For example, in training the models, you can type in this code: `cls.train_models(x_train, x_test, y_train, y_test)`. You can view more of the uses of this functions in the churn_notebook.ipynb where they are used for purposes of data obtainance, data exploration and preprocessing and modelling.



