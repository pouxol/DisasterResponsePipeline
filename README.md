# Disaster Response Pipeline Project
by Pouya Kholdi

## Installations
There are no other libraries necessary beyond the ones contained in the Anaconda distribution of Python.
The code should run with no issues using Python versions 3.*.

## Project Motivation
Finding people in need and identifying what their needs are is very important in a natural disaster.
This project helps identifying and classifying disaster respondes messages.

## File Description
* **app** folder containing the html files for the web-app and
	* **run.py** is the python code for running the web-app and creating the plotly visualizations.
* **data** folder containing
	* **disaster_categories.csv**
	* **disaster_messages.csv**
	* **process_data.py** is the modularized python code that runs the ETL pipeline.
* **models** folder containing
	* **train_classifier.py** is the modularized python code that runs the ML pipeline.
* **ETL Pipeline Preparation.ipynb** is the Jupyter Notebook where the ETL code was initially written and tested.
* **ML Pipeline Preparation.ipynb** is the Jupyter Notebook where the ML code was initially written and tested.

## How to interact with the project.
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements, etc.
The data for this project was provided by Figure Eight.