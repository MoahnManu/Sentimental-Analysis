Sentiment Analysis Web App 💬
This project is a web-based sentiment analysis tool built with Python and the Flask framework. It classifies user-provided text as either positive or negative using a machine learning model trained on a large dataset of movie reviews.

Project Description
The application takes a sentence or paragraph as input from a simple web interface. In the background, it uses an NLP pipeline to clean and process the text before feeding it into a pre-trained Multinomial Naive Bayes model. This project is a great demonstration of building an end-to-end data science application.

Key Features
Natural Language Processing (NLP): Implements text preprocessing techniques like cleaning and tokenization.

TF-IDF Vectorization: Converts raw text data into a numerical feature matrix that can be used by the machine learning model.

Machine Learning Model: Utilizes a Multinomial Naive Bayes classifier for sentiment prediction.

Web Integration: A simple, clean user interface built with Flask allows users to interact with the model via a web browser.

Technologies Used
Python: The core programming language.

Flask: A lightweight web framework for the application's front-end.

Pandas: Used for data manipulation and handling the dataset.

Scikit-learn: The primary library for building and evaluating the machine learning model.

Regex: Used for cleaning and preparing text data.

Dataset
The project uses the IMDb Dataset of 50K Movie Reviews, which you can use for this project.

Link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Setup and Installation
Follow these steps to set up and run the project on your local machine.

Prerequisites
Python 3.x installed.

pip for installing packages.

Step 1: Clone the Repository
First, clone this repository to your local machine.

Bash

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
(Note: Replace your-username and your-repository-name with your actual GitHub details.)

Step 2: Install Dependencies
Install all the required Python libraries using pip.

Bash

pip install Flask pandas scikit-learn
Step 3: Download the Dataset
Download the IMDB Dataset.csv file from the Kaggle link provided above.

Place the file directly into the project's root directory.

Step 4: Run the Application
With the files in place, you can now run the Flask application from your terminal.

Bash

python app.py
The application will start, and you will see a message indicating that the server is running.

Step 5: Access the Web App
Open your web browser and navigate to the following address:
http://127.0.0.1:5001/

You can now enter any text to get an instant sentiment prediction.
