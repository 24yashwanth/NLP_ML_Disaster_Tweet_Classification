# ML_Project
Machine Learning Project for 5th Sem


**Steps to run the Model**

1. Install all the required packages

    `pip install numpy pandas wordcloud scikit-learn nltk Flask`

2. To run navigate to the project folder and run the following command

    `python main.py`
    

**Trouble Shooting**

1. If nltk related error occured then navigate to the `Utils` directory and then comment lines (similar to the below one) from `PreProcessData.py` file

    nltk.download('stopwords')

    nltk.download('wordnet')

    nltk.download('omw-1.4')
