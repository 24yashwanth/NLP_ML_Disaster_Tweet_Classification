from numpy import log
import pandas as pd
from Utils import PreProcessData, TestTrainSplit
from Models import NaiveBayes, LogisticRegression, KNearestNeighbours, DecisionTree
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def defaultRoute():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    keyword = request.form.get('TweetKeyWord')
    location = request.form.get('TweetLocation')
    text = request.form.get('TweetText')
    loading = True
    # ---------------------------Train Data set--------------------------------------------
    # getting the train dataset
    print('Retrieving the Train Data')
    train_data = pd.read_csv('./Data/train.csv')
    # preprocessing the Data
    train_data['keyword'].fillna('', inplace=True)
    train_data['location'].fillna('', inplace=True)
    train_data['text_1'] = train_data['keyword'] + " " + \
        train_data['location'] + " " + train_data['text']
    print('Preprocessing the Train Data')
    train_data['text_1'] = train_data['text_1'].apply(
        PreProcessData.preProcessData)

    # Splitting the Dataset at 80% to train and 20% to test
    X = train_data['text_1']
    y = train_data['target']
    print('Splitting the Train Data in 80:20 ratio to Train and Test')
    X_train, X_test, y_train, y_test = TestTrainSplit.getTestTrainSplit(X, y)

    # -------------------------------------------------------------------------------------

    # ------------------------------ Test Data set-----------------------------------------
    # getting the test dataset
    test_data = pd.read_csv('./Data/test.csv')

    # preprocessing the Data
    test_data['keyword'].fillna('', inplace=True)
    test_data['location'].fillna('', inplace=True)
    test_data.loc[len(test_data.index)] = [10876, keyword, location, text]
    test_data['text_1'] = test_data['keyword'] + " " + \
        test_data['location'] + " " + test_data['text']
    test_data['text_1'] = test_data['text_1'].apply(
        PreProcessData.preProcessData)

    # -------------------------------------------------------------------------------------

    # ----------------------------------- Naive Bayes -------------------------------------

    print('\n\nNaive Bayes Algorithm')
    naiveBayesPredicted = NaiveBayes.Algo(
        X_train, X_test, y_train, y_test, test_data)

    # -------------------------------------------------------------------------------------

    # ----------------------------------- Logistic Regression------------------------------

    print('\n\n\nLogistic Regression Algorithm')
    logisticRegressionPredicted = LogisticRegression.Algo(
        X_train, X_test, y_train, y_test, test_data)

    # -------------------------------------------------------------------------------------

    # ----------------------------------K Nearest Neighbours-------------------------------

    print('\n\n\nK Nearest Neighbours Algorithm')
    kNearestNeighboursPredicted = KNearestNeighbours.Algo(
        X_train, X_test, y_train, y_test, test_data)

    # -------------------------------------------------------------------------------------

    # -------------------------------------Decision Tree-----------------------------------

    print('\n\n\nDecision Tree Algorithm')
    decisionTreePredicted = DecisionTree.Algo(
        X_train, X_test, y_train, y_test, test_data)
    print("\n\n")

    # -------------------------------------------------------------------------------------
    loading = False

    print("NB :",naiveBayesPredicted,"\nLR :",logisticRegressionPredicted,"\nKNN :",kNearestNeighboursPredicted,"\nDT :",decisionTreePredicted)

    return render_template('predict.html', loading=loading,naiveBayesPredicted=naiveBayesPredicted[1], logisticRegressionPredicted=logisticRegressionPredicted[1],
                           kNearestNeighboursPredicted=kNearestNeighboursPredicted[1],decisionTreePredicted=decisionTreePredicted[1],accuracyNB=naiveBayesPredicted[0],
                           accuracyLR=logisticRegressionPredicted[0],accuracyKN=kNearestNeighboursPredicted[0],accuracyDT=decisionTreePredicted[0])


app.run(debug=True)
