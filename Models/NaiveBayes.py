from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from werkzeug.datastructures import Accept

from Utils import Metrics


# to train the model
def trainModel(X_train, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier


# to predict for test data
def predict(model, X_test):
    y_predicted = model.predict(X_test)
    return y_predicted


# to predict for test.csv
def predictForNewData(model, test_data):
    corpus = []
    X = test_data['text_1']
    dataFrameLen = len(X)
    for i in range(dataFrameLen):
        corpus.append(X[i])
    cv = CountVectorizer(max_features=1700)
    X = cv.fit_transform(corpus).toarray()
    test_data['predicted'] = predict(model, X)
    return test_data['predicted']


# algorithm
def Algo(X_train, X_test, y_train, y_test, test_data):
    print('Training the Naive Bayes Model')
    naiveBayesModel = trainModel(X_train, y_train)
    y_predicted = predict(naiveBayesModel, X_test)
    # printing the metrics
    print('Calculating the Metrics')
    Accuracy=Metrics.findAccuracy(y_test, y_predicted)
    Metrics.getConfusionMatrix(y_test, y_predicted)
    # testing on new data
    print('Predicting for New Data')
    test_y_predicted = predictForNewData(naiveBayesModel, test_data)
    print(test_data.tail(1))
    if (test_y_predicted[3263] == 1):
        print("Predicted Outcome : Disaster Tweet")
    else:
        print("Predicted Outcome : Not a Disaster Tweet")
    return ['{:.2f}'.format(Accuracy),test_y_predicted[3263]]
