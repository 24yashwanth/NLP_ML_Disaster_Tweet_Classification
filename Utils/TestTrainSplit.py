from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# to split the data set into train and test sets
def getTestTrainSplit(X, y, test_size=0.28, random_state=0):
    corpus = []
    dataFrameLen = len(X)
    for i in range(dataFrameLen):
        corpus.append(X[i])
    cv = CountVectorizer(max_features=1700)
    X = cv.fit_transform(corpus).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
