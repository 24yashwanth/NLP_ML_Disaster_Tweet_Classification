# to get the data info
def getInfo(dataFrame):
    print(dataFrame.info())


# to get the count of missing values
def getCountOfMissingValues(dataFrame):
    print(dataFrame.isna().sum())
