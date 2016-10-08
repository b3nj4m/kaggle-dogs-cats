from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression
import pandas
import numpy
import math
import matplotlib.pyplot as plot
import matplotlib
matplotlib.style.use('ggplot')

#TODO ratio of lot frontage to lot area?
# parameter tuning
# add ratings, sizes of basement types 1 and 2?
# subtract low quality sq ft from total?
# miscfeature/miscval?

def getTrainData():
  trainData = pandas.read_csv('./train.csv')
  trainOutcomes = trainData['SalePrice']
  trainData.drop(['SalePrice'], axis=1, inplace=True)
  return trainData, trainOutcomes

def getTestDataKaggle():
  testData = pandas.read_csv('./test.csv')
  ids = testData['Id']
  return testData, ids

def dataLocal():
  trainData, trainOutcomes = getTrainData()
  trainData = transformData(trainData)
  trainData, testData, trainOutcomes, testOutcomes = train_test_split(trainData, trainOutcomes, test_size=0.20)
  selection = selectFeatures(trainData, trainOutcomes)
  trainData = trainData.loc[:,selection]
  testData = testData.loc[:,selection]
  print('selected features', trainData.columns)
  return trainData, trainOutcomes, testData, testOutcomes

def dataKaggle():
  trainData, trainOutcomes = getTrainData()
  testData, ids = getTestDataKaggle()
  allData = pandas.concat([trainData, testData])
  allData = transformData(allData)
  trainSize, trainCols = trainData.shape
  testSize, testCols = testData.shape
  testData = allData.values.take(range(trainSize, trainSize + testSize), axis=0)
  trainData = allData.values.take(range(0, trainSize), axis=0)
  selection = selectFeatures(trainData, trainOutcomes)
  trainData = trainData[:,selection]
  testData = testData[:,selection]
  return trainData, trainOutcomes, testData, ids

def dummies(data, col):
  dumm = pandas.get_dummies(data[col], dummy_na=False, prefix=col + '_')
  data.drop([col], inplace=True, axis=1)
  data[dumm.columns] = dumm
  return data

def catToNumeric(series, values):
  valuesMap = {'nan': 0}
  i = 0
  for val in values:
    i = i + 1
    valuesMap[val] = i
  return series.apply(lambda x: valuesMap[str(x)])

def transformData(data):
  data['AreaPerFrontage'] = data['LotArea'] / data['LotFrontage']
  data['Age'] = 2011 - data['YearBuilt']
  data['GarageRecent'] = data['GarageYrBlt'].apply(lambda x: 1 if 2011 - x < 10 else 0)
  data['RemodRecent'] = data['YearRemodAdd'].apply(lambda x: 1 if 2011 - x < 10 else 0)
  data['SoldWinter'] = data['MoSold'].apply(lambda x: 1 if x > 11 or x < 3 else 0)
  data['SoldSummer'] = data['MoSold'].apply(lambda x: 1 if x > 5 and x < 9 else 0)

  qualityValues = ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
  qualityCols = [
    'ExterQual',
    'ExterCond',
    'BsmtQual',
    'BsmtCond',
    'HeatingQC',
    'KitchenQual',
    'FireplaceQu',
    'GarageQual',
    'GarageCond',
    'PoolQC',
  ]
  for col in qualityCols:
    data[col] = catToNumeric(data[col], qualityValues)

  dummyCols = [
    'MSSubClass',
    'MSZoning',
    'Street',
    'Alley',
    'LotShape',
    'LandContour',
    'Utilities',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'Foundation',
    'BsmtExposure', #TODO could be numeric...
    'BsmtFinType1', #TODO maybe combine with BsmtFinSF1
    'BsmtFinType2',
    'Heating',
    'CentralAir',
    'Electrical',
    'Functional',
    'GarageType',
    'GarageFinish',
    'PavedDrive',
    'Fence',
    'SaleType',
    'SaleCondition',
  ]
  for col in dummyCols:
    data = dummies(data, col)

  drops = [
    'Id',
    'YearBuilt',
    'YearRemodAdd',
    'GarageYrBlt',
    'MiscFeature',
    'MiscVal',
    'MoSold',
  ]
  data.drop(drops, axis=1, inplace=True)
  data.fillna(0, inplace=True)

  for col in data.columns:
    if data[col].var() == 0:
      data.drop([col], axis=1, inplace=True)
  data = (data - data.mean()) / data.std()

  return data

def selectFeatures(data, outcomes):
  selection = SelectPercentile(mutual_info_regression, percentile=50).fit(data, outcomes)
  return selection.get_support()

def testLocal(model):
  trainData, trainOutcomes, testData, testOutcomes = dataLocal()
  model.fit(trainData, trainOutcomes)
  prediction = model.predict(testData)
  print(prediction[:10])
  testOutcomes = testOutcomes.values.astype(numpy.float64)
  print(testOutcomes[:10])
  print(numpy.sqrt(mean_squared_error(numpy.log(testOutcomes), numpy.log(prediction))))
  return prediction

def testLocalNN(model, epochs = 10):
  trainData, trainOutcomes, testData, testOutcomes = dataLocal()
  model.fit(trainData.values, trainOutcomes.values.reshape(-1, 1), n_epoch=epochs)
  prediction = numpy.reshape(model.predict(testData.values), -1)
  print(prediction[:10])
  testOutcomes = testOutcomes.values.astype(numpy.float64)
  print(testOutcomes[:10])
  print(numpy.sqrt(mean_squared_error(numpy.log(testOutcomes), numpy.log(prediction))))
  return prediction

def testKaggle(model):
  trainData, trainOutcomes, testData, ids = dataKaggle()
  model.fit(trainData, trainOutcomes)
  prediction = model.predict(testData)
  predictionDF = pandas.DataFrame({
    'Id': ids,
    'SalePrice': prediction
  })
  predictionDF.to_csv('./prediction.csv', index=False)
  return prediction

def testKaggleNN(model, epochs = 10):
  trainData, trainOutcomes, testData, ids = dataKaggle()
  model.fit(trainData, trainOutcomes.values.reshape(-1, 1), n_epoch=epochs)
  prediction = numpy.reshape(model.predict(testData), -1)
  predictionDF = pandas.DataFrame({
    'Id': ids,
    'SalePrice': prediction
  })
  predictionDF.to_csv('./prediction.csv', index=False)
  return prediction

def getNN():
  tflearn.init_graph(num_cores=2)

  #TODO way to not hard-code shape
  net = tflearn.input_data(shape=[None, 132])
  net = tflearn.fully_connected(net, 256, activation='linear')
  net = tflearn.dropout(net, 0.8)
  net = tflearn.fully_connected(net, 64, activation='linear')
  net = tflearn.fully_connected(net, 1, activation='linear')
  net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.01)

  return tflearn.DNN(net)

if __name__ == '__main__':
  from sklearn.svm import SVR
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.linear_model import LassoCV, RidgeCV
  import tflearn
  #testLocal(LassoCV(alphas=numpy.logspace(0, -4, 10), n_jobs=2))
  testKaggle(RidgeCV(alphas=numpy.logspace(0, 5, 10)))
  #testLocalNN(getNN(), 300)
  #testLocal(SVR(kernel='linear', C=0.1, cache_size=2048))
  #testLocal(RandomForestRegressor(n_estimators=200, n_jobs=2, max_depth=12))
