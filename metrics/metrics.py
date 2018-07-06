import numpy as np
import pandas as pd

def multiclassLogLoss(actual, predicted, eps = 1e-14, doDeepCheck = True):
  """
  Do multiclass log loss calculations. 
  Note. With this metrics you better to be less confident then confident and wrong

  Function expects:
    `actual` - pandas dummy-encoded DataFrame for actual values, each value should be 0 or 1
    `predicted` - pandas DataFrame with probability for each class of actual
    `eps` - value to substitute 0 in probability to avoid log(0) -> -Inf
  """
  if isinstance(actual, pd.DataFrame):
    actual = actual.values
  if isinstance(actual, list):
    actual = np.array(actual)
  if isinstance(predicted, pd.DataFrame):
    predicted = predicted.values
  if isinstance(predicted, list):
    predicted = np.array(predicted)
  if not isinstance(actual, np.ndarray) or not isinstance(predicted, np.ndarray):
    raise TypeError("pandas or numpy expected as both `actual` and `predicted`")
  
  if np.any(actual.shape != predicted.shape):
    raise ValueError("Expected compatible in shape arrays. " + str(actual.shape) + " - " + str(predicted.shape))
  
  if len(actual.shape) != 2:
    raise ValueError("Expected both `actual` and `predicted` as a 2d arrays-like structures")

  if doDeepCheck and not(np.all(np.isin(actual, [0, 1])) and np.all(np.sum(actual, axis = 1) == 1)):
    raise ValueError("Expected `actual` to be dummy representation of var")

  if doDeepCheck and not(np.all(predicted > 0) and np.all(predicted < 1) and np.all(np.sum(predicted, axis = 1) == 1)):
    raise ValueError("Expected `predicted` to be probabilities of class for `actual`")
  else:
    predicted = np.clip(predicted, eps, 1 - eps)
  return -1.0 / actual.shape[0] * np.sum(actual * np.log(predicted))

# p1 = pd.DataFrame({"a": [1, 0, 0, 0], "b": [0, 1, 1, 0], "c": [0, 0, 0, 1]})
# p2 = pd.DataFrame({"a": [0.95, 0.01, 0.01, 0.25], "b": [0.01, 0.6, 0.5, 0.25], "c": [0.04, 0.39, 0.49, 0.5]})

# TODO: write class and test
# print(multiclassLogLoss(p1, p2))
# print(multiclassLogLoss(p1, "as"))
# print(multiclassLogLoss(p1, [[0.3, 0.4, 0.5], [0.7, 0.8, 0.2]]))
# print(multiclassLogLoss([[1, 2]], [[0.9, 0.1]]))
# print(multiclassLogLoss([[1, 0]], [[0.9, 0.15]]))