# ai-exp
ML experiments with Aerial Intelligence data for wheat yield forecasting.

## Problem description
Tabular geographic, time, weather and NDVI data for corresponding wheat yields is given. Through machine learning methods an estimator should be created which can forecast future yields based on measurements.

## Timeline
1. Read full instructions on github
2. Check the given data files:
  * How many columns?
  * What columns exist?
  * What are their data types?
3. Load CSV files with python and do more checks:
  * How many data rows?
    * 288033
  * Is data missing?
    * `precipIntensity`, `precipIntensityMax`, `precipProbability`, `pressure`, `visibility` has missing entries (-> drop them)
4. Visual analysis:
  * plot lat/lon with colorized yield on map
    * strong variations by lat/lon visible
  * other columns have almost zero (visible!) effect on yield
5. Machine learning
  * standardize data (zero mean and standard deviation of 1)
  * first try: linear regression (naiv and fast)
    * results in low accuracy (probably not linear relations!)
  * second try: decision tree (fast on big data sets)
    * better accuracy but maybe bagging methods give better results
  * thrid try: random forest (still fast but uses averaging over a set of (random) decision trees)
    * accuracy: 0.972 (+/- 0.001) very good!
    * 15 trees give reasonable accuracy in acceptable runtime
  * final improvements tried: different column combinations

## Best approach
*Random forest regressor*
Accuracy: **0.972 (+/- 0.001)** with 5 fold cross validation on training set (80% of
the data)
Explained variance score: **0.98**
Median absolute error: **0.09**

## Technical choices
* Programming language: **python2.7**
  * most experience and well-suited for ML
* Data manipulation toolkits: **numpy** and **pandas**
  * imho best choices for python
* ML toolkit: **scikit-learn**
  * works well with numpy (and pandas) + quick implementations possible
* Visualization toolkit: **matplotlib**
  * standard choice

## Challenges faced
* Display of lat/lon data on map in python (never done before)
* Big data amount reduced possible ML methods

## Lessons learned
* How to use matplotlib basemap module

## Possible To-Do
* Impute missing data (e.g. with average or most frequent value) and use these columns too
* Try to reduce dimensions with PCA

**PS: Since this is just an experiment I commented out the 'old' regression
estimators (linear, tree). I would not leave commented code in a commit
otherwise.**
