==============================
**M32895_Big_Data_Coursework**
**Bike Sharing Dataset**
==============================

==============================
## Introduction

This project builds a machine learning pipeline to predict hourly bike rental demand using the Bike Sharing Dataset from Kaggle.
The dataset is 17,379 hourly records from a bike sharing system in Washington D.C. between the years 2011 and 2012.
Categories are weather conditions, temperature, humidity, windspeed, season and hour of day.
The pipeline follows a data science workflow from data collection to model building, evaluation and prediction.
Two models were built and then compared, a linear regression as a baseline and a decision tree regressor as a more complex model.
The decision tree was then optimised using hyperparameter tuning to improve traning and performance.
The target variable `cnt` is the total number of bikes rented per hour.
This ML is a regression task because the model predicts a continuous numeric value.
The best performing model is saved and used to make predicitons on unseen data and then individual input.
All code is documented and referenced throughout the notebook.
Throughout the notebook, I regularly used `print()` statements to make the outputs clear and easy to follow. This helped me check that each stage of the pipeline was working correctly.
==============================

==============================
## Business Objectives

* **Purpose:** To predict hourly bike rental demand based on weather and seasonal conditions
* **Business question:** How many bikes will be rented in an hour while considering conditions such as weather, season and time of day?
* **Target:** Maximise R2 score and minimise RMSE on the test set to achieve the most accurate predictions possible
==============================

==============================
## ML Pipeline

The pipeline section below tells the story of the project from data acquisition to final prediction on unseen data.
Throughout the notebook, I regularly used `print()` statements to make the outputs clear and easy to follow. This helped me check that each stage of the pipeline was working correctly.

### Data Collection

The dataset was downloaded from Kaggle using the Kaggle API instead of manually downloading the CSV file from the Kaggle website.
When I set up Kaggle access, Kaggle provided an API token+code method. This allowed the notebook to authenticate and download the dataset directly into the dataset folder.
The dataset used was lakshmi25npathi/bike-sharing-dataset from Kaggle.

The code first authenticates with kaggle:

```
kaggle.api.authenticate()
```

It then downloads and unzips the Bike Sharing dataset into the project datasets folder:

```
kaggle.api.dataset_download_files(
    'lakshmi25npathi/bike-sharing-dataset',
    path='../datasets/',
    unzip=True
)

print("Dataset downloaded successfully")
```
**Validation:**

The dataset was validated by checking:

* The first five rows using `df.head()` to preview the structure of the dataset and check that the data loaded correctly
* Column types using `df.info()` to check they were suitable data types
* Statistical summary using `df.describe()` to see the range, mean, min and max values of the numeric columns
* Missing values using `df.isna().sum()` — no missing values were found
* Missing rows were still handled using `df.dropna()` for coursework presentation purposes to show an understanding of what that code does, eventhough it wasnt needed with this datatset.

**Splitting the Dataset**

The dataset was split into three subsets:

* **Training set** — used to train the models
* **Validation set** — created from the training data and used during hyperparameter optimisation
* **Test set** — 20% of the full dataset, kept completely unseen until final evaluation

The first split separated 20% of the full dataset as the final test set. 

```
X_train, X_test, y_train, y_test = train_test_split(
                                    df.drop(['cnt'], axis=1),
                                    df['cnt'],
                                    test_size=0.2,
                                    random_state=0
                                    )
print("* Train set:", X_train.shape, y_train.shape, "\n* Test set:", X_test.shape, y_test.shape)
```

The remaining 80% was then split again into training and validation data, so the validation set could be used to tune the Decision Tree `max_depth` without using the final test data.

```
X_train, X_val, y_train, y_val = train_test_split(
                                    X_train,
                                    y_train,
                                    test_size=0.2,
                                    random_state=0
                                    )

print("* Train set:", X_train.shape, y_train.shape)
print("* Validation set:", X_val.shape, y_val.shape)
print("* Test set:", X_test.shape, y_test.shape)
```

### EDA

The dataset contains 17,379 hourly records with 17 columns.

The histogram of `cnt` shows that most hourly records have lower bike rental counts, while very high bike rental counts happen less often.
This shows that bike demand is not evenly spread across the dataset and varies depending on different hours and conditions.

```
fig = px.histogram(
    df,
    x='cnt',
    marginal='box',
    title='Bike Rental Count Distribution'
)
fig.update_layout(bargap=0.1)
fig.show()
```
The histogram of `temp` shows that temperature values are spread across the dataset, with most records between around 0.2 and 0.7.
There are fewer records at very low and very high temperature values.
This graph helped me visualise the actual range of temperatures in the dataset and how it relates to bike rental demand. 

```
fig = px.histogram(
    df,
    x='temp',
    marginal='box',
    title='Temperature Distribution'
)
fig.update_layout(bargap=0.1)
fig.show()
```

The records by `season` graph shows that the dataset contains records from all four seasons.
The number of records is fairly similar across the seasons, so the dataset is not heavily focused on only one season.
This could have been useful later on because season could be included as a feature.

```
fig = px.histogram(
    df,
    x='season',
    color='season',  # colour bars by season
    title='Records by Season'
)
fig.show()
```

The temperature vs bike rentals scatter plot shows the relationship between `temp` and `cnt`.
The points show that bike rental counts usually increase as temperature increases.
This suggests that warmer weather is linked with higher bike rental demand.
However, the points are spread out which showed me that temperature is not the only factor affecting the number of rentals.
The points are coloured by season using `color= season`, which helps show the time of year aswell.

```
fig = px.scatter(
    df,
    x='temp',
    y='cnt',
    color='season',
    opacity=0.8,
    title='Temperature vs Bike Rentals'
)
fig.update_traces(marker_size=5)  # point size
fig.show()
```

The correlation matrix and heatmap check the relationships between the features.
This helped to see which columns had strong relationships with the target variable `cnt`.
The table showed that `temp` and `atemp` had a positive relationship with cnt, but `hum` had a negative relationship with cnt.
It also showed that `hr` was related to cnt, so hour of the day was useful for prediction.

```
df.select_dtypes(include='number').corr()
```
```
plt.figure(figsize=(14, 10))
sns.heatmap(
    df.select_dtypes(include='number').corr(),
    cmap='Reds',
    annot=True
)
plt.title('Correlation Matrix')
plt.show()
```

The correlation matrix also helped identify data leakage.
The columns `casual` and `registered` had very strong relationships with cnt because they add together to make the target value.
Keeping them would mean a data leakage because the model would effectively have the answer.
Because of this, I removed them before training the models.
The columns `instant` and `dteday` were also dropped because they are identifiers and not useful for this model.

```
df = df.drop(columns=[
    'instant',
    'dteday',
    'casual',
    'registered'
])

df.head()
```

Overall, the EDA helped identify patterns and useful features in the dataset and decide which columns should be removed before training.

### Model Building

Before training the models, I created a preprocessing pipeline using `Pipeline` and `StandardScaler`.
This was based on Lesson 8.1.
This was to scale the feature values so that all columns were treated more fairly during training.
Some features have larger numeric ranges than others, so scaling helps the input features to start on a similar scale.

The scaler was fit on the training set using `fit_transform()`.
This means the pipeline learned the scaling values from the training data only.
The scaling was then applied to the validation and test sets using `transform()`.
This is important because the validation and test data should stay unseen.

```
def pipeline_pre_processing():
    pipeline_base = Pipeline([

      ("feat_scaling", StandardScaler())

    ])
    return pipeline_base
```
```
X_train = pipeline.fit_transform(X_train)
X_val = pipeline.transform(X_val)
X_test = pipeline.transform(X_test)
```

Two models were built for this regression task:

**Linear Regression**

Linear Regression was chosen as a simple baseline model and was directly taught in Lesson 5.2.
A baseline model is useful because it gives a simple result to compare against a more complex model.
The model was created using `LinearRegression()` and trained using `.fit(X_train, y_train)`, which means it learned the relationship between the input features and the target variable `cnt`.

```
model_lr = LinearRegression()

model_lr.fit(X_train, y_train)

print("Linear Regression model trained successfully")
```

The output `w`shows the first coefficient printed from the Linear Regression model, and `b` the intercept is the starting value used by the linear model.

```
print("w: ", model_lr.coef_[0])
print("b: ", model_lr.intercept_)
```
**Decision Tree Regressor**

Lesson 6.1 taught decision trees using `DecisionTreeClassifier` for a classification problem.
My coursework is a little bit different because it predicts a continuous numeric value, so I used `DecisionTreeRegressor`instead.
This is a useful model becuase it is the regression equivalent to `DecisionTreeClassifier`.
This model was chosen because it can learn more complex patterns than Linear Regression and does not only rely on a straight-line relationship. (https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/)

The first Decision Tree Regressor used starting hyperparameters before optimisation.
The starting hyperparameter was `max_depth = 5` to limit how deep the tree could grow, which reduces overfitting.
I also used `min_samples_split = 50`, which means a branch would only split if it had enough samples.
The `random_state = 42` value was used to make the results reproducible.

```
max_depth = 5
min_samples_split = 50

model_dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

model_dt.fit(X_train, y_train)

print("Decision Tree model trained successfully")
```

The two models were then evaluated and compared.
The Linear Regression model acted as the simple baseline and the Decision Tree Regressor was used as the complex model and is later improved using optimisation.

### Model Evaluation

The models were evaluated using four regression metrics: R2, MAE, MSE and RMSE.
These were calculated on the training, validation and test sets so I could check how well each model performed on data it had trained on, data used during optimisation and unseen data.

I used R2 to see how much of the bike rental count the model could explain.
MAE showed the average prediction error.
MSE was also used because it gives bigger errors more importance.
RMSE was useful because it shows the error in the same unit as the target variable.

```
def regression_performance(X_train, y_train, X_val, y_val, X_test, y_test, pipeline):
    print("Model Evaluation \n")
    print("* Train Set")
    regression_evaluation(X_train, y_train, pipeline)
    print("* Validation Set")
    regression_evaluation(X_val, y_val, pipeline)
    print("* Test Set")
    regression_evaluation(X_test, y_test, pipeline)

def regression_evaluation(X, y, pipeline):
    prediction = pipeline.predict(X)
    print('R2 Score:', round(r2_score(y, prediction), 3)) (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
    print('Mean Absolute Error:', round(mean_absolute_error(y, prediction), 3)) (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
    print('Mean Squared Error:', round(mean_squared_error(y, prediction), 3)) (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
    print('Root Mean Squared Error:', round(np.sqrt(mean_squared_error(y, prediction)), 3)) (https://www.statology.org/how-to-interpret-rmse/)
    print("\n")
```
```
print("Linear Regression")
regression_performance(X_train, y_train, X_val, y_val, X_test, y_test, model_lr)

print("Decision Tree Regressor")
regression_performance(X_train, y_train, X_val, y_val, X_test, y_test, model_dt)
```

The first comparison showed that the Decision Tree Regressor performed better than Linear Regression on the test set.
Linear Regression had a test R2 score of 0.403 and a test RMSE of 141.130.
The first Decision Tree Regressor had a higher test R2 score of 0.645 and a lower test RMSE of 108.845.
This showed that the Decision Tree model made more accurate predictions than the Linear Regression baseline.

**Linear Regression**
**Model Evaluation Results** 

Train Set
- R2 Score: 0.387
- Mean Absolute Error: 105.777
- Mean Squared Error: 20058.751
- Root Mean Squared Error: 141.629

Validation Set
- R2 Score: 0.377
- Mean Absolute Error: 107.847
- Mean Squared Error: 20577.276
- Root Mean Squared Error: 143.448

Test Set
- R2 Score: 0.403
- Mean Absolute Error: 104.379
- Mean Squared Error: 19919.936
- Root Mean Squared Error: 141.138

**Decision Tree Regressor**
**Model Evaluation Results**

Train Set
- R2 Score: 0.651
- Mean Absolute Error: 70.471
- Mean Squared Error: 11435.263
- Root Mean Squared Error: 106.936

Validation Set
- R2 Score: 0.644
- Mean Absolute Error: 72.507
- Mean Squared Error: 11762.399
- Root Mean Squared Error: 108.455

Test Set
- R2 Score: 0.645
- Mean Absolute Error: 72.213
- Mean Squared Error: 11847.217
- Root Mean Squared Error: 108.845

I also used actual vs predicted plots to visually evaluate the models.
The red line represents perfect predictions, predicted value that match the actual value lie on this line.
Points closer to the red line show better predictions and points further away show large errors.

For Linear Regression, the plots showed that the model struggled to predict higher bike rental counts.
This scould suggest that Linear Regression was too simple for the patterns in the dataset.

For the Decision Tree Regressor, the predictions were closer overall and the test results were better.
However, the plots also showed horizontal bands.
This happens because a decision tree makes predictions by grouping data into leaf nodes, so multiple records receive the same predicted value. (https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/)

```
def regression_evaluation_plots(X_train, y_train, X_val, y_val, X_test, y_test, pipeline, alpha_scatter=0.5):
    pred_train = pipeline.predict(X_train).reshape(-1)
    pred_val = pipeline.predict(X_val).reshape(-1)
    pred_test = pipeline.predict(X_test).reshape(-1)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6)) (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)

    sns.scatterplot(x=y_train, y=pred_train, alpha=alpha_scatter, ax=axes[0]) (https://www.geeksforgeeks.org/python/scatterplot-using-seaborn-in-python/)
    sns.lineplot(x=y_train, y=y_train, color='red', ax=axes[0]) (https://seaborn.pydata.org/generated/seaborn.lineplot.html)
    axes[0].set_xlabel("Actual") # x axis label
    axes[0].set_ylabel("Predictions") # y axis label
    axes[0].set_title("Train Set") # plot title

    sns.scatterplot(x=y_val, y=pred_val, alpha=alpha_scatter, ax=axes[1])
    sns.lineplot(x=y_val, y=y_val, color='red', ax=axes[1])
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predictions")
    axes[1].set_title("Validation Set")

    sns.scatterplot(x=y_test, y=pred_test, alpha=alpha_scatter, ax=axes[2])
    sns.lineplot(x=y_test, y=y_test, color='red', ax=axes[2])
    axes[2].set_xlabel("Actual")
    axes[2].set_ylabel("Predictions")
    axes[2].set_title("Test Set")

    plt.show()
```
```
print("Linear Regression")
regression_evaluation_plots(X_train, y_train, X_val, y_val, X_test, y_test, model_lr)

print("Decision Tree Regressor")
regression_evaluation_plots(X_train, y_train, X_val, y_val, X_test, y_test, model_dt)
```

### Hyperparameter Optimisation

After the first Decision Tree Regressor was evaluated, I improved it using hyperparameter optimisation.
The hyperparameter I focused on was `max_depth`, which controls how deep the decision tree is allowed to grow.
A tree with a very small `max_depth` may be too simple and a tree with a very large `max_depth` may become too specific to the training data and overfit.
To test this, I trained Decision Tree models using `max_depth` values from 1 to 20.

```
def max_depth_error(md):
    model = DecisionTreeRegressor(max_depth=md, random_state=42)
    model.fit(X_train, y_train)
    train_error = 1 - model.score(X_train, y_train)
    val_error = 1 - model.score(X_val, y_val)
    return {'Max Depth': md, 'Training Error': train_error, 'Validation Error': val_error}
```
I created a function to test one max_depth value at a time.
Each time, the model was trained on the training set and then checked on both the training and validation sets.
I used `1 - model.score()` because `.score()` gives the R2 score, so this gave me an error value that was easy to compare

I then used a loop to test max_depth values from 1 to 20.
The results were stored in a list and converted into a DataFrame so I could view them as a table.

I plotted the training and validation errors to see how the model changed as max_depth increased.
The validation error was lowest at max_depth = 13, so I chose this as the best value.
After this point, the validation error started to increase slightly, which means the model was starting to overfit.

```
errors_list = []

for md in range(1, 21):
    result = max_depth_error(md)
    errors_list.append(result)

errors_df = pd.DataFrame(errors_list)
errors_df
```
```
plt.figure()
plt.scatter(errors_df['Max Depth'], errors_df['Training Error'])
plt.plot(errors_df['Max Depth'], errors_df['Validation Error'])
plt.title('Training vs Validation Error for Decision Tree Max Depth')
plt.xticks(range(0, 21, 2))
plt.xlabel('Max Depth')
plt.ylabel('Prediction Error')
plt.legend(['Training', 'Validation'])
plt.show()
```

The original Decision Tree model used `max_depth = 5`, which was too shallow.
I then retrained the Decision Tree Regressor with the optimised value of `max_depth = 13`.

```
max_depth = best_depth # lowest validation error found at max depth 13

model_dt_v2 = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

model_dt_v2.fit(X_train, y_train)

print("Optimised Decision Tree model trained successfully")
```

The optimised Decision Tree Regressor was then evaluated again on the training, validation and test sets.

```
print("Optimised Decision Tree Regressor")
regression_performance(X_train, y_train, X_val, y_val, X_test, y_test, model_dt_v2)
```

The optimised Decision Tree Regressor gave the best final performance.

**Optimised Decision Tree Regressor**
**Model Evaluation Results**

Train Set
- R2 Score: 0.93
- Mean Absolute Error: 29.501
- Mean Squared Error: 2274.898
- Root Mean Squared Error: 47.696

Validation Set
- R2 Score: 0.892
- Mean Absolute Error: 35.141
- Mean Squared Error: 3565.352
- Root Mean Squared Error: 59.711

Test Set
R2 Score: 0.902
Mean Absolute Error: 34.852
Mean Squared Error: 3281.879
Root Mean Squared Error: 57.288

### Prediction

The prediction section tested the model on data that had not been used for training.
I first made predictions using the first Decision Tree Regressor before optimisation.

```
pred_test = model_dt.predict(X_test)
pred_test
```
```
print("Actual vs Predicted:")
for i in range(10):
    print(f"Actual: {y_test.iloc[i]} | Predicted: {round(pred_test[i], 1)}")
```

The first Decision Tree model made some reasonable predictions, especially for lower bike rental counts.
For example, it predicted 6.5 when the actual value was 5.
However, it struggled more with higher bike rental counts.
For example, it predicted 325.6 when the actual value was 743, and 529.2 when the actual value was 925.
This showed that the first Decision Tree model had some good predictive ability but still needed improvement.

Actual vs Predicted:
Actual: 7 | Predicted: 14.4
Actual: 5 | Predicted: 6.5
Actual: 743 | Predicted: 325.6
Actual: 208 | Predicted: 198.5
Actual: 333 | Predicted: 238.0
Actual: 187 | Predicted: 120.0
Actual: 124 | Predicted: 325.6
Actual: 925 | Predicted: 529.2
Actual: 212 | Predicted: 349.9
Actual: 161 | Predicted: 120.0

I also tested the model on a single individual input.
This input represented one possible bike rental situation with season, year, month, hour, weather, temperature, humidity and windspeed.
The input was converted into a DataFrame so it had the same structure as the training data.

```
bike_input = {
    'season': [1],       # winter
    'yr': [1],           # year 2
    'mnth': [1],         # january
    'hr': [8],           # 8am
    'holiday': [0],      # not a holiday
    'weekday': [1],      # monday
    'workingday': [1],   # working day
    'weathersit': [1],   # clear weather
    'temp': [0.24],      # temperature
    'atemp': [0.2879],   # feels like temperature
    'hum': [0.81],       # humidity
    'windspeed': [0.0]   # windspeed
}

sample = pd.DataFrame(bike_input)
sample
```

The sample was then scaled using the same preprocessing pipeline that had been fitted on the training data.
This is important because the models were trained on scaled data, so the individual input also needed to be scaled before prediction.

```
sample_scaled = pipeline.transform(sample)

prediction = model_dt.predict(sample_scaled)
print(f"Predicted bike rentals: {round(prediction[0], 0)}")
```

After hyperparameter optimisation, I repeated the prediction process using the final optimised Decision Tree model.
This model used `max_depth = 13`.

```
pred_test_v2 = model_dt_v2.predict(X_test)

print("Actual vs Predicted (Optimised Model):")
for i in range(10):
    print(f"Actual: {y_test.iloc[i]} | Predicted: {round(pred_test_v2[i], 1)}")
```

The optimised model improved several predictions.
For example, the first model predicted 529.2 when the actual value was 925, while the optimised model predicted 871.7.
It also improved the prediction for an actual value of 124, changing from 325.6 in the first model to 127.6 in the optimised model.
This showed that the optimised model was better at predicting unseen test data.

Actual vs Predicted (Optimised Model):
Actual: 7 | Predicted: 7.5
Actual: 5 | Predicted: 4.5
Actual: 743 | Predicted: 375.3
Actual: 208 | Predicted: 205.1
Actual: 333 | Predicted: 291.2
Actual: 187 | Predicted: 201.2
Actual: 124 | Predicted: 127.6
Actual: 925 | Predicted: 871.7
Actual: 212 | Predicted: 199.3
Actual: 161 | Predicted: 114.7

I also repeated the individual input prediction using the optimised model.

```
sample_scaled = pipeline.transform(sample)
prediction_v2 = model_dt_v2.predict(sample_scaled)
print(f"Predicted bike rentals: {round(prediction_v2[0], 0)}")
```

Finally, I saved and compared all three models using the test set.
I calculated R2 and RMSE for Linear Regression, the first Decision Tree Regressor and the optimised Decision Tree Regressor.

```
pred_test_lr = model_lr.predict(X_test)
pred_test_dt = model_dt.predict(X_test)
pred_test_dt_v2 = model_dt_v2.predict(X_test)

r2_lr = r2_score(y_test, pred_test_lr)
r2_dt = r2_score(y_test, pred_test_dt)
r2_dt_v2 = r2_score(y_test, pred_test_dt_v2)

rmse_lr = np.sqrt(mean_squared_error(y_test, pred_test_lr))
rmse_dt = np.sqrt(mean_squared_error(y_test, pred_test_dt))
rmse_dt_v2 = np.sqrt(mean_squared_error(y_test, pred_test_dt_v2))

comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Optimised Decision Tree'],
    'Test R2': [r2_lr, r2_dt, r2_dt_v2],
    'Test RMSE': [rmse_lr, rmse_dt, rmse_dt_v2]
})

comparison_df
```

The final comparison showed that the optimised Decision Tree Regressor performed the best.
For R2, the best possible score is 1.0, so the model with the highest R2 was the best.
For RMSE, the best score is the lowest value, so the model with the lowest RMSE was the best.
The optimised Decision Tree had the highest R2 and the lowest RMSE, so it was selected as the final model.

Model	Test R2	Test RMSE
0	Linear Regression	0.403043	141.138003
1	Decision Tree	0.644965	108.844920
2	Optimised Decision Tree	0.901649	57.287688

==============================


==============================
## Jupyter Notebook Structure

[1] Notebook Objectives: Outlines the goals, inputs and outputs
- Inputs
- Outputs

[2] Data Collection: Downloads the dataset from Kaggle using the Kaggle API

[3] Import Libraries: Downloads all required python libraries and modules

[4] Data Loading and Validation: Loads the dataset and checks for missing values

[5] Exploratory Data Analysis (EDA): Explores the dataset using visulations and correlations

[6] Data Preparation: Drops columns that are not useful for predictions

[7] Split Dataset: splits the data into train, validation and test sets

[8] Data Preprocessing Pipeline: scales features using `StandardScaler`

[9] Model Building: builds and trains Linear Regression and Decision Tree Regressor
- Model 1: Linear Regression
- Model 2: Decision Tree Regressor

[10] Model Evaluation: evaluates both models using R2, MAE, MSE and RMSE

[11] Model Evaluation Plots: plots actual vs predicted for both models

[12] Optimisation of Hyperparameters: loops through max_depth values to find the best performing model

[13] Model Comparison: Compares the Linear Regression and the initial Decision Tree Regressor on the test set to select the current best
- Verdict

[14] Save Models: saves all trained models and pipeline to the models folder

[15] Prediction: tests the best model on unseen test data and individual input
- Test Set
- Test Set prediction results
[16] Individual Input

[17] Hyperparameter Optimisation - Model Imrpovement: rebuilds Decision Tree with optimised max_depth
- Analysis
[18] Rebuild Model
[19] Prediction on New Model: Runs predictions again on new optimised Decision Tree
- Overall

[19] Save Optimised model: saves newly trained model to the models folder

[20] Final Comparison Table: Comapres all trained models in the notebook on the test set R2 and RMSE to select the final best

[21] References: lists all references used throughout the notebook
==============================

==============================
## Future Work

* **More features:** additional data such as special event or public transport delays could improve predictions as these could also likely affect bike rental demand.
* **Other models:** a Neural Network or Random Forest model could be tested and compared against the Decision Tree to see if performance improves further
* **Larger dataset:** using more years of data beyond 2011-2012 could help the model predict better.
* **Better hyperparameter optimisation:** only `max_depth` was tuned in this project, so testing additional hyperparameters could further improve the Decision Tree model
* **Real time prediction:** the model could be published as a website to predict bike rental demand in real time based on current weather conditions
==============================

==============================
## Libraries and Modules

NumPy
* A numerical Python library used for mathematical operations.
* Used to calculate RMSE using `np.sqrt()`
* It provides fast array processing meaning it speeds up data handling, so its useful
* NumPy arrays are the foundation of most data processing in Python
* I used this throughout my pipeline wherever mathematical calculations were needed.
* (https://www.w3schools.com/python/numpy/numpy_intro.asp)

Pandas
* A data manipulation library used to load and explore the dataset.
* I used it to load the dataset with `pd.read_csv()`
* The functions `df.head()`, `df.info()` and `df.describe()` were used to inspect the data
* I also used it to create DataFrames for my individual input predictions section
* I used this throughout my data preparation and EDA stages
* (https://www.w3schools.com/python/pandas/default.asp)

Matplotlib
* A plotting library used to create visualisations in Python.
* I used this to create the hyperparameter optimisation plot.
* The plot showed training vs validation error across different `max_depth` values
* I used it with `plt.subplots()` to create side by side evaluation plots, to better visualise differences
* I used this throughout the model evaluation section
* (https://www.w3schools.com/python/matplotlib_intro.asp)

Seaborn
* A statistical visualisation library built on top of Matplotlib.
* I used to create the correlation heatmap in my EDA section
* I also used it to create scatter and line plots during the model evaluation section
* It provides cleaner and more visually appealing plots than just Matplotlib alone.
* It was used throughout my EDA and model evaluation sections
* (https://www.w3schools.com/python/numpy/numpy_random_seaborn.asp)

Plotly Express
* An interactive visualisation library used during my EDA section
* I used it to create histograms for `cnt` and `temp` distributions
* I also used it to create a scatter plot showing temperature vs bike rentals
* I also used it to create a bar chart showing records by season.
* All Plotly charts are interactive and can be zoomed and hovered over
* (https://www.geeksforgeeks.org/python/python-plotly-tutorial/)

Scikit-learn
* A machine learning library that provides tools for data processing, model building and model evaluation.
* I used this for train/test splitting, feature scaling, model building and evaluation.
* Modules used: `train_test_split`, `StandardScaler`, `Pipeline`, `LinearRegression`, `DecisionTreeRegressor`.
* Evaluation metrics used: `r2_score`, `mean_squared_error`, `mean_absolute_error`.
* The most important library used in this project.
* (https://www.geeksforgeeks.org/machine-learning/scikit-learn-tutorial/)

Joblib
* A library used to save and load Python objects.
* I used it to save trained models to the models folder using `joblib.dump()`.
* I used it to save the preprocessing pipeline so it can be reloaded without retraining.
* This ensures that the same scaling is applied to new inputs before prediction.
* Used in the two Save Models sections' of this notebook
* (https://www.geeksforgeeks.org/python/massively-speed-up-processing-using-joblib-in-python/)

==============================

==============================
## Unfixed Bugs

N/A, the notebook runs from top to bottom.
==============================

==============================
## Acknowledgements

[1] Dataset used: https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset

[2] The structure and formatting of this README was inspired by the README provided with my datatset by Hadi Fanaee-T and it was used to undertsnad my dataset.

[3] Beneath the RAEDME file of the Dataset used is this
"Use of this dataset in publications must be cited to the following publication:
[1] Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.

@article{
	year={2013},
	issn={2192-6352},
	journal={Progress in Artificial Intelligence},
	doi={10.1007/s13748-013-0040-3},
	title={Event labeling combining ensemble detectors and background knowledge},
	url={http://dx.doi.org/10.1007/s13748-013-0040-3},
	publisher={Springer Berlin Heidelberg},
	keywords={Event labeling; Event detection; Ensemble learning; Background knowledge},
	author={Fanaee-T, Hadi and Gama, Joao},
	pages={1-15}
}"

[4] AI Acknowledgement:
- Formatting my code so it follows the correct style for Python
- Reading error messages to help me understand why my code isnt working
- It made variable names for me that were readable and appropriate, as i struggle with coming up with educational names
- Getting feedback on my work
- For structuring my Jupyter notebook and README file for a better flow

[5] My code was written with guidance from the M32895 Big Data Applications module taught by Dr Sergey Yakovlev at the University of Portsmouth.
The code snippets used were from the following lessons: lessons 4.2, 5.1, 5.2, 6.1, 6.2, 8.1
All adapted code is referenced to the relevant lesson in markdowns
Module repository: https://github.com/sadiam09/m32895-tb2-2026

[6] The Malaria Detector Project by Kathrinmzl was used for help on pipeline structure: https://github.com/kathrinmzl/MalariaDetector/tree/main

## References

* (https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset)

* (https://www.kaggle.com/docs/api)

* (https://plotly.com/python-api-reference/generated/plotly.express.scatter)

* (https://plotly.com/python-api-reference/generated/plotly.express.histogram.html)

* (https://seaborn.pydata.org/generated/seaborn.heatmap.html)

* (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

* (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)

* (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

* (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)

* (https://www.statology.org/how-to-interpret-rmse/)

* (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)

* (https://www.geeksforgeeks.org/python/scatterplot-using-seaborn-in-python/)

* (https://seaborn.pydata.org/generated/seaborn.lineplot.html)

* (https://seaborn.pydata.org/generated/seaborn.lineplot.html)

* (https://www.bmc.com/blogs/mean-squared-error-r2-and-variance-in-regression-analysis/)

* (https://www.datacamp.com/tutorial/mean-absolute-error)

* (https://www.datacamp.com/tutorial/rmse)

* (https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html)

* (https://github.com/kathrinmzl/MalariaDetector/tree/main)

* (https://www.w3schools.com/python/numpy/numpy_intro.asp)

* (https://www.w3schools.com/python/pandas/default.asp)

* (https://www.w3schools.com/python/matplotlib_intro.asp)

* (https://www.w3schools.com/python/numpy/numpy_random_seaborn.asp)

* (https://www.geeksforgeeks.org/python/python-plotly-tutorial/)

* (https://www.geeksforgeeks.org/machine-learning/scikit-learn-tutorial/)

* (https://www.geeksforgeeks.org/python/massively-speed-up-processing-using-joblib-in-python/)
==============================

==============================
## Conclusion

* The ML pipelines successfully predicts hourly bike rental demand using the Bike Sharing dataset from Kaggle.
* Two models were built and compared using Linear Regression and Decision Tree Regressor
* The Decision Tree Regressor outperformed Linear Regression across all metrics on the test set.
* Hyperparameter optimisation improved the Decision Tree model more by changing the `max_depth = 13` as the optimal value using the validation set.
* The optimised Decision Tree achieved a test R2 score of 0.902. This means it explains 90.2% of the variance in bike rental demand
* The final model imrpoved predictions for btoh low and high rental counts, however some high rental counts were still harder to get an accurate prediction
* Future improvements could be testing additional models, tuning more hyperparameters, more features and using a larger dataset.
==============================
