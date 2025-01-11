# Summary 

<img width="1335" alt="Screenshot 2025-01-06 at 10 29 36 PM" src="https://github.com/user-attachments/assets/7b5e242a-31cb-4fb8-b9d7-1aedfc2eedeb" />

* This project aims to build a model that predict shooting success rate, given information about shooting attempts on NBA game. 

## Key findings

* Compared Logistic Regression, XGBoost and Neural Network 
  * In terms of validation metrics - accuracy, precision, recall, and ROC AUC - difference were not notable. 

  * Neural Network shoed slightly better scores, (especially AUC)

### Outcome

* Mostly, shot success probability was well-predicted

### Problem

* Model have difficulty with predicting success of 3-point shots
* It turns out that actual 3-point shots success rate in current NBA league is higher than what model expects (model under-predicts success rates of 3-point models.)

### Lesson learned and conclusion

* With 'Shooting' related data only, it is very hard to predict success rate for 3-point shots better than random guess. 
* Seems like population variability of 3-point shot itself' is too high (Which means, only with shooting related information, the best guess we could make is close to 'random' for 3-point shot.) - We need more external information (new features) to reduce this 'irreducible error'.

### Future improvements & Limitations

* This time, I used entire historical data to generate player stats - which could cause high variance. To reduce variability (according to Central Limit Theorum), using average of 'game-wise' statistic could reduce variability. 
* It turns out to be, variance for many features for each individual players are very high. Trying player-wise preidction, or grouping player groups (more detailed than just positions, but based on playtime, frequent actions etc) could help. 
* Since this is just a personal project, due to limitation of computing resources, each model parameters / hyper parameters could have not fully optimized. 

![image](https://github.com/user-attachments/assets/f1012f61-997c-4664-9e72-6f4e8d49cf62)

# Motivation

I personally am a Clippers fan, so I subscribe a service within NBA subscription called 'Clipper Vision'. It allow NBA fans to follow up all Clippers games through a dedicated streaming. I had discovered one interesting feature called **'real-time shooting percentages'** - when players carry the ball, or tries to shoot, it shows real time probability of the player making the shot successful.

![image](https://github.com/user-attachments/assets/35d4d7bb-c9fd-48d5-8dcb-3bd80cf692c9)

(Image from Clippers CourtvVision)

Well, each of the shots could be quantified - positions could be recorded as coordinates, and we have all the palyer stats, or distnace from the hoop. If I can build a model like this, I may be able to simulate players' shot by selecting a player, selecting shot type and position to predict shot success rate even when it's not during the game.

## Project outline: Building an app which predicts 'real-time shooting percentages'

<img width="1335" alt="Screenshot 2025-01-06 at 10 29 36 PM" src="https://github.com/user-attachments/assets/7b5e242a-31cb-4fb8-b9d7-1aedfc2eedeb" />

### The project will follow the steps below:

1. Process data and build a model that can predict shooting success rate. I will test one three candidates of Binary Classification models, which target prediction of 'success prob ability' and use 'threshold' for classification. (I will take the percentage as an outcome, but use classification label for validation). Three candidates are:
   * Logistic regression.
   * Tree based model (Boosting).
   * Neural Network.
2. Train the model with the model with best validation outcome.
3. Build interactive UI that takes player & shooting info as an input. This app with simple UI will return prediction of shooting success rate, taking this information as input data and using pre-trained model.

### Objective setting & assumptions I make for this project.

Throughout several experiments, I realized that it would not be realistic to target high evaluation metric score (accuracy, precision, recall and ROC score) for this project. Well, while watching NBA games, I found that even prediction provided by the ClipperVision service shows missing predictions quite frequently.

I assume this is because even the best prediction for shooting success rate can only tells about **'how tough is each of the shot in general'** - NBA is a sports, and among many different sports there are many amazing moves and highlights that breaks everyone's expectations. And star NBA players are ones who make 'tough shots'.

So, in terms of the business impact of this service, rather than precisely predicting result of each shot, I think it's more important to give insight of how is each of the shots were 'tough shot' - so audiences could more interactively react to each of player's move.

For instance, it could be a more of 'wow' moment for Clippers fan if James Harden makes tough deep-three pointer shot, which is marked as 23%, rather than when he makes easy open lay-up shot that has been marked as 95%.

Therefore, my objective for this project will be to aim

* 'Moderate' evaluation metric (accuracy, precision, ROC).
* Giving an 'explainable' result of prediction.

---

# Preprocessing data and features

Data source: https://github.com/DomSamangy/NBA_Shots_04_24

I used the above dataset which includes features as following (the data description is from the source above):

- Self-Explanatory
  - TEAM_NAME, PLAYER_NAME, POSITION_GROUP, POSITION, HOME_TEAM, AWAY_TEAM
- SEASON_1 & SEASON_2: Season indicator variables
- TEAM_ID: NBA's unique ID variable of that specific team in their API.
- PLAYER_ID: NBA's unique ID variable of that specific player in their API.
- GAME_DATE: Date of the game (M-D-Y // Month-Date-Year).
- GAME_ID: NBA's unique ID variable of that specific game in their API.
- EVENT_TYPE: Character variable denoting a shot outcome (Made Shot // Missed Shot).
- SHOT_MADE: True/False variable denoting a shot outcome (True // False).
- ACTION_TYPE: Description of shot type (layup, dunk, jump shot, etc.).
- SHOT_TYPE: Type of shot (2PT or 3PT).
- BASIC_ZONE: Name of the court zone the shot took place in.
  - Restricted Area, In the Paint (non-RA), Midrange, Left Corner 3, Right Corner 3, Above the Break, Backcourt.
- ZONE_NAME: Name of the side of court the shot took place in.
  - left, left side center, center, right side center, right
- ZONE_ABB: Abbreviation of the side of court.
  - (L), (LC), (C), (RC), (R).
- ZONE_RANGE: Distance range of shot by zones.
  - Less than 8 ft., 8-16 ft. 16-24 ft. 24+ ft.
- LOC_X: X coordinate of the shot in the x, y plane of the court (0, 50).
- LOC_Y: Y coordinate of the shot in the x, y plane of the court (0, 50).
- SHOT_DISTANCE: Distance of the shot with respect to the center of the hoop, in feet.
- QUARTER: Quarter of the game.
- MINS_LEFT: Minutes remaining in the quarter.
- SECS_LEFT: Seconds remaining in minute of the quarter.

## Removed redundant categorical features.

Each of the categorical features were one-hot encoded, and numerical features are scaled with Standard Scaler. However, since there are numerous categorical features, 'redundant' information were removed - for instance, 'Zone name' and 'Zone ABB' shows 1.0 correlation, because they are basically identical information, with different way of expressing.

## Processing 'Team' and Player name' Information.

I will only predict shot success probability based on the conditions related to the games, and individual capability of the players. **Which means, I will not use 'TEAM' information.** There are two reasons.

1. Shot success rate prediction should be more focused on 'the moment'. For instance, Golden State Warriors could have high 3 pt success rate because of Stephen Curry.  This does not mean that any other players in GSW should have advantage when computing probability of shot success. In terms of strategy, maybe taking 'team' information coule be meaningful. In 2010's GSW also lead the strategy of making many 3pt shots, playing actively outside the box and 3 point lines, so maybe including team information might reflect information abouth their strategy - but, for this project I will only consider information about the player himself, and circumstances at the moment.

   If we are using regression based model (e.g. Logistic regression), meaningful feaeture should add information helpful for prediction of the target. Here, we are asumming that knowing which team is the player in does not help predict his shot success rate.

2. Like player name, encoding teams into categorical data will increase number of features to large extent, increasing complexity of the model.

### Problem of 'Player name' feature

'Who is the player that made a shot?', is very important feature. However, it is very difficult to put it as categorical values. So, let's think about, 'why' is the player name important. **When we see James Harden shoots 3 pointer, know that 'historically', since he made many 3 pointer shots,** he is the one wh's good at shot. So, it's all about historical data of each player's shoot success.

So, I would like to

1. Make ave_soot_stat dictionary for all players.
2. Replace player's name with their ave shoot success rate.
3. When predicting, we can refer to the dictionary, to use that player's avg_shoot_stat.

This is the code I used to process avg_shoot_stat feature:

```python
players = np.unique(df[["PLAYER_ID"]])
player_goal_stat = {}

for p in players:
    temp = df.loc[df["PLAYER_ID"] == p]["SHOT_MADE"]
    avg = round(sum(temp.values) / temp.shape[0], 2)
    player_goal_stat[p] = avg
```

---

# Modeling

## Logistic Regression

![image](https://github.com/user-attachments/assets/8e564237-9e22-4d95-8613-194dc04433a8)

With Logistic Regression model, it showed ROC AUC slightly better than random guess (0.61), accuracy and precision around 0.60 + and very low recall (0.39). 

This is not a good score for 'classification' purpose, but considering that NBA shot probability have high variability (since it's a sports), and our objective is to get 'probability' prediction, explainability of the predicted outcome is more important.

Therefore, if 'predicted probability' of failed shots are clearly lower than predicted probability of successfult shots (for validation set), we could say the model is working fine.

![image](https://github.com/user-attachments/assets/862fe392-e585-4b7d-a8b9-deb008846b14)

* The 'average' predicted probability of failed shots are lower than average predicted probability of successful shot attempts. 
* From the distribution of extimated probabilities, we could see that successful shots result in higher predicted probability.
* However, if we see the distribution of extimated probabilities, we could see that around 0.3~0.4 range of predicted probability, both number of made shots and failed shots are high 
* This implies that the model is underpredicting probability of some of shots, **that were actually successful** in this probability range.
* We could see a lot of 'humps' for successfult shots' predicted probability - implying that some of the prediction of probability might be less consistent, or intuitive.

### Parameter optimization

With GridCV, we could proceed parameter optimization. 

```python 
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression

# Logistic Regression with hyperparameter tuning
param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.1, 0.01, 0.001, 0.0001],
    "solver": ["liblinear"],
}

logistic_regression = LogisticRegression()

grid_search = GridSearchCV(
    estimator=logistic_regression,
    param_grid=param_grid,
    scoring="neg_log_loss",  # Optimize for Log Loss
    cv=5,  # 5-fold cross-validation
    verbose=1,
)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_transformed, y, test_size=0.3, random_state=20
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best model and hyperparameters
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate the best model
y_prob = best_model.predict_proba(X_val)[:, 1]
logloss = log_loss(y_val, y_prob)
print(f"Optimized Log Loss: {logloss}")
```



### Inclusion - Exclusion test with R for feature coefficient significance

### Methodology

Using R script, I have proceeded inclusion-exclusion test to inspect statistical significance of each feature coefficients. For **multiple logistic regression,** log-odd function that represents probability of the class being 1 is defined as:
$$
Pr(Y=1 | X) = \frac{\exp(\beta_0 + \beta_1X_1 + ...\beta_nX_n)}{1 + \exp(\beta_0 + \beta_1X_1...\beta_nX_n)}
$$
Here, if certain variable $X_i$ is not significant for the model, we can set a null hypothesis:
$$
H_0 : \beta_i = 0
$$
For logistic regression model, we get 'deviance' as an evaluation metric:
$$
Deviance = -2log(L)
$$
where $log(L)$ is log-likelihood. Therefore minimizing deviance maximizes log-liklihood, implying that the model is better fit for the data.

Therefore, 

1. By evaluating change in deviance from 'full model', and 'model that excludes $X_i$'
2. And evaluating if the difference is 'significantly large'

We could rather reject null, or keep null hypothesis. 
$$
\Delta D = \text{Deviance}_{excluding_{X_i}} - \text{Deviance}_{full}
$$
Here, change of deviance follows Chi-square distribution.
$$
\Delta D \sim \chi^2
$$
Therefore, we could get P-value :
$$
p = 1-\chi^2_{(\Delta D, \Delta df)} \text{Where }df \text{ is defree of freedm.}
$$

### R-Script

```R
library(ggplot2)

data <- read.csv("aggregated_data_for_R_processed.csv")

out <- glm(TARGET ~ ., data = data, family = binomial(link = "logit"))
summary(out)

### Only numerical data
data <- read.csv("numerical_only.csv")
out_numerical <- glm(TARGET ~ ., data = data, family = binomial(link = "logit"))
summary(out_numerical)

### Comparing full and numerical model
delta_df <- out_numerical@@0@@df.resid
delta_dev <- out_numerical@@1@@dev

print('P-value:')
1-pchisq(delta_dev, df = delta_df)
# Null hypothesis: categorical variable coefficients are zero.
# Reject null : Deviance is not zero. Significant difference in deviance. Full model deviance is lower.

# Adding categorical data 1 by 1
data <- read.csv("adding_SEASON_2.csv")
out_adding_SEASON_2<- glm(TARGET ~ ., data = data, family = binomial(link = "logit"))
summary(out_adding_SEASON_2)

delta_df <- out_numerical@@2@@df.resid
delta_dev <- out_numerical@@3@@dev

print('P-value:')
1-pchisq(delta_dev, df = delta_df)
# We need to include season variable

data <- read.csv("adding_ZONE_ABB.csv")
out_adding_ZONE_ABB<- glm(TARGET ~ ., data = data, family = binomial(link = "logit"))
summary(out_adding_ZONE_ABB)

delta_df <- out_numerical@@4@@df.resid
delta_dev <- out_numerical@@5@@dev

print('P-value:')
1-pchisq(delta_dev, df = delta_df)

# >> Include Zone ABB

data <- read.csv("adding_ACTION_TYPE.csv")
adding_ACTION_TYPE<- glm(TARGET ~ ., data = data, family = binomial(link = "logit"))
summary(adding_ACTION_TYPE)

delta_df <- out_numerical@@6@@df.resid
delta_dev <- out_numerical@@7@@dev

print('P-value:')
1-pchisq(delta_dev, df = delta_df)
# >> Include

data <- read.csv("adding_SHOT_TYPE.csv")
adding_SHOT_TYPE<- glm(TARGET ~ ., data = data, family = binomial(link = "logit"))
summary(adding_SHOT_TYPE)

delta_df <- out_numerical@@8@@df.resid
delta_dev <- out_numerical@@9@@dev

print('P-value:')
1-pchisq(delta_dev, df = delta_df)
# >> Include

data <- read.csv("adding_BASIC_ZONE.csv")
adding_BASIC_ZONE<- glm(TARGET ~ ., data = data, family = binomial(link = "logit"))
summary(adding_BASIC_ZONE)

delta_df <- out_numerical@@10@@df.resid
delta_dev <- out_numerical@@11@@dev

print('P-value:')
1-pchisq(delta_dev, df = delta_df)

###

data <- read.csv("adding_POSITION.csv")
adding_POSITION<- glm(TARGET ~ ., data = data, family = binomial(link = "logit"))
summary(adding_POSITION)

delta_df <- out_numerical@@12@@df.resid
delta_dev <- out_numerical@@13@@dev

print('P-value:')
1-pchisq(delta_dev, df = delta_df)

###
data <- read.csv("adding_IS_PLAYER_HOME.csv")
adding_IS_PLAYER_HOME<- glm(TARGET ~ ., data = data, family = binomial(link = "logit"))
summary(adding_IS_PLAYER_HOME)

delta_df <- out_numerical@@14@@df.resid
delta_dev <- out_numerical@@15@@dev

print('P-value:')
1-pchisq(delta_dev, df = delta_df)

###
data <- read.csv("adding_ZONE_RANGE.csv")
adding_ZONE_RANGE<- glm(TARGET ~ ., data = data, family = binomial(link = "logit"))
summary(adding_ZONE_RANGE)

delta_df <- out_numerical@@16@@df.resid
delta_dev <- out_numerical@@17@@dev

print('P-value:')
1-pchisq(delta_dev, df = delta_df)
```

### Results

According to the inclusion-exclusion test results, all features for logistic regression was statistically valid (helpful decreasing deviance of model) under 95% confidence level.

## XGBoost

XGBoost, which is tree-based model could work better if there are 'non-linearity' in the patterns we are trying to predict - Logistic Regression is based in linear relationship between variables.

![image](https://github.com/user-attachments/assets/b3e0f938-01cf-4c6e-8606-655b184a5f22)

### Result comparison

In terms of classification metric with validation set, all of the metrics were slightly improves compared to logistic regression results, but not significant difference.

* One noticeable change in the distribution of predicted probabilities is the reduction in the pronounced 'humps' for successful shots.
* However, there remains an issue of underprediction in the 0.3–0.4 probability range. This suggests that ***shots within this probability range might share specific traits that make them particularly challenging for the model to predict accurately.***

## Hypothesis - 'Tough' shots

Here, I made an hypothesis that this underprediction issue is due to 'tough' shots.

* The model is well predicting success rate for most of the shots
* But, having difficulty for giving prediction for mid-low predicted probability range (0.2~0.5)
* This might imply that 'these shots are predicted to have low success rates, but actually being successful'
* **If there are some type of shots which are considered to be 'tough' shot attempts in common sense, but NBA players are actually skillful at it, it could explain this issue.**

---

# Analyzing reasons for the issue

I had analyzed and visualized several features that could be relevant to shot success rates. First I devided predicted 'shot success rate' value into 4 different zones:

* Very low (< 0.3)
* **Low (0.3~0.5)**
* High (0.5~0.7)
* Very high (>=0.7)

and try to find some inconsistent 'spikes' around 'Low' range (in which we are having issue.)

![image](https://github.com/user-attachments/assets/5c23532e-2e33-4388-8ac6-2e313604c621)

Most of the features showed results consistent with common sense - e.g. like we see from the first plot, as players' average shot success rate is high, the model predicts the shot attempt to be higher. 

However, the spike was detected from the 'shot distance' feature (plot 2)

<img width="770" alt="Screenshot 2025-01-07 at 5 55 57 PM" src="https://github.com/user-attachments/assets/aca8395d-2038-4489-959f-f22f47c08f5f" />

The data shows that, in general—for both made and failed shots—as shot distance increases (i.e., attempts from farther distances), the model tends to predict a lower success rate. However, there is a specific range of distances where the frequency of successful shots increases.

Additionally, a spike in the number of shot attempts was observed within this range, and a relatively higher proportion of successful shots was found compared to other distances.

## Semi-concluision: It's about 3 point shots.

<img width="814" alt="Screenshot 2025-01-07 at 5 57 19 PM" src="https://github.com/user-attachments/assets/d696af75-22d1-4f34-88d9-bf4b5b001cc8" />

Under the common sense, success rate of 3-point shots are considered to be low, but since more and more NBA players are becoming skillful in 3-point shots, it's not easy to predict success rate of 3-point shots with one model. After 2010s, under current NBA trend many teams and players are actively using 3-point play strategies - ***according to the [ShotTracker NBA statistics](https://shottracker.com/articles/the-3-point-revolution), 3 point shot attempts & success rate had been continuously increasing for over 20 years.***

For our own data, we could see that predicted probability of 3-point shots are much less distinguishable compared to 2 point shots - supporting our hypothesis.

# Trying Neural Network models

Neural networks are better at spotting more complicated features and sophisticated non-linear relationships. 

If the reason for having difficulty predicting probability around 0.3~0.7 (probably because of 3-point shot patterns), models better at complicated feature relationship could solve the issue. However, if it doesn't solve the issue, it may not be solved by changing the model - maybe we need different approach for 3-point shots, or ***'population variability'*** of 3-pt shots are too high, so it is hardly possible to predict 3-point shot success rate with predictive models. 

![image](https://github.com/user-attachments/assets/9753fc8b-b433-4484-af9c-d793464674d2)

Through Neural Net build with Keras, 4 layered model [64, 32, 16, 1] with 0.2 dropout rates, 32 batch size and 50 epochs, the metrics were almost in similar range with other two models. However, the result showed highest AUC (0.66) implying predicted probabilities for 0 and 1 labels are better splitted than other models. From the histogram, it showed much less overlapping area for made and failed shots than other two models. 

However, still under-prediction issue for 0.3~0.5 range exists. 

### Interaction terms. 

Adding interaction terms are one of the ways to address interrelated features. For instance, if as one $X_1$ variable increases, the impact of the other variable $X_2$ on targer variable $Y_2$increases, adding interaction term $X_1*X_2$ could address this. This is because for regression with interaction term like this:
$$
y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_3X_1X_2
$$
partial derivative for one variable would be:
$$
\frac{dy}{dX_1} = \beta_1 + \beta_3X_2
$$
Implying that, the impact of $X_1$ fixing all other variables constant is now not only $\beta_1$, but include term of $X_2$. I tried several interaction terms related to shooting range and 3pt shooting, for instance:

* SHOT_DISTANCE * 3PT_PLAYER_RATE_AVG
* SHOT_DISTANCE * PLAYER_RATE_AVG

While these approaches very slightly improve overall metric, ***it did not help solving the issue around 0.3~0.6 range predictions.*** 

# Conclusion after some of more efforts: Predicting 3pt shots are difficult to predict  - It's like a 'RANDOM GUESS'

## Is 3PT shot predictable enough?

If 3-point shot success rates are actually 'predictable' (of, 'explanable with some models' with historical data) the ***population variability (or irreducible error)*** should not be too big - which means, if irreducible error for the population data itself is too large, we cannot probably be able to predict the target. 

Based on the 'feature importances' for each of the models, I could see for most of the models, features as 'distance' or 'player avg 3pt rate' are very significant feature for model - which means, model count a lot on these features. We could see how these features are highly correlated with predicted probability:

![image](https://github.com/user-attachments/assets/fbddc31e-feb3-46d5-8cb8-49df84b65a8f)
![image](https://github.com/user-attachments/assets/d769b564-0d9e-4809-b546-62e1f9de57fc)

But, it never means that 3pt shots are predictable enough. For instance, if most of features are noisy both for 0 and 1 labels, it will cause high variability (irreducible error) causing any good models difficult to predict 3-pt shots. 

## Feature value distribution of successful / failed 3pt shots. 

 I first looked if feature values for successful / failed shot have meaningfully different distribution. First, I ran t-test to figure out which features show significant differences for successful and failed shots. Among more than 100 features (including one-hot encoded features), only 34 features showed P-value lower than 0.05, meaning their average value were significantly different for 0 and 1 labels. These are the followings. 

|      | PLAYER_AVG_3PT_RATE                    | -35.419002 | 5.170311e-274 |
| ---- | -------------------------------------- | ---------- | ------------- |
| 1    | SHOT_DISTANCE                          | 33.205393  | 3.566147e-241 |
| 2    | distance_sqiared                       | 31.073748  | 1.568393e-211 |
| 3    | ZONE_ABB_BC                            | 26.543489  | 5.364414e-155 |
| 4    | ZONE_RANGE_Back Court Shot             | 26.543489  | 5.364414e-155 |
| 5    | interaction_term_distance_2ptrate      | 26.295668  | 3.698803e-152 |
| 6    | interaction_term_2pt_3pt               | -25.755273 | 4.637047e-146 |
| 7    | BASIC_ZONE_Backcourt                   | 25.617697  | 1.580159e-144 |
| 8    | interaction_term_distance_playerrate   | 24.885282  | 1.660085e-136 |
| 9    | ZONE_RANGE_24+ ft.                     | -24.340283 | 1.094780e-130 |
| 10   | LOC_Y                                  | 16.452051  | 8.814953e-61  |
| 11   | SECS_LEFT                              | -10.708477 | 9.429729e-27  |
| 12   | BASIC_ZONE_Above the Break 3           | 10.476500  | 1.123193e-25  |
| 13   | ZONE_ABB_L                             | -10.225110 | 1.549996e-24  |
| 14   | BASIC_ZONE_Left Corner 3               | -10.163714 | 2.914679e-24  |
| 15   | BASIC_ZONE_Right Corner 3              | -9.711574  | 2.719340e-22  |
| 16   | ZONE_ABB_R                             | -9.654977  | 4.730009e-22  |
| 17   | ACTION_TYPE_Jump Shot                  | 9.008898   | 2.097361e-19  |
| 18   | QUARTER                                | 8.214586   | 2.140324e-16  |
| 19   | ACTION_TYPE_Jump Bank Shot             | -7.141329  | 9.271684e-13  |
| 20   | ACTION_TYPE_Running Jump Shot          | -6.250485  | 4.099251e-10  |
| 21   | POSITION_SG                            | -6.193580  | 5.891600e-10  |
| 22   | POSITION_C                             | 4.902756   | 9.456854e-07  |
| 23   | ACTION_TYPE_Pullup Jump shot           | -4.598011  | 4.267806e-06  |
| 24   | ZONE_ABB_C                             | 4.271420   | 1.943137e-05  |
| 25   | ACTION_TYPE_Step Back Jump shot        | -4.129266  | 3.640571e-05  |
| 26   | ACTION_TYPE_Driving Floating Jump Shot | 4.084281   | 4.422915e-05  |
| 27   | ZONE_ABB_LC                            | 3.577234   | 3.473243e-04  |
| 28   | PLAYER_AVG_SHOT_RATE                   | -3.347947  | 8.142614e-04  |
| 29   | ACTION_TYPE_Fadeaway Jump Shot         | 3.216672   | 1.297056e-03  |
| 30   | ACTION_TYPE_Turnaround Fadeaway shot   | 3.119271   | 1.813224e-03  |
| 31   | interaction_term_distance_3ptrate      | 2.943950   | 3.240863e-03  |
| 32   | ACTION_TYPE_Running Pull-Up Jump Shot  | -2.834318  | 4.592775e-03  |
| 33   | ZONE_ABB_RC                            | 2.406099   | 1.612473e-02  |
| 34   | ACTION_TYPE_Turnaround Jump Shot       | 2.193085   | 2.830231e-02  |

Intuitively, distance, zone, and player stat related features seem to have significant differences between succeddful and failed shots. Interestingly, for position information, 'Center' and 'Shooting Guard' only had significant differences for 3-point shot success rate. 

However, even though these features show significantly different mean values, it does not guarantee we could predict 3-point shot rate with these features. 

![image](https://github.com/user-attachments/assets/53a89116-d418-43c3-8cba-b9b3372874e0)

For instance, features as 'Shot distance' or 'avg 3 point rate' had difference in 'Mean' for 0 and 1 labels, but se could see that their variance are very high - implying that some NBA players are actually making many 3-point shots even though they are from very far distance, or their 3-point success rate used to be relatively low. 

## Maybe, 'population variability' of 3-point shots in NBA itself is too high to be predictable.

![image](https://github.com/user-attachments/assets/6de4aab7-d874-43ac-b582-155bbff62fb3)

This is result of training XGB & Logistic Regression model only with selected features above, which shoed significant mean differences between 0 and 1 labels. ***It is close to random guess, showing AUC near 0.5!*** 

3-point shot success rate for NBA league is hard to predict, because it's irreducible error might be very high. This make sense if we understand NBA league. Current 3pt shot average success rate is close to **39%** (even it's the time with highest 3pt success rate.). 

![image](https://github.com/user-attachments/assets/580242f6-c8da-40be-a967-6f8ebbdc3f32)We can see see **a lot of 3-point shot attempts (as wee saw spikes in our model result probability distributions above)**, but there are much more 'failed' 3-point shot samples. 

Are all of the players equally good at 3-point shots? Are most of the players even shooting (attempting) 3-point shots? ![image](https://github.com/user-attachments/assets/7c3a3e0e-9f6d-407a-9745-716fe19f00ec)

Well, we can see 3 point shot attemps per game of each players are very widely distributed, with high variance. Number of 3pt shots per game is little bit skewed, and also have high variance. 

![image](https://github.com/user-attachments/assets/a0c1e1f5-cfc4-41d5-9ad8-03cf2a4d291c)

Even number of games played per player is not balanced - because some of the members are running all-time for most of the games, while some members are mostly on bench, or running on garbage-times. 

### How could we improve prediction? - Further improvements

Therefore, further actions may have to focus on reducing variability related to '3-point shots'. 

1. **This time I used all historical data. Change them to game-wise metric (to reduce variance):**  For this project, I used 'shooting success rate' of players by using all historical data of each players - however, looking into high variance in total 3pt attempt, using 'average per game shooting success rate' metric may relieve variability.  (Due to central limit theorum, mean of random variable will have lower variance. )
2. **Trying 'player wise' prediction**: We can see variance for many metric is very high for each players - some are very good at 3-point shots, some are barely trying 3-point shots, while some are taking par in much more games then others. 'Player stats' may have to tell about each of the players' features, but the dataset I used are focused on features related to the 'shooting' itself. Therefore, trying to make prediction for each of the players may result in better prediction. 
   * This won't be realistic because we need number of models equal to number of NBA players. Therefore, we could do approcah to grouping players (e.g. by positions, by play types, whether they are bench member, garbage time members or not etc). 
