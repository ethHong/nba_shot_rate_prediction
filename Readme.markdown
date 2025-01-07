# Summary 

<img width="1335" alt="Screenshot 2025-01-06 at 10 29 36 PM" src="https://github.com/user-attachments/assets/7b5e242a-31cb-4fb8-b9d7-1aedfc2eedeb" />

* This project aims to build a model that predict shooting success rate, given information about shooting attempts on NBA game.
* So far, I am working on the model, but also thinking of building an interactive UI!

### Key findings

* Compared Logistic Regression, XGBoost and Neural Network to compare results of accuracy, precision, recall, and ROC AUC - differences in evaluation metric were not large, but Neural Network showed highest validation score.

* Even with parameter tuning & feature engineering, prediction showed limitation for certain probability range: 
  * Shot attempts with high probability prediction (0.5 to 1.0) were mostly successful shot, while low prediction (0 to 0.3) were mostly failed shots, implying that the prediction is working well.
  * However, the model prodiction for 0.3 to 0.5 range probability could not distinguish successful shots and failed shots.
  * This turned out to be 'shot distance' is one of the important feature (longer distance shots are less likely to be successful), but ***under current NBA trend number of attempts & success rate of 3pt shots are comparativelty higher.***

Under the common sense, success rate of 3-point shots are considered to be low, but since more and more NBA players are becoming skillful in 3-point shots, it's not easy to predict success rate of 3-point shots with one model. 

* Tried adding interaction term for 3-point shot & 3-point shot success rate of the player, but did not improve.

(For future improvements, approaches like training a model dedicated for 3-point shots can be helpful.)

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

# Data Source

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
