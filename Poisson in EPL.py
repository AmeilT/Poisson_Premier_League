import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson
from scipy.optimize import minimize
from pprint import pprint

from functions import * #for simplicity (bad practice)
from constants import DATADIR

#Import results data
seasons = [2020, 2021]
PL = pd.DataFrame()
for season in seasons:
    add = pd.read_csv(os.getcwd()+f'\\PL Odds {season}-{season + 1}.csv')
    PL = pd.concat([PL, add])
    PL["Home Goals"] = [int(x.split(":")[0]) for x in PL["Score"]]
    PL["Away Goals"] = [int(x.split(":")[1]) for x in PL["Score"]]

# DisPlot of Home vs Away
#We see a reduced number of away teams scoring 3+ goals (as expected). The home advantage is not large here due to the lack of fans in stadiums.
PL_copy=PL.append(PL)
PL_copy.reset_index(inplace=True)
PL_copy["Type"]=(["Home"]*PL.shape[0])+(["Away"]*PL.shape[0])
PL_copy.loc[PL_copy["Type"] == "Home", 'Goals']=PL_copy["Home Goals"]
PL_copy.loc[PL_copy["Type"] == "Away", 'Goals']=PL_copy["Away Goals"]
sns.displot(PL_copy,x="Goals",hue="Type",kde=True,multiple="dodge")

# Count matrix (div by 760 as each match is duped)
# z=PL_copy.pivot_table(index='Home Goals', columns='Away Goals', aggfunc=len).fillna(0).astype('int') alternate way of producing the count matrix
count_matrix=PL_copy.groupby(["Home Goals","Away Goals"]).size().unstack(fill_value=0)*100/760

#Create a Poisson model to predict outcomes
# Convert Date to DateTime
PL['DateTime'] = PL["Date"] + " " + PL["Time"]
PL['DateTime'] = pd.to_datetime(PL['Date'])
PL.sort_values("DateTime", ascending=True, inplace=True)

# Clean Team names
PL["Home Team"] = PL["Home Team"].apply(lambda x: x.strip())
PL["Away Team"] = PL["Away Team"].apply(lambda x: x.strip())

# Removing x number of days from training set 
remove_days = 20  # remove last 20 days
PL_learn = PL[:-remove_days]
fixtures_df = PL[-remove_days:]

# Calc mean home and away goals for each team
avg_goal_df_home = PL_learn.groupby("Home Team").mean().reset_index()
avg_goal_df_home.rename(columns={"Home Team": "Team"}, inplace=True)
avg_goal_df_away = PL_learn.groupby("Away Team").mean().reset_index()
avg_goal_df_away.rename(columns={"Away Team": "Team"}, inplace=True)

# Display Home and Away avg goals
# fig, axes = plt.subplots(2,1,figsize=(6, 8))
# ax=axes.flat
# ax[0].bar(x="Team",height="Home Goals",data=avg_goal_df_home,align="center")
# ax[1].bar(x="Team",height="Away Goals",data=avg_goal_df_away,align="center")
# ax[0].set_xticklabels(avg_goal_df_home["Team"],rotation = 45, ha="right")
# ax[1].set_xticklabels(avg_goal_df_away["Team"],rotation = 45, ha="right")

#Join the avg home and away goal data together
avg_goal_df = avg_goal_df_home[["Team", "Home Goals"]].merge(avg_goal_df_away[["Team", "Away Goals"]], on="Team")
avg_goal_df.set_index("Team", inplace=True)
avg_goal_dict = avg_goal_df.to_dict()

# Viz of score predictions
#the score_matrix uses a simple poisson.pmf function where the expected mean goals is the avg goals scored in the training set for the home and away team respectively.
#it then combines them to form a matrix of score predictions
#win loss or draw are sums of the upper triangle,lower triangle and diagonals
example=score_matrix(avg_goal_dict,"Manchester City","Everton")

# Predict future results
#Using the same basic poisson model we predict outcomes (highest probability outcome)
fixtures_df["Predict"] = fixtures_df.apply(lambda row: outcomes(home=row["Home Team"], away=row["Away Team"]), axis=1)

# Show actual result
fixtures_df["Result"] = fixtures_df.apply(match_result, axis=1)

# Expected goals scored = avg of previous goals scored. This is a simple model but we can leverage Dixon-Coles paper and improve our estimate of expected goals score.
#The original work comes from: http://www.pena.lt/y/2021/06/18/predicting-football-results-using-the-poisson-distribution/
#Now we use:
# goals_home = home_advantage + home_attack + defence_away
#goals_away = away_attack + defence_home
#To calculate the 5 variables above we maximise the log likeliood function (maximise the probability that the expected goals scored by the poisson model match those of reality)
# DIXON COLES
model_params = fit_poisson_model(PL_learn)
pprint(model_params)
#Organise Params
param_df = pd.DataFrame.from_dict(model_params, orient='index')
param_df_attack = param_df[:20]
param_df_defence = param_df[20:-2]
param_misc = param_df[-2:]
param_df_attack.sort_values(by=0, ascending=False, inplace=True)
param_df_defence.sort_values(by=0, ascending=False, inplace=True)

#Compare Prediction with results
fixtures_df["Predict DC"] = fixtures_df.apply(
    lambda row: dixon_coles_predict(params=model_params, home_team=row["Home Team"], away_team=row["Away Team"])[0],
    axis=1)
(fixtures_df["Result"] == fixtures_df["Predict"]).value_counts()
(fixtures_df["Result"] == fixtures_df["Predict DC"]).value_counts()

fixtures_df["Outcome"] = fixtures_df.apply(home_draw_away, axis=1)

#Use basic and DC models with betting odds to calculate potential strategy
fixtures_df["Profit Basic"] = fixtures_df.apply(
    lambda row: bet_poisson(home_team=row["Home Team"], away_team=row["Away Team"], home_odds=row["Home Odds"],
                            away_odds=row["Away Odds"], draw_odds=row["Draw Odds"], result=row["Result"],
                            prediction=row["Predict"]), axis=1)
fixtures_df["Profit DC"] = fixtures_df.apply(
    lambda row: bet_poisson(home_team=row["Home Team"], away_team=row["Away Team"], home_odds=row["Home Odds"],
                            away_odds=row["Away Odds"], draw_odds=row["Draw Odds"], result=row["Result"],
                            prediction=row["Predict DC"]), axis=1)

fixtures_df["Cummulative Profit Basic"] = fixtures_df["Profit Basic"].cumsum()
fixtures_df["Cummulative Profit DC"] = fixtures_df["Profit DC"].cumsum()
