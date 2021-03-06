import numpy as np
from scipy.stats import poisson
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

from constants import dict_team

class Percent(float):
    def __str__(self):
        return '{:.2%}'.format(self)

def import_data(dict_team=dict_team):
    seasons = [2020, 2021]
    PL = pd.DataFrame()
    for season in seasons:
        add = pd.read_csv(os.getcwd()+f'\\PL Odds {season}-{season + 1}.csv')
        PL = pd.concat([PL, add])
        PL["Home Goals"] = [int(x.split(":")[0]) for x in PL["Score"]]
        PL["Away Goals"] = [int(x.split(":")[1]) for x in PL["Score"]]
    # Convert Date to DateTime
    PL['DateTime'] = PL["Date"] + " " + PL["Time"]
    PL['DateTime'] = pd.to_datetime(PL['Date'])
    PL.sort_values("DateTime", ascending=True, inplace=True)

    # Clean Team names
    PL["Home Team"] = PL["Home Team"].apply(lambda x: x.strip())
    PL["Away Team"] = PL["Away Team"].apply(lambda x: x.strip())
    PL.rename(columns={"Home Team": "HomeTeam", "Away Team": "AwayTeam"}, inplace=True)

    PL["HomeTeam"] = PL["HomeTeam"].map(dict_team)
    PL["AwayTeam"] = PL["AwayTeam"].map(dict_team)

    return PL

def score_matrix(dict,home,away):
    home_mean = dict["Home Goals"][home]
    away_mean = dict["Away Goals"][away] #todo

    goal_diff=[]
    odds=[]
    for x in np.arange(0,6):
        odd=poisson.pmf(x, home_mean)
        goal_diff.append(x)
        odds.append(odd)
    dict_home={k:v for (k,v) in zip(goal_diff,odds)}
    home_win_df=pd.DataFrame.from_dict(dict_home, orient='index')
    home_win_df.rename(columns={0:"Home"},inplace=True)

    goal_diff=[]
    odds=[]
    for x in np.arange(0,6):
        odd=poisson.pmf(x,away_mean)
        goal_diff.append(x)
        odds.append(odd)
    dict_away={k:v for (k,v) in zip(goal_diff,odds)}
    away_win_df=pd.DataFrame.from_dict(dict_away, orient='index')
    away_win_df.rename(columns={0:"Away"},inplace=True)


    home_odds=home_win_df["Home"].to_numpy()
    home_odds=np.tile(home_odds, (6, 1))
    away_odds=away_win_df["Away"].to_numpy()
    away_odds=np.tile(away_odds, (6, 1))
    away_odds=np.transpose(away_odds)
    joint_odds=np.multiply(home_odds,away_odds)*100

    upper_sum = np.triu(joint_odds).sum() - np.trace(joint_odds)
    lower_sum = np.tril(joint_odds).sum() - np.trace(joint_odds)
    draw=np.trace(joint_odds)

    ax = sns.heatmap(joint_odds, linewidth=0.5, annot=True)
    ax.set_title(f"{home} vs {away} -Poisson Prediction",fontsize=20)
    plt.xlabel(f"{home}", fontsize=15)  # x-axis label with fontsize 15
    plt.ylabel(f"{away}", fontsize=15)  # y-axis label with fontsize 15

    ax.text(6.7, 1.2, f"{home} win ={Percent(upper_sum/100)}\n{away} win ={Percent(lower_sum/100)}\nDraw ={Percent(draw/100)}", fontsize=14,
            verticalalignment='top',clip_on=False)
    return joint_odds

def outcomes(home,away,dict):
    home_mean = dict["Home Goals"][home]
    away_mean = dict["Away Goals"][away]

    goal_diff=[]
    odds=[]
    for x in np.arange(0,6):
        odd=poisson.pmf(x, home_mean)
        goal_diff.append(x)
        odds.append(odd)
    dict_home={k:v for (k,v) in zip(goal_diff,odds)}
    home_win_df=pd.DataFrame.from_dict(dict_home, orient='index')
    home_win_df.rename(columns={0:"Home"},inplace=True)

    goal_diff=[]
    odds=[]
    for x in np.arange(0,6):
        odd=poisson.pmf(x,away_mean)
        goal_diff.append(x)
        odds.append(odd)
    dict_away={k:v for (k,v) in zip(goal_diff,odds)}
    away_win_df=pd.DataFrame.from_dict(dict_away, orient='index')
    away_win_df.rename(columns={0:"Away"},inplace=True)


    home_odds=home_win_df["Home"].to_numpy()
    home_odds=np.tile(home_odds, (6, 1))
    away_odds=away_win_df["Away"].to_numpy()
    away_odds=np.tile(away_odds, (6, 1))
    away_odds=np.transpose(away_odds)
    joint_odds=np.multiply(home_odds,away_odds)*100

    upper_sum = np.triu(joint_odds).sum() - np.trace(joint_odds)
    lower_sum = np.tril(joint_odds).sum() - np.trace(joint_odds)
    draw=np.trace(joint_odds)

    if upper_sum==max(upper_sum,lower_sum,draw):
        predict=home
    elif lower_sum==max(upper_sum,lower_sum,draw):
        predict=away
    else:
        predict="Draw"

    return predict

def match_result(df):
    if df["Home Goals"]>df["Away Goals"]:
        return df["HomeTeam"]
    elif df["Home Goals"]<df["Away Goals"]:
        return df["AwayTeam"]
    else:
        return "Draw"

def dc_decay(xi, t):
    return np.exp(-xi * t)

def rho_correction(goals_home, goals_away, home_exp, away_exp, rho):
    if goals_home == 0 and goals_away == 0:
        return 1 - (home_exp * away_exp * rho)
    elif goals_home == 0 and goals_away == 1:
        return 1 + (home_exp * rho)
    elif goals_home == 1 and goals_away == 0:
        return 1 + (away_exp * rho)
    elif goals_home == 1 and goals_away == 1:
        return 1 - rho
    else:
        return 1.0


def log_likelihood(
    goals_home_observed,
    goals_away_observed,
    home_attack,
    home_defence,
    away_attack,
    away_defence,
    home_advantage,
    rho,
):
    goal_expectation_home = np.exp(home_attack + away_defence + home_advantage)
    goal_expectation_away = np.exp(away_attack + home_defence)

    home_llk = poisson.pmf(goals_home_observed, goal_expectation_home)
    away_llk = poisson.pmf(goals_away_observed, goal_expectation_away)
    adj_llk = rho_correction(
        goals_home_observed,
        goals_away_observed,
        goal_expectation_home,
        goal_expectation_away,
        rho,
    )

    if goal_expectation_home < 0 or goal_expectation_away < 0 or adj_llk < 0:
        return 10000
    if 0 in [home_llk,away_llk,adj_llk]:
        return 10000
    else:
        log_llk = np.log(home_llk) + np.log(away_llk) + np.log(adj_llk)
        return -log_llk

#xi value determined by trial and error to find the value that minimised rank score probability 
def fit_poisson_model(df):
    teams = np.sort(np.unique(np.concatenate([df["HomeTeam"], df["AwayTeam"]])))
    n_teams = len(teams)

    params = np.concatenate(
        (
            np.random.uniform(0.5, 1.5, (n_teams)),  # attack strength
            np.random.uniform(0, -1, (n_teams)),  # defence strength
            [0.25],  # home advantage
            [-0.1], # rho
        )
    )

    def _fit(params, df, teams):
        attack_params = dict(zip(teams, params[:n_teams]))
        defence_params = dict(zip(teams, params[n_teams : (2 * n_teams)]))
        home_advantage = params[-2]
        rho = params[-1]

        llk = list()
        for idx, row in df.iterrows():
            tmp = log_likelihood(
                row["Home Goals"],
                row["Away Goals"],
                attack_params[row["HomeTeam"]],
                defence_params[row["HomeTeam"]],
                attack_params[row["AwayTeam"]],
                defence_params[row["AwayTeam"]],
                home_advantage,
                rho
            )
            llk.append(tmp)

        return np.sum(llk)

    options = {
        "maxiter": 100,
        "disp": False,
    }

    constraints = [{"type": "eq", "fun": lambda x: sum(x[:n_teams]) - n_teams}]

    res = minimize(
        _fit,
        params,
        args=(df, teams),
        constraints=constraints,
        options=options,
    )

    model_params = dict(
        zip(
            ["attack_" + team for team in teams]
            + ["defence_" + team for team in teams]
            + ["home_adv", "rho"],
            res["x"],
        )
    )

    print("Log Likelihood: ", res["fun"])

    return model_params


def predict(params, home_team, away_team):
    home_attack = params["attack_" + home_team]
    home_defence = params["defence_" + home_team]
    away_attack = params["attack_" + away_team]
    away_defence = params["defence_" + away_team]
    home_advantage = params["home_adv"]
    rho = params["rho"]

    home_goal_expectation = np.exp(home_attack + away_defence + home_advantage)
    away_goal_expectation = np.exp(away_attack + home_defence)

    home_probs = poisson.pmf(range(10), home_goal_expectation)
    away_probs = poisson.pmf(range(10), away_goal_expectation)

    m = np.outer(home_probs, away_probs)

    m[0, 0] *= 1 - home_goal_expectation * away_goal_expectation * rho
    m[0, 1] *= 1 + home_goal_expectation * rho
    m[1, 0] *= 1 + away_goal_expectation * rho
    m[1, 1] *= 1 - rho

    home = np.sum(np.tril(m, -1))
    draw = np.sum(np.diag(m))
    away = np.sum(np.triu(m, 1))

    return home, draw, away

def dixon_coles_predict(params, home_team, away_team):
    outcomes=[home_team,"Draw",away_team]
    probabilities=predict(params, home_team, away_team)
    outcome_prob=max(probabilities)
    outcome=outcomes[probabilities.index(outcome_prob)]
    return [outcome,outcome_prob]

def dixon_coles_predict_H_A_D(params, home_team, away_team,show="home_team"):
    probabilities = predict(params, home_team, away_team)
    if show=="home_team":
        return probabilities[0]
    elif show=="away_team":
        return probabilities[2]
    else:
        return probabilities[1]


def home_draw_away(df):
    if df["Home Goals"]>df["Away Goals"]:
        return "Home"
    elif df["Home Goals"]<df["Away Goals"]:
        return "Away"
    else:
        return "Draw"


def bet_poisson(home_team,away_team,home_odds,away_odds,draw_odds,result,prediction):
    fixture={home_team:home_odds,away_team:away_odds,"Draw":draw_odds}
    if prediction==result:
        profit = fixture[result] - 1
        return profit
    else:
        profit=- 1
        return profit


def gw_predictions(gameweek,params,goal_dict):
    fixtures_df = pd.read_csv("PL Fixtures.csv")
    fixtures_df = fixtures_df[fixtures_df["Round Number"] == gameweek]
    fixtures_df["Home Team"] = fixtures_df["Home Team"].apply(lambda x: x.strip())
    fixtures_df["Away Team"] = fixtures_df["Away Team"].apply(lambda x: x.strip())
    fixtures_df.rename(columns={"Home Team": "HomeTeam", "Away Team": "AwayTeam"}, inplace=True)

    fixtures_df["HomeTeam"] = fixtures_df["HomeTeam"].map(dict_team)
    fixtures_df["AwayTeam"] = fixtures_df["AwayTeam"].map(dict_team)
    fixtures_df["Predict DC"] = fixtures_df.apply(
        lambda row: dixon_coles_predict(params=params, home_team=row["HomeTeam"], away_team=row["AwayTeam"])[0],
        axis=1)

    fixtures_df["Predict"] = fixtures_df.apply(
        lambda row: outcomes(home=row["HomeTeam"], away=row["AwayTeam"], dict=goal_dict), axis=1)
    fixtures_df["Home Win DC"] = fixtures_df.apply(
        lambda row: dixon_coles_predict_H_A_D(params=params, home_team=row["HomeTeam"], away_team=row["AwayTeam"],
                                              show="home_team"), axis=1)
    fixtures_df["Draw DC"] = fixtures_df.apply(
        lambda row: dixon_coles_predict_H_A_D(params=params, home_team=row["HomeTeam"], away_team=row["AwayTeam"],
                                              show="draw"), axis=1)
    fixtures_df["Away Win DC"] = fixtures_df.apply(
        lambda row: dixon_coles_predict_H_A_D(params=params, home_team=row["HomeTeam"], away_team=row["AwayTeam"],
                                              show="away_team"), axis=1)

    return fixtures_df