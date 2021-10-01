import pandas as pd

DATADIR = r"C:\Users\ameil\Documents\GitHub\Poisson Prediction"
teams=pd.DataFrame()
teams["Team"] = ['ARS','AVL','BHA','BUR','CHE','CRY','EVE','FUL','LEE','LEI','LIV','MCI','MUN','NEW','SHU','SOU','TOT','WBA','WHU','WOL',"BRE","WAT","NOR","TOT","LEI","MCI","MUN","SHU"]
teams["Full Name"]=["Arsenal","Aston Villa","Brighton","Burnley","Chelsea","Crystal Palace","Everton","Fulham","Leeds","Leicester City","Liverpool","Man City","Man Utd",
"Newcastle","Sheff Utd","Southampton","Tottenham","West Brom","West Ham","Wolves","Brentford","Watford","Norwich","Spurs","Leicester","Manchester City","Manchester Utd","Sheffield Utd"]
dict_team={k:v for k,v in zip(teams["Full Name"], teams["Team"])}