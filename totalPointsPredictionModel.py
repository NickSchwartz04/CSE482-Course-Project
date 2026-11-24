import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def main():
    get_logs()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(mean_absolute_error(y_test, predictions))



def get_logs():
    columns_to_keep = ['Team_ID', 'Game_ID', 'GAME_DATE', 'MATCHUP',]

    for team in teams.get_teams():
        team_id = team['id']
        team_name = team['full_name']

        gamelog = teamgamelog.TeamGameLog(team_id=team_id, season='2025-26').get_data_frames()[0]
        gamelog = gamelog[columns_to_keep]
        gamelog.to_csv(f'{team_name}_gamelog.csv', index=False)
