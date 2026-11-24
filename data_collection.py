
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog


def get_logs():

    all_logs = []

    columns_to_keep = [
        'Team_ID',
        'Game_ID',
        'GAME_DATE',
        'MATCHUP',
        'PTS'
    ]

    for team in teams.get_teams():

        team_id = team['id']
        team_name = team['full_name']

        print(f"Collecting {team_name}...")

        gamelog = teamgamelog.TeamGameLog(team_id=team_id, season='2025-26').get_data_frames()[0]

        gamelog = gamelog[columns_to_keep].copy()
        gamelog['TEAM_NAME'] = team_name

        all_logs.append(gamelog)

    all_games = pd.concat(all_logs, ignore_index=True)

    home = all_games[all_games['MATCHUP'].str.contains('vs.')].copy()

    away = all_games[all_games['MATCHUP'].str.contains('@')].copy()

    games = pd.merge(home, away, on='Game_ID', suffixes=('_HOME', '_AWAY'))

    final_data = pd.DataFrame({
        'GAME_ID': games['Game_ID'],
        'GAME_DATE': games['GAME_DATE_HOME'],
        'HOME_TEAM': games['TEAM_NAME_HOME'],
        'AWAY_TEAM': games['TEAM_NAME_AWAY'],
        'HOME_POINTS': games['PTS_HOME'],
        'AWAY_POINTS': games['PTS_AWAY'],
        'TOTAL_POINTS': games['PTS_HOME'] + games['PTS_AWAY']
    })

    final_data = final_data.drop_duplicates(subset=['GAME_ID'])

    final_data.to_csv('nba_games.csv', index=False)

    print(final_data.head())


if __name__ == "__main__":
    get_logs()
