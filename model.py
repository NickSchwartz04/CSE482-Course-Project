import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

FEATURE_COLUMNS = ["HOME_AVG_POINTS", "AWAY_AVG_POINTS", "HOME_AVG_TOTAL", "AWAY_AVG_TOTAL"]

def build_dataset():
    games = pd.read_csv("nba_games.csv")
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], format="mixed")
    games = games.sort_values("GAME_DATE").reset_index(drop=True)

    home = games[["GAME_ID", "GAME_DATE", "HOME_TEAM", "HOME_POINTS", "TOTAL_POINTS"]].copy()
    home.columns = ["GAME_ID", "GAME_DATE", "TEAM", "POINTS", "TOTAL"]

    away = games[["GAME_ID", "GAME_DATE", "AWAY_TEAM", "AWAY_POINTS", "TOTAL_POINTS"]].copy()
    away.columns = ["GAME_ID", "GAME_DATE", "TEAM", "POINTS", "TOTAL"]

    history = pd.concat([home, away]).sort_values(["TEAM", "GAME_DATE", "GAME_ID"])

    history["AVG_POINTS"] = history.groupby("TEAM")["POINTS"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    history["AVG_TOTAL"] = history.groupby("TEAM")["TOTAL"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    for side in ["HOME", "AWAY"]:
        team_history = history.merge(
            games[["GAME_ID", side + "_TEAM"]],
            left_on=["GAME_ID", "TEAM"],
            right_on=["GAME_ID", side + "_TEAM"]
        )

        team_history = team_history[["GAME_ID", "AVG_POINTS", "AVG_TOTAL"]]
        team_history.columns = ["GAME_ID", side + "_AVG_POINTS", side + "_AVG_TOTAL"]

        games = games.merge(team_history, on="GAME_ID")

    return games, history


def train_model(games):
    games = games.dropna(subset=FEATURE_COLUMNS)

    X = games[FEATURE_COLUMNS]
    y = games["TOTAL_POINTS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))

    print("\nSample Predictions:")
    print(pd.DataFrame({
        "Actual Total": y_test.head(10),
        "Predicted Total": predictions[:10]
    }).round(1))

    # Test rows keep their original index (shuffle=False), so this pulls each
    # test game's date to plot predictions across the season instead of just one.
    test_dates = games.loc[X_test.index, "GAME_DATE"]
    plot_predictions_over_time(test_dates, y_test, predictions)

    return model


def plot_predictions_over_time(dates, actual, predicted):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual.values, label="Actual Total", marker="o")
    plt.plot(dates, predicted, label="Predicted Total", marker="o")
    plt.title("Predicted vs Actual Total Points Over Time")
    plt.xlabel("Game Date")
    plt.ylabel("Total Points")
    plt.legend()
    plt.grid(axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def resolve_team_name(history, team):
    normalized = team.strip().lower()
    all_teams = history["TEAM"].unique()

    exact = [t for t in all_teams if t.lower() == normalized]
    if exact:
        return exact[0]

    # Fall back to a substring match so nicknames like "Pistons" or "Magic"
    # resolve to "Detroit Pistons" / "Orlando Magic" without needing the full name.
    partial = [t for t in all_teams if normalized in t.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ValueError(
            f"'{team}' matches multiple teams: {', '.join(partial)}. Be more specific."
        )

    raise ValueError(f"Team '{team}' not found in the dataset.")


def latest_team_averages(history, team):
    resolved_team = resolve_team_name(history, team)
    matches = history[history["TEAM"] == resolved_team]

    latest = matches.iloc[-1]
    return latest["AVG_POINTS"], latest["AVG_TOTAL"]

def predict_total_points(model, history, home_team, away_team):
    home_avg_points, home_avg_total = latest_team_averages(history, home_team)
    away_avg_points, away_avg_total = latest_team_averages(history, away_team)

    features = pd.DataFrame([{
        "HOME_AVG_POINTS": home_avg_points,
        "AWAY_AVG_POINTS": away_avg_points,
        "HOME_AVG_TOTAL": home_avg_total,
        "AWAY_AVG_TOTAL": away_avg_total
    }])

    return model.predict(features)[0]

def main():
    games, history = build_dataset()
    model = train_model(games)

    home_team = input("Enter the home team name: ")
    away_team = input("Enter the away team name: ")

    predicted_total = predict_total_points(model, history, home_team, away_team)
    print(f"\nPredicted total points for {home_team} vs {away_team}: {round(predicted_total, 1)}")


if __name__ == "__main__":
    main()
