
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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
    team_history.columns = [
        "GAME_ID", side + "_AVG_POINTS", side + "_AVG_TOTAL"
    ]

    games = games.merge(team_history, on="GAME_ID")

features = [
    "HOME_TEAM", "AWAY_TEAM",
    "HOME_AVG_POINTS", "AWAY_AVG_POINTS",
    "HOME_AVG_TOTAL", "AWAY_AVG_TOTAL"
]

games = games.dropna(subset=features[2:])

X = pd.get_dummies(games[features], dtype=int)
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
