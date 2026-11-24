## Startup Information

# 1. Create Virtual Environment and Install Dependencies

Makes sure you have the correct dependencies to run this program.

```
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

# 2. Create nba_games.csv by running data_collection.py

Run data_collection.py

```
python3 data_collection.py
```

# 3. Run the model

```
python3 model.py
```

This will read the data from nba_games.csv and create a Linear Regression model

It uses these features:
- Teams Playing
- Teams scoring average (past 5 games)
- Total game points (past 5 games)

Will display the Mean Absolute Error as well as 10 sample predictions.

