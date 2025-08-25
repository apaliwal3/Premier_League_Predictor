# ===========================
# Premier League Predictor
# ===========================

import kagglehub
import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# ---------------------------
# Load Data
# ---------------------------
path = kagglehub.dataset_download("marcohuiii/english-premier-league-epl-match-data-2000-2025")

print("Dataset downloaded to:", path)

files = os.listdir(path)
print("Files in dataset:", files)

df = pd.read_csv(os.path.join(path, "epl_final.csv"))

print("DataFrame shape:", df.shape)

df.head()

# ---------------------------
# Feature Engineering
# ---------------------------
def add_recent_form_features(df, n=5):
    """
    Adds rolling form features (last n matches) for both Home and Away teams.
    """
    df = df.sort_values("MatchDate").copy()

    # Initialize storage for new columns
    for col in ['Home_avg_goals_scored','Home_avg_goals_against','Home_avg_points',
                'Away_avg_goals_scored','Away_avg_goals_against','Away_avg_points']:
        df[col] = np.nan

    # Process each team separately
    for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
        team_matches = team_matches.sort_values("MatchDate")

        # Assign goals scored/conceded depending on home/away
        team_matches['goals_scored'] = team_matches.apply(
            lambda row: row['FullTimeHomeGoals'] if row['HomeTeam'] == team else row['FullTimeAwayGoals'], axis=1
        )
        team_matches['goals_against'] = team_matches.apply(
            lambda row: row['FullTimeAwayGoals'] if row['HomeTeam'] == team else row['FullTimeHomeGoals'], axis=1
        )

        # Points system (3 win, 1 draw, 0 loss)
        team_matches['points'] = team_matches.apply(
            lambda row: 3 if ((row['HomeTeam'] == team and row['FullTimeHomeGoals'] > row['FullTimeAwayGoals']) or
                              (row['AwayTeam'] == team and row['FullTimeAwayGoals'] > row['FullTimeHomeGoals']))
            else (1 if row['FullTimeHomeGoals'] == row['FullTimeAwayGoals'] else 0),
            axis=1
        )

        # Rolling stats (shifted so they exclude current game)
        team_matches['avg_goals_scored'] = team_matches['goals_scored'].rolling(n, min_periods=1).mean().shift(1)
        team_matches['avg_goals_against'] = team_matches['goals_against'].rolling(n, min_periods=1).mean().shift(1)
        team_matches['avg_points'] = team_matches['points'].rolling(n, min_periods=1).mean().shift(1)

        # Write results back into main df
        for idx, row in team_matches.iterrows():
            if row['HomeTeam'] == team:
                df.at[idx, 'Home_avg_goals_scored'] = row['avg_goals_scored']
                df.at[idx, 'Home_avg_goals_against'] = row['avg_goals_against']
                df.at[idx, 'Home_avg_points'] = row['avg_points']
            else:
                df.at[idx, 'Away_avg_goals_scored'] = row['avg_goals_scored']
                df.at[idx, 'Away_avg_goals_against'] = row['avg_goals_against']
                df.at[idx, 'Away_avg_points'] = row['avg_points']

    return df

# def add_home_away_performance(df, n=5):
#     """
#     Adds rolling home/away performance features for each team.
#     """
#     df = df.sort_values("MatchDate").copy()
#     for col in [
#         'Home_avg_home_goals_scored', 'Home_avg_home_goals_against',
#         'Away_avg_away_goals_scored', 'Away_avg_away_goals_against']:
#         df[col] = np.nan

#     for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
#         # Home performance
#         home_matches = df[df['HomeTeam'] == team].sort_values("MatchDate")
#         home_matches['goals_scored'] = home_matches['FullTimeHomeGoals']
#         home_matches['goals_against'] = home_matches['FullTimeAwayGoals']
#         home_matches['avg_home_goals_scored'] = home_matches['goals_scored'].rolling(n, min_periods=1).mean().shift(1)
#         home_matches['avg_home_goals_against'] = home_matches['goals_against'].rolling(n, min_periods=1).mean().shift(1)
#         for idx, row in home_matches.iterrows():
#             df.at[idx, 'Home_avg_home_goals_scored'] = row['avg_home_goals_scored']
#             df.at[idx, 'Home_avg_home_goals_against'] = row['avg_home_goals_against']

#         # Away performance
#         away_matches = df[df['AwayTeam'] == team].sort_values("MatchDate")
#         away_matches['goals_scored'] = away_matches['FullTimeAwayGoals']
#         away_matches['goals_against'] = away_matches['FullTimeHomeGoals']
#         away_matches['avg_away_goals_scored'] = away_matches['goals_scored'].rolling(n, min_periods=1).mean().shift(1)
#         away_matches['avg_away_goals_against'] = away_matches['goals_against'].rolling(n, min_periods=1).mean().shift(1)
#         for idx, row in away_matches.iterrows():
#             df.at[idx, 'Away_avg_away_goals_scored'] = row['avg_away_goals_scored']
#             df.at[idx, 'Away_avg_away_goals_against'] = row['avg_away_goals_against']
#     return df

# def add_head_to_head_features(df, n=3):
#     """
#     Adds rolling head-to-head features for each match.
#     """
#     df = df.sort_values("MatchDate").copy()
#     df['h2h_avg_home_goals'] = np.nan
#     df['h2h_avg_away_goals'] = np.nan
#     df['h2h_avg_result'] = np.nan
#     for idx, row in df.iterrows():
#         home = row['HomeTeam']
#         away = row['AwayTeam']
#         match_date = row['MatchDate']
#         # Find previous n head-to-heads before this match
#         prev_h2h = df[((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
#                       ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))]
#         prev_h2h = prev_h2h[prev_h2h['MatchDate'] < match_date].sort_values('MatchDate').tail(n)
#         if not prev_h2h.empty:
#             # Home goals: when home, use FullTimeHomeGoals; when away, use FullTimeAwayGoals
#             home_goals = prev_h2h.apply(lambda r: r['FullTimeHomeGoals'] if r['HomeTeam'] == home else r['FullTimeAwayGoals'], axis=1)
#             away_goals = prev_h2h.apply(lambda r: r['FullTimeAwayGoals'] if r['AwayTeam'] == away else r['FullTimeHomeGoals'], axis=1)
#             # Result: 1=home win, 0=draw, -1=away win (from home team perspective)
#             def h2h_result(r):
#                 if r['FullTimeHomeGoals'] > r['FullTimeAwayGoals']:
#                     return 1 if r['HomeTeam'] == home else -1
#                 elif r['FullTimeHomeGoals'] < r['FullTimeAwayGoals']:
#                     return -1 if r['HomeTeam'] == home else 1
#                 else:
#                     return 0
#             results = prev_h2h.apply(h2h_result, axis=1)
#             df.at[idx, 'h2h_avg_home_goals'] = home_goals.mean()
#             df.at[idx, 'h2h_avg_away_goals'] = away_goals.mean()
#             df.at[idx, 'h2h_avg_result'] = results.mean()
#     return df

def add_attacking_efficiency_features(df, n=5):
    """
    Adds rolling attacking efficiency features (goals per shot, goals per shot on target) for both Home and Away teams.
    """
    df = df.sort_values("MatchDate").copy()
    for col in [
        'Home_avg_goals_per_shot', 'Home_avg_goals_per_shot_on_target',
        'Away_avg_goals_per_shot', 'Away_avg_goals_per_shot_on_target']:
        df[col] = np.nan

    for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
        team_matches = team_matches.sort_values("MatchDate")

        # Assign shots and goals depending on home/away
        team_matches['goals'] = team_matches.apply(
            lambda row: row['FullTimeHomeGoals'] if row['HomeTeam'] == team else row['FullTimeAwayGoals'], axis=1)
        team_matches['shots'] = team_matches.apply(
            lambda row: row['HomeShots'] if row['HomeTeam'] == team else row['AwayShots'], axis=1)
        team_matches['shots_on_target'] = team_matches.apply(
            lambda row: row['HomeShotsOnTarget'] if row['HomeTeam'] == team else row['AwayShotsOnTarget'], axis=1)

        # Calculate rolling attacking efficiency (shifted to exclude current match)
        team_matches['goals_per_shot'] = (team_matches['goals'] / team_matches['shots'].replace(0, np.nan)).rolling(n, min_periods=1).mean().shift(1)
        team_matches['goals_per_shot_on_target'] = (team_matches['goals'] / team_matches['shots_on_target'].replace(0, np.nan)).rolling(n, min_periods=1).mean().shift(1)

        for idx, row in team_matches.iterrows():
            if row['HomeTeam'] == team:
                df.at[idx, 'Home_avg_goals_per_shot'] = row['goals_per_shot']
                df.at[idx, 'Home_avg_goals_per_shot_on_target'] = row['goals_per_shot_on_target']
            else:
                df.at[idx, 'Away_avg_goals_per_shot'] = row['goals_per_shot']
                df.at[idx, 'Away_avg_goals_per_shot_on_target'] = row['goals_per_shot_on_target']
    return df

df = add_recent_form_features(df, n=5)
# df = add_home_away_performance(df, n=5)
# df = add_head_to_head_features(df, n=3)
df = add_attacking_efficiency_features(df, n=5)
df = df.dropna()

# ---------------------------
# Model Training
# ---------------------------
# Define target (1 = home win, 0 = draw, -1 = away win)
df['FullTimeResult'] = df.apply(
    lambda row: 1 if row['FullTimeHomeGoals'] > row['FullTimeAwayGoals'] 
                else (-1 if row['FullTimeHomeGoals'] < row['FullTimeAwayGoals'] else 0), axis=1
)



feature_cols = [
    'Home_avg_goals_scored','Home_avg_goals_against','Home_avg_points',
    'Away_avg_goals_scored','Away_avg_goals_against','Away_avg_points',
    # 'Home_avg_home_goals_scored','Home_avg_home_goals_against',
    # 'Away_avg_away_goals_scored','Away_avg_away_goals_against',
    # 'h2h_avg_home_goals','h2h_avg_away_goals','h2h_avg_result',
    'Home_avg_goals_per_shot','Home_avg_goals_per_shot_on_target',
    'Away_avg_goals_per_shot','Away_avg_goals_per_shot_on_target'
]


# Remap y: -1 (Away Win) -> 0, 0 (Draw) -> 1, 1 (Home Win) -> 2
label_map = {-1: 0, 0: 1, 1: 2}
inv_label_map = {0: -1, 1: 0, 2: 1}
X = df[feature_cols]
y = df['FullTimeResult'].map(label_map)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


# Use XGBoost Classifier
print("Training model...")
model = XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', use_label_encoder=False)
model.fit(X_train, y_train)
print("Model training complete.")

print("Model Evaluation:")
print(classification_report(y_test, model.predict(X_test)))

# ---------------------------
# Prediction Function
# ---------------------------
def predict_match(home_team, away_team, df, model, feature_cols):
    """
    Predict the outcome of a match given Home and Away team names.
    Uses the most recent row for those teams.
    """
    # Get last match between these teams (or most recent game involving them)
    match = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
           ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))].iloc[-1]

    # Convert to DataFrame to preserve feature names
    X_new = pd.DataFrame([match[feature_cols].values], columns=feature_cols)
    prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)[0]

    # Map prediction and probabilities back to original labels
    mapped_pred = inv_label_map[prediction]
    outcome_map = {1: "Home Win", 0: "Draw", -1: "Away Win"}


    if mapped_pred == 1:
        pred_str = f"Prediction: {home_team} wins"
    elif mapped_pred == 0:
        pred_str = f"Prediction: Draw between {home_team} and {away_team}"
    else:
        pred_str = f"Prediction: {away_team} wins"
    print(pred_str)
    print(f"Probabilities → {home_team} Win: {probabilities[2]:.2f}, "
          f"Draw: {probabilities[1]:.2f}, "
          f"{away_team} Win: {probabilities[0]:.2f}")

    return mapped_pred, probabilities

# ---------------------------
# Command-line Interface
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Premier League Match Predictor")
    parser.add_argument('--home', type=str, help='Home team name', required=True)
    parser.add_argument('--away', type=str, help='Away team name', required=True)
    args = parser.parse_args()

    predict_match(args.home, args.away, df, model, feature_cols)