import pandas as pd

mp = pd.read_csv("clean_dataset/match_players_with_clusters.csv")

comp = (
    mp.groupby(["match_id", "team", "won_match"])["cluster"]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

print(comp.head(10))

# Compute win rate grouped by composition
winrate_by_comp = (
    comp.groupby([0, 1, 2, 3, 4, 5, "team"])["won_match"]
    .mean()
    .reset_index()
    .sort_values("won_match", ascending=False)
)

print(winrate_by_comp.head(20))
print("\nWorst 10 Compositions:")
print(winrate_by_comp.tail(10))
