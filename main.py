from ucimlrepo import fetch_ucirepo

# fetch dataset
horse_colic = fetch_ucirepo(id=47)

# data (as pandas dataframes)
X = horse_colic.data.features
y = horse_colic.data.targets

# metadata
print(horse_colic.metadata)

# variable information
print(horse_colic.variables)

# Кто прочитает - тот брюква