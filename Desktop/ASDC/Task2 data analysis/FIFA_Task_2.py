# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 10:26:06 2023

@author: AHMED YASSER
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/xx6k/Desktop/ASDC/Task2 data analysis/players_22.csv")
print(df.head())
print(df.isnull().sum(axis=0))
print(df.dtypes)
print(df.info())

# Set the display option to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Example for handling missing values in numeric columns
numeric_columns = ['value_eur', 'wage_eur', 'release_clause_eur', 'pace', 'shooting',
                   'passing', 'dribbling', 'defending', 'physic', 'goalkeeping_speed']
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
# Example for handling missing values in categorical columns
categorical_columns = ['club_team_id','nation_team_id', 'club_name', 'league_name',
                       'league_level', 'player_tags','player_traits','club_position',
                       'club_jersey_number', 'club_loaned_from', 'club_joined', 
                       'club_contract_valid_until','nation_position','nation_jersey_number']
df[categorical_columns] = df[categorical_columns].fillna('Unknown')
print(df.isnull().sum(axis=0))
"""
# Handling missing values
df.dropna(inplace=True)  # Remove rows with missing values
# Remove duplicates
df.drop_duplicates(inplace=True)
print(df.isnull().sum(axis=0))
"""
# Summary statistics of numeric columns
print("\nSummary Statistics:")
print(df.describe())

# (1) Identify the player with the highest overall rating
highest_rated_player = df.loc[df['overall'].idxmax()]
print("Player with the Highest Overall Rating:")
print(highest_rated_player[['short_name','long_name', 'overall','club_name']])

# (2) Identify the player with the lowest overall rating
lowest_rated_player = df.loc[df['overall'].idxmin()]
print("\nPlayer with the Lowest Overall Rating:")
print(lowest_rated_player[['short_name','long_name', 'overall','club_name']])

# (3)Identify the top 5 players with the highest overall ratings
top_5_players = df.nlargest(5, 'overall') 
#nlargest :Return the first n rows with the largest values in columns, in descending order. The columns that are not specified are returned as well, but not used for ordering.
print("Top 5 Players with the Highest Overall Ratings:")
print(top_5_players[['short_name','overall','club_name']])
#(4)Identify the lowest 5 players with the smallest overall ratings
smallest_5_plyers=df.nsmallest(5,'overall')
print("low 5 Players with the smallest Overall Ratings:")
print(smallest_5_plyers[['short_name','overall','club_name']])

#(5)Identify the top 5 players with MarketValue
top_5_players_value_eur = df.nlargest(5, 'value_eur') 
print("Top 5 Players MarketValue:")
print(top_5_players_value_eur[['short_name','value_eur','club_name']])

#(6)Identify the lowest 5 players with the MarketValue
smallest_5_plyers_Value=df.nsmallest(5,'value_eur')
print("low 5 Players MarketValue:")
print(smallest_5_plyers_Value[['short_name','value_eur','club_name']])

#(7)Scatter plot for age vs. Overall Rating
plt.figure(figsize=(12, 6))
sns.scatterplot(x='age', y='overall', data=df, alpha=0.5)
plt.title('Relationship between Overall Rating and Age')
plt.xlabel('Age')
plt.ylabel('Overall Rating')
plt.show()

#(8)Scatter plot for Market Value vs. Overall Rating
sns.scatterplot(x='overall', y='value_eur', data=df, alpha=0.5)
plt.title('Market Value vs. Overall Rating')
plt.xlabel('Overall Rating')
plt.ylabel('Market Value')
plt.show()

#(9)Scatter plot for age vs.wage
sns.scatterplot(x='age', y='wage_eur', data=df, alpha=0.5)
plt.title('age vs. wage')
plt.xlabel('age')
plt.ylabel('wage')
plt.show()

#(10)Scatter plot for age vs.Market Value
sns.scatterplot(x='age', y='value_eur', data=df, alpha=0.5)
plt.title('age vs. Market Value')
plt.xlabel('age')
plt.ylabel('Market Value')
plt.show()

#(11)Calculate the average overall rating of players the age is greater than or equal 35
# Filter the DataFrame to include only players at the age of 35
players_at_age_35 = df[df['age'] >= 35]
average_rating_at_age_35 = players_at_age_35['overall'].mean()
print(average_rating_at_age_35)

CR7=df.loc[(df['short_name'] == "Cristiano Ronaldo")]
print("IMPORTANT details about Cristiano Ronaldo ")
print(CR7[['short_name','overall','club_name']])


# Select relevant columns for correlation analysis
selected_columns = ['overall', 'value_eur', 'age', 'potential', 'wage_eur']
# Create a DataFrame with selected columns
selected_df = df[selected_columns]
# Calculate the correlation matrix
correlation_matrix = selected_df.corr()
# Create a heatmap to visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Overall Ratings, Market Values, potential , wage')
plt.show()

#Distribution of Players Based on Skill Moves:
plt.figure(figsize=(10, 6))
sns.countplot(x='skill_moves', data=df, order=df['skill_moves'].value_counts().index)
plt.title('Distribution of Players Based on Skill Moves in FIFA 22')
plt.xlabel('Skill Moves')
plt.ylabel('Count')
plt.show()

#Distribution of Players Based on Work Rate
plt.figure(figsize=(20, 6))
sns.countplot(x='work_rate', data=df, order=df['work_rate'].value_counts().index)
plt.title('Distribution of Players Based on Work Rate in FIFA 22')
plt.xlabel('Work Rate')
plt.ylabel('Count')
plt.show()

#Distribution of Players Based on Preferred Foot:
plt.figure(figsize=(8, 6))
sns.countplot(x='preferred_foot', data=df, order=df['preferred_foot'].value_counts().index)
plt.title('Distribution of Players Based on Preferred Foot in FIFA 22')
plt.xlabel('Preferred Foot')
plt.ylabel('Count')
plt.show()
