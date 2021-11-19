import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import seaborn as sns
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import math

# dataload
data_2010s = pd.read_csv('dataset/dataset-of-10s.csv')
data_2000s = pd.read_csv('dataset/dataset-of-00s.csv')
data_1990s = pd.read_csv('dataset/dataset-of-90s.csv')
data_1980s = pd.read_csv('dataset/dataset-of-80s.csv')

# checking null data
data_2010s.isnull().sum()
data_2000s.isnull().sum()
data_1990s.isnull().sum()
data_1980s.isnull().sum()

# add column for year
data_2010s["year"] = 2010
data_2000s["year"] = 2000
data_1990s["year"] = 1990
data_1980s["year"] = 1980

# combining all data
df_list = [data_2010s, data_2000s, data_1990s, data_1980s]
data = pd.concat(df_list)

# revise and add column for song's duration 
data["duration_s"] = data["duration_ms"]/1000

# devide hits and flot data in case
hits = data[data["target"] == 1]
flop = data[data["target"] == 0]

# overall describe varibles 
data_description = data.describe()

# numerical data 
# variables
var_list = ['danceability', 'energy', 'key', 'loudness',
       'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'time_signature', 'chorus_hit', 'duration_s']

# hits and flop histogram
plt.style.use('ggplot')
fig, ax = plt.subplots(2, 7, figsize=(20, 20))
fig.suptitle('Hits or Flop', fontsize=40)

column_idx = 0
for i in range(2):
    for j in range(7):
        ax[i][j].hist([hits[var_list[column_idx]], flop[var_list[column_idx]]], bins=30, label=['hit', 'flop'])
        ax[i][j].set_title(var_list[column_idx])
        ax[i][j].legend()
        column_idx += 1
plt.show();
# plt.savefig('image1.png', facecolor='white')

# features distribution boxplot 
# plt.style.use('ggplot')
fig, axes = plt.subplots(2, 7, figsize = (20, 20))
fig.suptitle('features distribution', fontsize = 30)

column_idx = 0
for i in range(2):
  for j in range(7):
    sns.boxplot(x = data['target'], y = data[var_list[column_idx]], hue = data['target'], notch = True, width = 0.3, ax = axes[i][j]).set_title(var_list[column_idx], fontsize = 15)
    axes[i][j].set_xlabel('')
    axes[i][j].set_ylabel('')
    column_idx +=1
plt.show();
# plt.savefig('image2.png', facecolor='white')

# variables correlation heatmap
data_corr = data.drop(['duration_ms'], axis=1)
corr = data_corr.corr()
# plt.style.use('ggplot')
plt.figure(figsize=(20, 15))
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=True, cmap='RdBu', annot=True, fmt='.2f')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('features correlation', fontsize=40)
plt.show();
# plt.savefig('image3.png', facecolor='white')

# variable's correaltion over 0.5 
def var_scatter(x, y, num):
    plt.figure(figsize=(20, 15))
    sns.scatterplot(x=x, y=y, hue=data['target'])
    plt.show();
    plt.savefig(f'scatter{num}.png', facecolor='white')

# danceability vs valence
var_scatter(data['danceability'], data['valence'], 1)

# energy vs acousticness, loudness
var_scatter(data['energy'], data['acousticness'], 2)
var_scatter(data['energy'], data['loudness'], 3)

# loudness vs acousticness
var_scatter(data['loudness'], data['acousticness'], 4)

# sections vs duration_s
var_scatter(data['sections'], data['duration_s'], 5)

# categorical data
# hit or flop artist by track
data_hits = data.groupby('artist')['track'].agg(len).sort_values(ascending = False).to_frame().reset_index()
# artist_hits = hits.groupby('artist')['track'].agg(len).sort_values(ascending = False).to_frame().reset_index()
# flop_hits = flop.groupby('artist')['track'].agg(len).sort_values(ascending = False).to_frame().reset_index()

artist_list = data_hits[:50]['artist']
data_hits50 = data.loc[data['artist'].isin(artist_list)]

plt.figure(figsize=(20, 15))
g = sns.countplot(x=data_hits50['artist'], hue=data_hits50['target'])
g.set_xticklabels(g.get_xticklabels(), rotation=45,
                  horizontalalignment='right')
plt.title("Hit Artist or Flop Artist", fontsize=40)
plt.show();
# plt.savefig('image4.png', facecolor='white')

