# Spotify Song Hit or Flop Prediction

히트송에는 어떤 조건들이 포함될지를 탐색해보고 히트할지 안할지를 예측해는 프로젝트

## 목차
- [Introduction](#introduction)
- [Overview of the Data](#overview-of-the-data)
    * [Preprocess](#preprocess)
- [Exploratory Data Analysis](#exploratory-data-analysis)
    * [Numerical Data](#numerical-data)
    * [Categorical Data](#categorical-data)
- [Machine Learning Modeling](#machine-learning-modeling)
    * [Testing algorithm](#testing-algorithm)
    * [Learning Curve](#learning-curve)
    * [Confusion Matrix](#confusion-matrix)

# Introduction

blah blah

kaggle data blah blah

# Overview of the Data
- 19개의 컬럼 특성 컬럼과 1개의 타켓 컬롬 총 20개의 컬럼으로 구성 
- 년도별로 데이터 따로 존재

### Preprocess
- 1980년대 부터 2010년대 데이터를 사용
- 연도 구분을 위해 'year' 과 단위를 'ms' 에서 's'로 변경하기 위해 'duration_s' 컬럼 추가
- 총 21개의 컬럼 (18개 numerical data, 3개 categorical data)

| Index | Attribute | Description |
| 1 | track | blah |
| 2 | artist | blah |
| 3 | uri | blah |
| 4 | danceability | blah 
| 5 | energy | blah |
| 6 | key | blah |
| 7 | loudness | blah |
| 8 | mode | blah |
| 9  speechiness | blah |
| 10 | acousticness | blah |
| 11 | instrumentalness | blah |
| 12 | liveness | blah |
| 13 | valence | blah |
| 14 | tempo | blah |
| 15 | duration_ms | blah |
| 16 | time_signature | blah |
| 17 | chorus_hit | blah |
| 18 | sections | blah |
| 19 | target | blah |
| 20 | year | blah |
| 21 | duration_s | blah |

# Exploratory Data Analysis 

### Numerical Data
### Categorical Data

# Machine Learning Modeling

### Testing algorithm
### Learning Curve
### Confusion Matrix