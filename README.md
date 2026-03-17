# Neural Network Approach to ATP Tennis Match Prediction

This repository contains an academic project on ATP tennis match prediction. It studies whether historical match data and betting odds can be used to estimate the probability that one player beats another, and it combines data preparation, tennis-specific feature engineering, and neural network modeling to do so.

The dataset used in the project covers ATP matches from 2015 to 2023, for a total of 22,092 matches after preprocessing. The main research goal is to compare model predictions against betting-market implied probabilities.

## Research Focus

- merges ATP match records with bookmaker odds
- cleans and standardizes tournament, player, and match information
- creates tennis-specific features such as Elo ratings, fatigue, head-to-head history, tournament history, round-level stats, and recent form
- converts each match into a symmetric player-versus-player representation for modeling
- trains and evaluates a neural network for tennis match outcome prediction

## Model Summary

The final model combines:

- a symmetric Wide & Deep network for player and environment features
- a GRU-based encoder for each player's recent match history
- FiLM conditioning to inject form information into the main prediction network

The symmetric design is intended to keep predictions consistent when player order is swapped.

## Repository Contents

- `raw_data/`: original ATP and betting datasets used in the study
- `preprocessing/`: notebooks for merging and cleaning the raw data
- `feature_creation/`: scripts for constructing the engineered features used in the experiments
- `models/`: notebook and helper code for model training, evaluation, and analysis, plus saved scaler artifacts

## Results

The final model achieved:

- top-50 player match Brier score: `0.1961 ± 0.0005`

For comparison, the betting-market baseline achieved:

- top-50 player match Brier score: `0.1917`

This is a fairly close result, especially considering how difficult it is to beat the betting market.
