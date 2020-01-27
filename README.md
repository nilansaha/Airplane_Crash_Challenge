## Airplane Crash Challenge

This challenge was posted on HackerEarth and I was able to get a rank of 85 using this model on the public leaderboard

Challenge Link: [Click Here](https://www.hackerearth.com/challenges/competitive/airplane-accident-severity-hackerearth-machine-learning-challenge/problems/)

### File Details

```
main.py - Contains the main code to generate output prediction file
lgbm_param_search.py - Code to search the best hyperparameter for LightGBM model using Hyperopt
forward_selection.py - Code for personal Forward Feature Selection Class
```

**Note:**  
The features selected in the final model were selected using the forward selection code. I have not added code to use that but the docs are pretty clear for that so shouldn't be a big issue to use it for this dataset

There might be some redundant code that is not used in the main model. I tried out a lot of things but did not bother cleaning them up