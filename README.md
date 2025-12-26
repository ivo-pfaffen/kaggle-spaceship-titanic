## Kaggle competition: Spaceship Titanic 🌌
For my second ML/data science project, I'm participating in Kaggle's Spaceship Titanic competition.

### Data

We are provided multiple CSVs:
* `train.csv`: personal records for about two-thirds (~8700) of the passengers, to be used as training data
* `test.csv`: personal records for the remaining one-third (~4300) of the passengers, to be used as test data
* `sample_submission`: a submission file in the correct format
Which are all located in the `data/` directory.

The one we use for training looks like this:
```csv
PassengerId,HomePlanet,CryoSleep,Cabin,Destination,Age,VIP,RoomService,FoodCourt,ShoppingMall,Spa,VRDeck,Name,Transported
0001_01,Europa,False,B/0/P,TRAPPIST-1e,39.0,False,0.0,0.0,0.0,0.0,0.0,Maham Ofracculy,False
0002_01,Earth,False,F/0/S,TRAPPIST-1e,24.0,False,109.0,9.0,25.0,549.0,44.0,Juanna Vines,True
(...)
```

The goal here is to predict the status of ~4k passengers in a spaceship based on ~8k passenger records. 

### Approach
Our first task is to load and preprocess the data to be able to feed it into our neural network for training. As we can see, there are lots of non-numeric data.
We are going to perform feature encoding for each of the columns containing non-numerical data, and some feature engineering after that to improve model performance.

~~After that, I intend to build a multilayered feed-forward neural network using Pytorch to predict the outcome (the `Transported` column) for each of the passengers in the `test.csv` file.~~

That approach didn't work out as I expected, so I used XGBoost to train a classifier, using the `xgboost` package.    

### How to run the code
Eveything is located in the three Notebooks. To run them, follow the steps:

- Clone the repository
- Create a virtual environment (`python3 -m venv .venv`) 
- Activate the virtual enviroment (`source .venv/bin/activate`)
- Install the dependencies (`pip install -r requirements.txt`)

After that:
- To preprocess the data, run
```sh
papermill data_preprocessing.ipynb \
          data_preprocessing_train.ipynb \
          -p IS_TRAIN true

papermill data_preprocessing.ipynb \
          data_preprocessing_test.ipynb \
          -p IS_TRAIN false
```
- To train the XGBoost classifier, open and run the `training_xgb.ipynb` notebook in your local IDE. Just make sure that it's running inside the virtual enviromnent (tutorial for VSCode [here](https://code.visualstudio.com/docs/python/environments#_using-the-create-environment-command)).

