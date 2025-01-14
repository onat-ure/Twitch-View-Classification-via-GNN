# Twitch View Prediction via GNN

## MATH 482 Project

This project uses Graph Neural Networks (GNN) to predict Twitch views based on the social connections of Twitch users. The dataset used is from the [Twitch Gamers dataset](https://snap.stanford.edu/data/twitch_gamers.html), a graph dataset containing information about the connections between Twitch streamers and their attributes.

---

## Dataset Information

- **Source**: [Twitch Gamers dataset](https://snap.stanford.edu/data/twitch_gamers.html)
- **Type**: Undirected graph
- **Number of Nodes**: 168,114
- **Number of Edges**: 6,797,557

### Features:
- Nodes represent Twitch streamers.
- Edges represent mutual friendships between streamers.
- Node features include:
    - **Identifier**: Numeric vertex identifier (Index).
    - **Dead Account**: Inactive user account (Categorical).
    - **Broadcaster Language**: Languages used for broadcasting (Categorical). 
    - **Affiliate Status**: Affiliate status of the user (Categorical).
    - **Explicit Content**: Explicit content on the channel (Categorical).
    - **Creation Date**: Joining date of the user (Date).
    - **Last Update**: Last stream of the user (Date).
    - **Account Lifetime**: Days between first and last stream (Count).
    - **Time Since Creation**: Days since the user joined Twitch (Count).


---

## Dataset Creation

1. **Download the Dataset:**
   - Download the Twitch Gamers dataset from the [source](https://snap.stanford.edu/data/twitch_gamers.html).
   - Extract the dataset files.


3. **Prepare Dataset:**
   Run the `\utils\data-utils\create_data.ipynb` notebook to create the dataset.


## Training

1. **Install Dependencies:**
   Ensure you have the necessary libraries installed:
   ```
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   Run the `train.py` script to train the model.
   ```
   python train.py
   ```

---

## Testing and Evaluation

1. **Evaluate the Model:**
   Run the `test.py` script to evaluate the model. Enter the number of the latest best model to evaluate. This will also generate a plot of the predicted vs true views. 

   ```
   python test.py
   ```

