#TODO: Remove unneeded libraries
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from collections import defaultdict
from surprise.model_selection import KFold
import numpy as np
import pandas as pd
from collections import defaultdict



# Use surprise library
# Prepare data
reader = Reader(
        line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))
coruse_dataset = Dataset.load_from_file("course_ratings.csv", reader=reader)

# Use surprise library
# Test train split
trainset, testset = train_test_split(coruse_dataset, test_size=.25)

#Threshold for the ratings = 2.5

def precision_recall_at_k(predictions, k=10, threshold=2.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


    #Get average precision and recalls
def average_dicts(dictionary):
    sum = 0
    count = 0
    for value in dictionary.values():
        sum += value
        count += 1
    return sum / count

# Create the KNNBasic model with the optimal value of k
knn_1 = KNNBasic(k=10, sim_options = {'name': 'msd'})

# Fit the KNNBasic model to the entire training set
knn_1.fit(trainset)

# Predict ratings for the test set
predictions_1 = knn_1.test(testset)

precisions, recalls = precision_recall_at_k(predictions = predictions_1, k=10, threshold=2.5)

# Get average Precision and recalls
precision_average = average_dicts(precisions)
recalls_average = average_dicts(recalls)


#f1 value
f1_model_1 = 2 * (precision_average * recalls_average) / (precision_average + recalls_average)


## Model 2 -> cosine


# Create the KNNBasic model with the optimal value of k
knn_2 = KNNBasic(k=10, sim_options = {'name': 'cosine'})

# Fit the KNNBasic model to the entire training set
knn_2.fit(trainset)

# Predict ratings for the test set
predictions_2 = knn_2.test(testset)

#Precision and recall
precisions, recalls = precision_recall_at_k(predictions = predictions_2, k=10, threshold=2.5)

# Get average Precision and recalls
precision_average = average_dicts(precisions)
recalls_average = average_dicts(recalls)

#f1_model
f1_model_2 = 2 * (precision_average * recalls_average) / (precision_average + recalls_average)


# Create the KNNBasic model with the optimal value of k
knn_3 = KNNBasic(k=20, sim_options = {'name': 'cosine'})

# Fit the KNNBasic model to the entire training set
knn_3.fit(trainset)

# Predict ratings for the test set
predictions_3 = knn_3.test(testset)

#Precision and recall
precisions, recalls = precision_recall_at_k(predictions = predictions_3, k=10, threshold=2.5)

# Get average Precision and recalls
precision_average = average_dicts(precisions)
recalls_average = average_dicts(recalls)

#f1_model
f1_model_3 = 2 * (precision_average * recalls_average) / (precision_average + recalls_average)

import matplotlib.pyplot as plt
import seaborn as sns
f1_values = [f1_model_1, f1_model_2, f1_model_3]

maximum = max(f1_values)
position = f1_values.index(maximum)
print(maximum)
print(f1_values[position])

print(min(f1_values[0], f1_values[1], f1_values[2]))
print(max(f1_values[0], f1_values[1], f1_values[2]))

#TODO: Add a comment here
f1_names = ["f1_model_1", "f1_model_2", "f1_model_3"]
# create a dataframe with the values and names
df_f1 = pd.DataFrame({'f1_score (higher the better)': f1_values, 'f1_model': f1_names})

# use seaborn to plot the bar chart
sns.barplot(x='f1_model', y='f1_score (higher the better)', data=df_f1)
plt.ylim(0.904026, 0.904031)
plt.title('f1_values')
plt.plot()



