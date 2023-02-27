import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler, Normalizer
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
#user profiles
user_profile_df = pd.read_csv("user_profile.csv")

#Users we want to give reccomendations to
test_users_df = pd.read_csv('rs_content_test.csv')[['user', 'item']]

# Surpress any warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import string


# Covaraiance matrix to see how features are correlated
features = user_profile_df.loc[:, user_profile_df.columns != 'user']

sns.set_theme(style="white")

# Compute the correlation matrix
corr = features.cov()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


plt.show()


def cluster_df_algorithm(scaler, cluster_optimizer, use_pca = None):
    # Load data
    df = pd.read_csv("user_profile.csv")
    feature_names = df.columns[1:]
    user_ids = df['user']

    # Normalize the feature vectors
    df[feature_names] = scaler.fit_transform(df[feature_names])
    X = df[feature_names].values
    
    if use_pca == True:
        # Define the parameter grid for the grid search
        param_grid = {'n_components': range(1, 15)}

        # Define the PCA model
        pca = PCA()

        # Define the grid search
        grid_search = GridSearchCV(pca, param_grid, cv=5)

        # Fit the grid search to the data
        grid_search.fit(X)

        # The optimal number of components can be found in the best_params_ attribute of the grid search
        optimal_num_components = grid_search.best_params_['n_components']

        # Get the PCA model with the optimal number of components
        #optimal_pca = PCA(n_components=optimal_num_components)
        optimal_pca = PCA(n_components=7)
        # Fit the PCA model to the data
        optimal_pca.fit(X)

        # Transform the data to the reduced number of components
        X_transformed = pca.transform(X)
        X = X_transformed
        # Get the variance explained by each component
        explained_variance = optimal_pca.explained_variance_ratio_

        # Calculate the accumulated variance
        accumulated_variance = [explained_variance[:i+1].sum() for i in range(len(explained_variance))]


        # Plot the number of components on the x-axis and the accumulated variance ratio on the y-axis as a bar plot
        plt.bar(range(1, len(explained_variance)+1), accumulated_variance)

        # Add a trendline to the bar plot
        z = np.polyfit(range(1, len(explained_variance)+1), accumulated_variance, 1)
        p = np.poly1d(z)
        plt.plot(range(1, len(explained_variance)+1), p(range(1, len(explained_variance)+1)), "r--")
        plt.title('PCA')
        plt.xlabel('Number of components')
        plt.ylabel('Accumulated variance ratio')
        plt.show()
        
        # Create a new dataframe with the user ids and transformed features
        df = pd.DataFrame({'user_id': user_ids, 'feature_1': X_transformed[:, 0], 'feature_2': X_transformed[:, 1], 
                           'feature_3': X_transformed[:, 2], 'feature_4': X_transformed[:, 3],
                           'feature_5': X_transformed[:, 4], 'feature_6': X_transformed[:, 5],
                           'feature_7': X_transformed[:, 6]})

    
    
    def combine_cluster_labels(user_ids, labels, X):
        labels_df = pd.DataFrame(labels)
        cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
        cluster_df.columns = ['user', 'cluster']
        score = silhouette_score(X, labels)
        return cluster_df, score

    if cluster_optimizer == 'gridsearch':
        # Define a custom scoring function that returns the sum of squared distances
        def sum_squared_distances(estimator, X):
            centers = estimator.cluster_centers_
            cluster_assignments = estimator.predict(X)
            sum_squared_distances = 0
            for i in range(estimator.n_clusters):
                mask = cluster_assignments == i
                sum_squared_distances += np.sum((X[mask] - centers[i]) ** 2)
            return sum_squared_distances

        # Create a k-means model
        kmeans = KMeans()
        # Define the range of clusters to try
        param_grid = {'n_clusters': range(1, 30)}

        # Use grid search to find the best number of clusters
        grid_search = GridSearchCV(kmeans, param_grid, scoring=sum_squared_distances, cv=5)
        grid_search.fit(X)

        def plot_grid_search(grid_search):
            # Extract the results from the grid search object
            results = grid_search.cv_results_
            n_clusters = results['param_n_clusters']
            mean_scores = results['mean_test_score']

            # Plot the mean cross-validation scores for each value of n_clusters
            plt.plot(n_clusters, mean_scores)
            plt.xlabel('Number of clusters')
            plt.ylabel('Mean cross-validation score')
            plt.show()

        plot_grid_search(grid_search)

        # Get the best number of clusters
        best_n_clusters = grid_search.best_params_['n_clusters']
        sum_squared_distances = -1 * grid_search.cv_results_['mean_test_score']
        # Use the best number of clusters to fit the model
        best_kmeans = KMeans(n_clusters=15)
        best_kmeans.fit(X)  
        cluster_labels = best_kmeans.labels_
        
        # If the optimizer is lowest sum of squares
    elif cluster_optimizer == 'lowest sum of squares':
            
            # Fit k-means model to find the optimal number of clusters
            model = KMeans()
            scores = []
            for n_clusters in range(1, 30):
                model.n_clusters = n_clusters
                model.fit(X)
                scores.append(-model.score(X))
            best_n_clusters = np.argmin(scores) + 1
            #model.n_clusters = best_n_clusters
            model.n_clusters = 22
            model.fit(X)
            # Assign cluster labels to each user
            clusters = model.predict(X)
            cluster_labels = model.labels_
              
    elif cluster_optimizer == 'gap_statistic':
            
            def optimize_clusters_gap_statistic(X, n_reference_datasets=10, max_clusters=30):
                #Random seed for reproducibility
                np.random.seed(64)
                # Generate reference datasets
                reference_datasets = [np.random.permutation(X) for _ in range(n_reference_datasets)]

                # Compute the gap statistic for each number of clusters
                gap_statistics = []
                for n_clusters in range(1, max_clusters+1):
                    model = KMeans(n_clusters=n_clusters)
                    model.fit(X)
                    Wk = model.inertia_
                    Wk_reference = [model.fit(X_reference).inertia_ for X_reference in reference_datasets]
                    gap = np.log(np.mean(Wk_reference)) - np.log(Wk)
                    gap_statistics.append(gap)

                # Choose the optimal number of clusters
                best_n_clusters = np.argmax(gap_statistics) + 1
                return best_n_clusters

            # Run the Gap statistic optimizer
            #best_n_clusters = optimize_clusters_gap_statistic(X)
            best_n_clusters = 19
            # Fit k-means model with the optimal number of clusters
            model = KMeans(n_clusters=best_n_clusters)
            model.fit(X)

            # Assign cluster labels to each user
            clusters = model.predict(X)
            cluster_labels = model.labels_
        
        

        
    cluster_df, score = combine_cluster_labels(user_ids, cluster_labels, X)
    return cluster_df, score


def cluster_item_enrol(cluster_df, test_users_df):
    #Merge the test_df with the cluster_df to assign cluster label to test user
    test_users_labelled = pd.merge(test_users_df, cluster_df, left_on='user', right_on='user')

    #Enrollments count for each course in each group
    courses_cluster = test_users_labelled[['item', 'cluster']]
    courses_cluster['count'] = [1] * len(courses_cluster)
    cluster_item_enrol_df = courses_cluster.groupby(['cluster','item']).agg(enrollments = ('count','sum')).reset_index()
    return cluster_item_enrol_df, test_users_labelled

def reccomend_unseen(cluster_item_enrol_df, test_users_labelled, legacy_results_dict = None, hyperparms = None):
    # set threshold
    enrollment_count_threshold = 10
    # Filter the filtered_courses_df dataframe to only include courses with an enrollment count larger than the threshold
    popular_courses_df = cluster_item_enrol_df[cluster_item_enrol_df['enrollments'] > enrollment_count_threshold]
    # Find the courses in the popular_courses_df dataframe that are not in the list of courses taken by the test user
    unseen_courses = popular_courses_df[~popular_courses_df['item'].isin(test_users_labelled)]
    # Merge the test_users_labelled and cluster_item_enrol_df dataframes on the cluster and item columns
    merged_df = pd.merge(test_users_labelled, cluster_item_enrol_df, on=['cluster', 'item'])

    # Group the merged dataframe by the user column and get the list of courses taken by each test user
    test_user_courses = merged_df.groupby('user')['item'].apply(list).to_dict()

    # Initialize a dictionary to store the recommendation results for each test user
    recommendation_results = {}

    # Loop through the test users
    for user, cluster in test_users_labelled[['user', 'cluster']].values:
        # Filter the popular_courses_df dataframe to only include courses taken by users in the same cluster as the test user
        filtered_courses_df = popular_courses_df[popular_courses_df['cluster'] == cluster]

        # Get the list of courses taken by the test user or an empty list if the user is not found
        if user in test_user_courses:
            test_user_courses = test_user_courses[user]
        else:
            test_user_courses = []
    
    # Find the courses in the filtered dataframe that are not in the list of courses taken by the test user
    recommendations = filtered_courses_df[~filtered_courses_df['item'].isin(test_user_courses)]
    
    # Add the recommendation results for the test user to the dictionary
    recommendation_results[user] = recommendations

    # Calculate the mean number of recommendations for each user
    user_mean_recommendations = [len(recommendations) for recommendations in recommendation_results.values()]

    # Calculate the mean of the user_mean_recommendations list
    average_num_recommendations = np.mean(user_mean_recommendations)

    # Concatenate the dataframes in the recommendation_results dictionary into a single dataframe
    all_recommendations_df = pd.concat(recommendation_results.values())
    
    #Course counts
    course_counts_df = all_recommendations_df.groupby('item')['item'].count().reset_index(name='count')
    
    #What are the most frequently recommended courses? Return the top-10 commonly recommended courses across all users.
    top_10_courses = course_counts_df.sort_values(by='count', ascending=False).head(10)
    
    results_dict_new = {"all_recommendations_df":all_recommendations_df,"average_num_recommendations": average_num_recommendations, "user_mean_recommendations": user_mean_recommendations, "top_10_courses" :top_10_courses}
    if legacy_results_dict == True:
        for key in results_dict_new.keys():
            results_dict_new[key +'_hyperparams'] = results_dict.pop(key)
        results_dict = legacy_results_dict.update(results_dict_new)
    else:
        results_dict = results_dict_new
    
    return results_dict

cluster_optimizer = 'gridsearch'
scaler = StandardScaler()
cluster_df, score_1 = cluster_df_algorithm(scaler = scaler, cluster_optimizer = cluster_optimizer, use_pca = False)
cluster_item_enrol_df, test_users_labelled = cluster_item_enrol(cluster_df, test_users_df)
results_dict_1 = reccomend_unseen(cluster_item_enrol_df, test_users_labelled, legacy_results_dict = None, hyperparms = 'grid_standard')

cluster_optimizer = 'lowest sum of squares'
scaler = StandardScaler()
cluster_df, score_2 = cluster_df_algorithm(scaler = scaler, cluster_optimizer = cluster_optimizer, use_pca = False)
cluster_item_enrol_df, test_users_labelled = cluster_item_enrol(cluster_df, test_users_df)
results_dict_2 = reccomend_unseen(cluster_item_enrol_df, test_users_labelled, legacy_results_dict = None, hyperparms = 'sse_standard')


cluster_optimizer = 'gap_statistic'
scaler = StandardScaler()
cluster_df, score_3 = cluster_df_algorithm(scaler = scaler, cluster_optimizer = cluster_optimizer, use_pca = False)
cluster_item_enrol_df, test_users_labelled = cluster_item_enrol(cluster_df, test_users_df)
results_dict_3 = reccomend_unseen(cluster_item_enrol_df, test_users_labelled, legacy_results_dict = None, hyperparms = 'gs_standard')

results_dict_1

score_list = [score_1 , score_2 , score_3]
print(score_list)
print(max(score_list))

results_dict_2['average_num_recommendations']

fin_df = results_dict_2['all_recommendations_df'].iloc[:,1:]

# Sort the dataframe by the 'size' column in descending order
sorted_courses = fin_df.sort_values(by='enrollments', ascending=False)
#Reset index
sorted_courses = sorted_courses.reset_index(drop=True)
sorted_courses.head(10)

df_list_names = ['model 1', 'model 2', 'model 3']
import matplotlib.pyplot as plt
import seaborn as sns
# create a dataframe with the values and names
df_results_final = pd.DataFrame({'Silhoette score': score_list, 'model': df_list_names})

# use seaborn to plot the bar chart
sns.barplot(x='model', y='Silhoette score', data=df_results_final)
plt.title('Performance of each model')
plt.ylim(0.76, 0.78)
plt.plot()

sorted_courses['enrollments'].mean()

fin_df = results_dict_2['all_recommendations_df']
fin_df_2 = fin_df.sort_values(by='enrollments', ascending=False).head(10).reset_index(drop = True)
fin_df_3 = fin_df_2.iloc[:,1:]


course_genres_df = pd.read_csv('course_genre.csv')
merged_df = pd.merge(fin_df_3, course_genres_df, left_on='item', right_on='COURSE_ID')

merged_df.iloc[:,1:4]
fin_df['enrollments'].mean()
