import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import directed_hausdorff,pdist


#Random state
rs = 64

#test dataset, containing test users to whom we want to make course recommendations
test_users_df = pd.read_csv('rs_content_test.csv')

#The profile dataframe contains the course interests for each use
profile_df = pd.read_csv('user_profile.csv')

#Course Genres dataframe
course_genres_df = pd.read_csv('course_genre.csv')


def compute_silhouette_score(courses, sim_scores):
    # Calculate the average similarity score for each course
    avg_sim_scores = []
    for i, course in enumerate(courses):
        avg_sim_scores.append(np.mean(sim_scores[i]))

    # Calculate the silhouette score for each course
    silhouette_scores = []
    for i, course in enumerate(courses):
        # Calculate the difference between the average similarity score for this course
        # and the average similarity scores for the other courses
        diff = []
        for j, other_course in enumerate(courses):
            if i != j:
                diff.append(abs(avg_sim_scores[i] - avg_sim_scores[j]))

        # Calculate the silhouette score for this course
        silhouette_scores.append(max(diff) / (avg_sim_scores[i] + np.min(diff)))

    # Return the average silhouette score over all courses
    return np.mean(silhouette_scores)

# #Define the vectors

# #Define the unseen courses vector
all_courses = set(course_genres_df['COURSE_ID'].values)

# Define the test_user_ids vector
test_users = test_users_df.groupby(['user']).max().reset_index(drop=False)
test_user_ids = test_users['user'].to_list()

#Reload the dataframes

#test dataset, containing test users to whom we want to make course recommendations
test_users_df = pd.read_csv('rs_content_test.csv')

#The profile dataframe contains the course interests for each use
profile_df = pd.read_csv('user_profile.csv')

#Course Genres dataframe
course_genres_df = pd.read_csv('course_genre.csv')

# We first reload all datasets again, and create an empty dictionary to store the results
res_dict = {}

# Only keep the score larger than the recommendation threshold
# The threshold can be fine-tuned to adjust the size of generated recommendations
score_threshold = 5

#Define generate_recommendation_scores() to generate recommendation score for all users.
def generate_recommendation_scores_dot_product():
    users = []
    courses = []
    scores = []
    for user_id in test_user_ids:
        # get user vector for the current user id
        test_user_profile = profile_df[profile_df['user'] == user_id]
        test_user_vector = test_user_profile.iloc[:, 1:].values

        # get the unknown course ids for the current user id
        enrolled_courses = test_users_df[test_users_df['user'] == user_id]['item'].to_list()
        unknown_courses = all_courses.difference(enrolled_courses)
        unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
        unknown_course_ids = unknown_course_df['COURSE_ID'].values
        unknown_course_vectors = unknown_course_df.iloc[:, 2:].values
        #print(unknown_course_vectors[0])
        # reshape the user vector to have the same number of columns as the course vectors
        test_user_vector = test_user_vector.reshape(1, -1)
        
        # use np.dot() to get the recommendation scores for each course
        recommendation_scores = np.dot(test_user_vector, unknown_course_vectors.T)
        recommendation_scores = np.transpose(recommendation_scores)

        # Append the results into the users, courses, and scores list
        for i in range(len(unknown_course_ids)):
            score = recommendation_scores[i]
            # Only keep the courses with high recommendation score
            if score >= score_threshold:
                # Get the float value from the score ndarray
                score_float = score.item(0)
                users.append(user_id)
                courses.append(unknown_course_ids[i])
                scores.append(score_float)
    return users, courses, scores


users, courses, scores = generate_recommendation_scores_dot_product()
res_dict['USER'] = users
res_dict['COURSE_ID'] = courses
res_dict['SCORE'] = scores
res_df_1 = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
silhouette_score_1 = compute_silhouette_score(courses, scores)
res_df_1

# Group the data by the 'Course' column and count the number of rows for each course
course_counts = res_df_1.groupby('COURSE_ID').size().reset_index(name='number_of_reccomendations')
# Sort the dataframe by the 'size' column in descending order
sorted_courses = course_counts.sort_values(by='number_of_reccomendations', ascending=False)
#Reset index
sorted_courses = sorted_courses.reset_index(drop=True)

# Select the top 10 courses
top_10_courses_model_1 = sorted_courses.head(10)

#Average number of reccomended courses
average_reccomendations_model_1 = sorted_courses['number_of_reccomendations'].mean()

average_reccomendations_model_1

#Change the score_threshold hypyer parameter
score_threshold = 10

users, courses, scores = generate_recommendation_scores_dot_product()
res_dict['USER'] = users
res_dict['COURSE_ID'] = courses
res_dict['SCORE'] = scores
res_df_2 = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
silhouette_score_2 = compute_silhouette_score(courses, scores)
res_df_2

# Group the data by the 'Course' column and count the number of rows for each course
course_counts = res_df_2.groupby('COURSE_ID').size().reset_index(name='number_of_reccomendations')
# Sort the dataframe by the 'size' column in descending order
sorted_courses = course_counts.sort_values(by='number_of_reccomendations', ascending=False)
#Reset index
sorted_courses = sorted_courses.reset_index(drop=True)

# Select the top 10 courses
top_10_courses_model_2 = sorted_courses.head(10)

#Average number of reccomended courses
average_reccomendations_model_2 = sorted_courses['number_of_reccomendations'].mean()

average_reccomendations_model_2

#Change the score_threshold hypyer parameter
score_threshold = 20

users, courses, scores = generate_recommendation_scores_dot_product()
res_dict['USER'] = users
res_dict['COURSE_ID'] = courses
res_dict['SCORE'] = scores
res_df_3 = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
silhouette_score_3 = compute_silhouette_score(courses, scores)
res_df_3

# Group the data by the 'Course' column and count the number of rows for each course
course_counts = res_df_3.groupby('COURSE_ID').size().reset_index(name='number_of_reccomendations')
# Sort the dataframe by the 'size' column in descending order
sorted_courses = course_counts.sort_values(by='number_of_reccomendations', ascending=False)
#Reset index
sorted_courses = sorted_courses.reset_index(drop=True)

# Select the top 10 courses
top_10_courses_model_3 = sorted_courses.head(10)

#Average number of reccomended courses
average_reccomendations_model_3 = sorted_courses['number_of_reccomendations'].mean()

average_reccomendations_model_3

number_of_reccomendations = [average_reccomendations_model_1, average_reccomendations_model_2, average_reccomendations_model_3]

maximum = max(number_of_reccomendations)
position = number_of_reccomendations.index(maximum)
minimum = min(number_of_reccomendations)
print(maximum)
print(minimum)

import matplotlib.pyplot as plt
import seaborn as sns
model_number = ["5", "10", "20"]
# create a dataframe with the values and names
df_results = pd.DataFrame({'Avg number of reccomendations': number_of_reccomendations, 'Score threshold': model_number})

# use seaborn to plot the bar chart
sns.barplot(x='Score threshold', y='Avg number of reccomendations', data=df_results)
plt.title('Avg Number of reccomendations from each model (per user)')
plt.plot()

number_of_reccomendations

silhouette_score_list = [silhouette_score_1, silhouette_score_2, silhouette_score_3]

# create a dataframe with the values and names
df_accuracy = pd.DataFrame({'Silhoette score': silhouette_score_list, 'Score threshold': model_number})

# use seaborn to plot the bar chart
sns.barplot(x='Score threshold', y='Silhoette score', data=df_accuracy)
plt.title('Performance of each model')
plt.ylim(0.65, 0.88)
plt.plot()

silhouette_score_list = [0.855591555111784, 0.7048235135407214, 0.735516692120864]
silhouette_score_1 = silhouette_score_list[0]
silhouette_score_2 = silhouette_score_list[1]
silhouette_score_3 = silhouette_score_list[2]

top_10_courses_model_1

merged_df = pd.merge(top_10_courses_model_1, course_genres_df, left_on='COURSE_ID', right_on='COURSE_ID')

merged_df.iloc[:, :3]