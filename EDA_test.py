#Import library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Set random state
rs = 64 

#Csvs for the projects
course_genre_df = pd.read_csv('course_genre.csv')
course_ratings_df = pd.read_csv('course_ratings.csv')

#TODO: make this bit better
print(course_genre_df.head())
print(course_genre_df.shape)
print(course_ratings_df.head())

# Get course counts per genre #TODO: why?
#TODO: optimize the function below using list comprehensions and make it only take [1:]
genre_counts = course_genre_df.iloc[:, 1:].apply(lambda x: x.sum(), axis=0)
print(genre_counts[1:])

# Convert the values in the genre_counts series to integers
genre_counts = genre_counts[1:].apply(int)

# Sort the genre counts in descending order
sorted_genre_counts = genre_counts.sort_values(ascending=False)
print(sorted_genre_counts)

# Plot the sorted genre counts as a bar chart
plt.figure(figsize=(10, 6))

#TODO: Add a comment here
sorted_genre_counts.plot(kind='bar')

# Set the x-axis label
plt.xlabel('Genre')

# Set the y-axis label
plt.ylabel('Course Count')

# Show the plot
plt.show()

# Group the ratings dataframe by the user column and get the size of each group
rating_counts = course_ratings_df.groupby('user').size()

# Get the total number of users after aggregation
num_users = rating_counts.size

# Print the total number of users
print(f'Total number of users: {num_users}')

# Describe ratings counts #TODO: why?
rating_counts.describe()

# Set the figure size
sns.set(rc={'figure.figsize':(10,6)})

# Plot the histogram of rating counts with separated bars
sns.displot(rating_counts, kde=False, bins='auto')

# Set the x-axis label
plt.xlabel('Rating Count')

# Set the y-axis label
plt.ylabel('Number of Users')

# Show the plot
plt.show()

#TODO: simplify between line 77 and line 100

# Group the ratings dataframe by the item column and get the size of each group
enrollment_counts = course_ratings_df.groupby('item').size()

# Sort the enrollment counts in descending order
sorted_enrollment_counts = enrollment_counts.sort_values(ascending=False)

# Get the top 20 courses
top_20_courses = sorted_enrollment_counts[:20]

#Use Pandas merge() method to join the course_df (contains the course title column).
# Convert the top_20_courses series into a dataframe
top_20_courses_df = top_20_courses.reset_index()

# Rename the columns of the top_20_courses_df dataframe
top_20_courses_df.columns = ['item', 'enrollments']

# Rename the COURSE_ID column to item
course_genre_df = course_genre_df.rename(columns={'COURSE_ID': 'item'})

# Merge the course_df and top_20_courses_df dataframes on the item column
merged_df = pd.merge(course_genre_df, top_20_courses_df, on='item')

#Print the merged df
print(merged_df.loc[:, ['TITLE','enrollments']].sort_values(by=['enrollments'], ascending=False).reset_index(drop = True))

#Print total enrollments? #TODO: why? -> could this be in a different place?
print(f"The total course enrollments is {course_ratings_df.shape[0]}")

# Top 20 Percentage enrollments #TODO:(check grammar) and simplify
merged_df['percentage'] = merged_df['enrollments'].apply(lambda x: x / merged_df['enrollments'].sum() * 100)
filtered_df = merged_df.loc[:, ['TITLE', 'enrollments', 'percentage']]
filtered_df['percentage (2 d.p)'] = filtered_df['percentage'].round(2)
print(filtered_df.loc[:, ['TITLE', 'enrollments', 'percentage (2 d.p)']])

#TODO: ADD why?
print(f"The percentage of enrollments of each course is {filtered_df['percentage'].mean()}")
print(f"The mean of enrollments of each course is {filtered_df['enrollments'].mean()}")

#TODO: ADD why a word cloud is used:

#Plot a word cloud from course titles
titles = " ".join(course_genre_df['TITLE'].astype(str))

# English Stopwords
stopwords = set(STOPWORDS)
stopwords.update(["getting started", "using", "enabling", "template", "university", "end", "introduction", "basic"])

# Then, we create a WordCloud object and generate wordcloud from the titles.
wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400)
wordcloud.generate(titles)
plt.axis("off")
plt.figure(figsize=(40,20))
plt.tight_layout(pad=0)
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()