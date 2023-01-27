# Course Recommendor System IBM

## Overview of project

The aim of this project was to use various machine learning models for a course reccomendor system. The dataset which was used was the IBM Course Recommendations dataset which is a collection of data related to IBM courses (specifically those on python) and the interactions students had with them (these were saved as CSVs)

For this project both supervised and unsupervised models were used which were then compared with each other. The project itself consists of several notebooks (listed below) along with a pdf of a summary of the project. This project was part of the work done for the IBM Machine Learning Professional certificate.

The machine learning models used for this project included:

##### Unsuperivsed Learning

- ***Using dot product to compare vectors for reccomendations***
- ***Using Bag of Words (Bows) and a similarity matrix***
- ***Clustering and PCA***

#### Supervised Learning


- ***KNN from surprise library***
- ***NMF from surprise library***
- ***Tensor flow Neural Network classifier***


## list of Notebooks in this project:

### EDA
- EDA IBM capstone project

### Unsupervised Learning
- Unsupervised (clustering) IBM capstone project

Flowchart of notebook:
![Screenshot_20230106_062328](https://user-images.githubusercontent.com/68299933/215196386-54d62240-94dc-46f8-bd72-3c43a53ea530.png)

- Unsupervised (course similarity) IBM Capstone project

Flowchart of notebook:
![Screenshot_20230106_113111](https://user-images.githubusercontent.com/68299933/215196627-b730303a-4079-4c8c-814f-05517a1a1ce7.png)

- Unsupervised (course vectors) IBM capstone project

Flowchart of notebook:
![Screenshot_20230106_114034](https://user-images.githubusercontent.com/68299933/215196933-e54e5f8e-7748-49ea-a598-d1778b4c32f8.png)



### Supervised Learning
- Supervised (KNN) IBM capstone project

Flowchart of notebook:
![Screenshot_20230106_114406](https://user-images.githubusercontent.com/68299933/215197123-0f4d8421-f24a-4a90-8937-1c599199f24c.png)

- Supervsied (NMF) IBM capstone project

Flowchart of notebook:
![Screenshot_20230106_121950](https://user-images.githubusercontent.com/68299933/215197225-c9149ca0-fdfe-4b46-889f-93d1a108bf70.png)

- Supervised (Neural Network) IBM capstone project

Flowchart of notebook:
![Screenshot_20230106_122119](https://user-images.githubusercontent.com/68299933/215197276-298b1169-f496-46f2-8326-bf0df7c9b4ae.png)




## Summary of each notebook

### EDA IBM capstone project

This notebook conducted an Exploratory Data Analysis (EDA) of the data. This was done through:

- A barchart  to obtain the balance of genres in the dataset (figure 1)

![Screenshot_20230106_063319](https://user-images.githubusercontent.com/68299933/215202863-acab82ed-4269-4560-b3d8-a89d097f7a07.png)
(figure 1)

- A histogram plot to check the distribution of the dataset (figure 2)

![Screenshot_20230106_063601](https://user-images.githubusercontent.com/68299933/215203057-7f5b77d4-8993-4d7b-87f2-02e7a7c88664.png)
(figure 2)

-A dataframe of the 20 most popular courses to see what courses are most likely to be reccomended to users (figure 3)

![Screenshot_20230106_094652](https://user-images.githubusercontent.com/68299933/215203436-13c22dc6-e48a-4d84-bc07-6ec94d85bd42.png)
(figure 3)

-A wordcloud (figure 4) to visually see what key words appear the most and used stopwords was used to eliminate common english words
![Screenshot_20230106_082341](https://user-images.githubusercontent.com/68299933/215203717-6c720539-ef07-4165-874e-419b803afe49.png)
(figure 4)



The libraries used for this notebook were:

- ***Pandas***
- ***numpy***
- ***matplotlib***
- ***seaborn***
- ***wordcloud*** which was used to import ***WordCloud, STOPWORDS and ImageColorGenerator***









## Summary of project (Conclusions)



