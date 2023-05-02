# EduTrends Predictions
- By Fermin Garcia


# Project Description

The purpose of this project is to create a model to indenify risk categories for students based off socioeconomic status 

# Project Goal

The goal of this project is to identify students with high risk ratings to focus additional resources for those students to help them succeed in their acedemics.

# Initial Hypotheses

- Does parents education affect the risk category of students?
- Do students who are on free/reduce lunch have an affect on the risk category
- Does completing a test prep affect risk category ?
- If we cluster the alcohol content into groups, will wines with high alcohol content have a higher quality score on average than wines with lower alcohol content?
- Does volatile acidity level have a correlation with quality of the wine?

- H$_0$: There is no correlation between the risk category and individual features.
- H$_a$:There is a correlation between the risk category and individual features.


# Project Plan

- Planning - The steps required to be taken during this project will be laid out in this readme file. 

- Acquisition - Data will be acquired from https://kaggle.com which is a publicly available dataset. Once the files have been downloaded, the data will be combined into one local csv file by our acquire.py script.

- Preparation - We will do inital exploration of our data to determine if there are outliers or null values. If the dataset contains outliers or nulls, we will make determinations on what to do with each based on the effect on the overall dataset. We will rename columns in order to make them easier to understand or work with. If there are any data types that are not best suited for the data, we will change the data types. We will also be splitting our data into train, validate and test groups to limit potential data poisoning.

- Preparation - We will do inital exploration of our data to determine if there are outliers or null values. If the dataset contains outliers or nulls, we will make determinations on what to do with each based on the effect on the overall dataset. This is a rare instance where we decided to keep our duplicates. We made this determination based off visual review of our duplicates, while there are some across the board, some of the numerical values that extend to four decimal places are identical. This lead us to conclude that some scores are destined to be duplicated as wine must meet specific range of qualifications to qualify as wine, so duplicate rows are sure to be duplicated.We will rename columns in order to make them easier to understand or work with. If there are any data types that are not best suited for the data, we will change the data types. We will also be splitting our data into train, validate and test groups to limit potential data poisoning. Since we are approaching the project as a regression problem.

- Exploration - We will explore the  data to find statistically valid correlations to our target variable. We will also be looking at combinations of variables that could be useful for clustering. in this section we created two new categories. Final score and risk categories. Final score was created based off the average of the three score, reading, writing and math. risk categories were created based on clustering patterns we were able to identify from final score. We were able to find three distinct groups in final score. We labeled these low-risk, at risk, high risk. low risk were for students who were at or above passing. At risk were students who were at or below passing. High risk were for students who were below passing. Since the risk categories were created from final score we decided to explore on final score to see what factors affected the class that risk categories were dirived from.

- Modeling - We will be approaching the problem as a classification problem. Therefore we will be making multiple models using classification algorithms such as K Nearest neighbour (KNN), Descion Tress and Random Forest. We will be creating a baseline model using the mode of risk categories from our training dataset. We will be evaluating our models using the recall for the high risk categories.


        Recall is a performance metric that measures the proportion of relevant items that were correctly identified by a model. In the context of binary classification, recall is the ratio of true positives to the sum of true positives and false negatives.
        In the task of predicting high-risk students based on socio-economic class, recall would be a relevant metric to consider because it emphasizes the importance of correctly identifying all the students who are actually at high risk, even if some non-high risk students are mistakenly identified as high risk. This is particularly important in situations where false negatives (i.e., high-risk students who are not identified) can have serious consequences, such as the students not receiving necessary interventions or support.

        By using recall as a performance metric, we can evaluate how well our model is able to correctly identify high-risk students from different socio-economic classes, and we can compare the performance of different models or algorithms to choose the best one for our specific needs.

- Delivery - We will be packaging our findings in a final_report.ipynb file.
