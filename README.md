# EduTrends Predictions
- By Fermin Garcia


# Project Description

With open source data from kaggle.com, Using parents'’ socioeconomic status, we created a model to identify high-risk students who were statistically more likely to struggle academically before entering a classroom.  My analysis and modeling will ultimately enable the support of disadvantaged youths pursuing the education they deserve.

# Project Goal

Our goal was to use machine learning algorithms to idnetify students who are more to struggle acedemically based off parents demographics.  

# Initial Hypotheses

- Does parents education affect the risk category of students?
- Do students who are on free/reduce lunch have an affect on the risk category
- Does completing a test prep affect risk category ?


- H$_0$: There is no correlation between the risk category and individual features.
- H$_a$:There is a correlation between the risk category and individual features.


# Project Plan

- Planning - The steps required to be taken during this project will be laid out in this readme file. 

- Acquisition - Dwe acquired our data from Kaggle.com, imported into google sheets and used python to import the data. 

- Prepare - In prepare we removed the unnamede olumns which were just redos of the index. We also standardized the structure of the data. to keep data integrity consistent and to avoid stripping a student of their voice we imputated the values to ensure no null values in our data structure. We also calucated the final score between the math, reading, and writing scores to calulate overall average.

- Preprocessing - In preprocessing we identified convderted coloumns related to the parents to true or false then converted the boolean values (True or False) to their biary repersentation. We additionally calualted a risk score that rates a student high risk if their final score was at or below a 75 overall average. 

- Exploration - In our exploritory analysis We explored the statstical significance of the data and which values correlate to the parents and not the students. We determine which values had the biggest contribution to the students risk score.

- Modeling - In modeling we used serveral classification models to attempt to predict high risk students. Our strongest one that matched our base line model of an F1 score of 81%. We were unable to beat our base line. We decided to use the the predicted proabability to better gauge the at risk student. 

- Delivery - We will be packaging our findings in a final_report.ipynb file.

- Reproducability:
    Running these notebooks from top to bottom will result in the same results. This is also under the assumption that you have all the installs from requirements.txt. 

- Production:
# Link to google form `https://forms.gle/ewZwvsauLAUZXGkD9`
    After completing the form ypu can run the following in the command line
`python3 production.py`
    This will pull the latest data in the google form sheet and will give a predicted proability of being high risk. If you fill out the form the student id is a fictious alpha numeric number. 


## Data Dictionary

| Variable Name                          | Description                                                                                                                                                                                                                      |
|----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| is_male                                | A binary variable indicating whether the student is male (1) or not (0)                                                                                                                                                        |
| free_reduced_lunch                      | A binary variable indicating whether the student receives free or reduced-price lunch (1) or not (0)                                                                                                                          |
| test_prep_completed                     | A binary variable indicating whether the student completed a test preparation course (1) or not (0)                                                                                                                            |
| practicesport                          | A binary variable indicating whether the student practices a sport (1) or not (0)                                                                                                                                              |
| is_first_child                         | A binary variable indicating whether the student is the first child in the family (1) or not (0)                                                                                                                               |
| nrsiblings                             | The number of siblings the student has                                                                                                                                                                                          |
| rides_bus                              | A binary variable indicating whether the student rides the bus to school (1) or not (0)                                                                                                                                         |
| math_score                             | The score the student obtained on the math test                                                                                                                                                                                 |
| reading_score                          | The score the student obtained on the reading test                                                                                                                                                                              |
| writing_score                          | The score the student obtained on the writing test                                                                                                                                                                              |
| final_score                            | The final score obtained by the student, which is the sum of the math, reading, and writing scores divided by three                                                                                                                              |
| parent_educ_bachelors_degree           | A binary variable indicating whether the parent has a Bachelor's degree (1) or not (0)                                                                                                                                          |
| parent_educ_high_school                | A binary variable indicating whether the parent has a high school education (1) or not (0)                                                                                                                                      |
| parent_educ_masters_degree             | A binary variable indicating whether the parent has a Master's degree (1) or not (0)                                                                                                                                            |
| parent_educ_some_college               | A binary variable indicating whether the parent has some college education (1) or not (0)                                                                                                                                       |
| parent_educ_some_high_school           | A binary variable indicating whether the parent has some high school education (1) or not (0)                                                                                                                                   |
| parent_marital_status_married          | A binary variable indicating whether the parent is married (1) or not (0)                                                                                                                                                      |
| parent_marital_status_single           | A binary variable indicating whether the parent is single (1) or not (0)                                                                                                                                                       |
| parent_marital_status_widowed          | A binary variable indicating whether the parent is widowed (1) or not (0)                                                                                                                                                      |
| wkly_study_hours_<_5                   | A binary variable indicating whether the student studies less than 5 hours a week (1) or not (0)                                                                                                                                 |
| wkly_study_hours_>_10                  | A binary variable indicating whether the student studies more than 10 hours a week (1) or not (0)                                                                                                                               |
| risk_rating                            | A variable indicating the risk level of the student. |
