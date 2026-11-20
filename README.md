# EduTrends Predictions

By Fermin Garcia

## Project Description

Using open-source student data from Kaggle, I built a model that uses parents' socioeconomic information to identify students who are statistically more likely to struggle academically, before they ever enter a classroom. The goal is to help schools direct support to disadvantaged students early so they get the education they deserve.

## Project Goal

Use machine learning to identify students who are likely to struggle academically based on family and socioeconomic factors.

## Initial Hypotheses

- Does parents' education level affect a student's risk category?
- Does receiving free or reduced-price lunch affect a student's risk category?
- Does completing a test prep course affect a student's risk category?

- H<sub>0</sub>: There is no correlation between the risk category and the individual features.
- H<sub>a</sub>: There is a correlation between the risk category and the individual features.

## Project Plan

- **Planning:** The steps for this project are laid out in this README.

- **Acquisition:** The data comes from Kaggle. It was loaded into Google Sheets and read into Python from there.

- **Preparation:** Removed unnamed columns that duplicated the index and standardized the structure of the data. To keep the data consistent without discarding any student's record, missing values were imputed so there are no nulls. A final score was calculated as the average of the math, reading, and writing scores.

- **Preprocessing:** Converted the parent-related columns to true/false values and then to 1/0. Added a risk label that marks a student as high risk if their final score is 75 or below.

- **Exploration:** Tested which features had a statistically significant relationship with the risk label, focusing on features that describe the parents rather than the student, and measured which ones contributed most to a student's risk.

- **Modeling:** Trained several classification models to predict high-risk students. The best model matched the baseline F1 score of 81% but did not beat it. Because of that, the final approach uses each model's predicted probability of being high risk, which gives a more useful sense of how at-risk a student is than a yes/no label.

- **Delivery:** Findings are packaged in `final_report.ipynb`.

- **Reproducibility:** Running the notebooks from top to bottom reproduces the results, assuming the packages in `requirements.txt` are installed.

- **Production:** New student records can be submitted through this [Google Form](https://forms.gle/ewZwvsauLAUZXGkD9). After submitting the form, run:

  ```bash
  python3 production.py
  ```

  The script pulls the latest form responses and returns each student's predicted probability of being high risk. The student ID on the form is a made-up alphanumeric value, so no real student information is needed.

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
| risk_rating                            | The target: 1 if the student is high risk (final score of 75 or below), otherwise 0 |
