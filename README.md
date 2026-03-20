# Fake News Detection Project model README

## Project Overview
This project focuses on building a fake news detection system using various machine learning classification algorithms. The goal is to classify news articles as either 'fake' or 'true' based on their textual content. The dataset consists of news articles, each labeled as either fake or true.

## Dataset
The dataset used for this project comprises two CSV files: `Fake.csv` and `True.csv`.

- `Fake.csv`: Contains news articles identified as fake. Each entry includes a 'title', 'text', 'subject', and 'date'.
- `True.csv`: Contains news articles identified as true. Each entry includes a 'title', 'text', 'subject', and 'date'.

### Data Preparation Steps:
1.  **Labeling**: A `target` column was added to both dataframes, with `0` for fake news (`df_fakenws`) and `1` for true news (`df_truenws`).
2.  **Manual Testing Set**: The last 15 rows from both `df_fakenws` and `df_truenws` were extracted to create `df_fake_manual_testing` and `df_true_manual_testing` respectively. These were then concatenated to form `df_manual_testing.csv` for future manual evaluation.
3.  **Concatenation**: The remaining `df_fakenws` and `df_truenws` were merged into a single dataframe `df_merge`.
4.  **Feature Selection**: The 'title', 'subject', and 'date' columns were dropped, retaining only the 'text' and 'target' columns for model training.
5.  **Text Preprocessing**: A `wordopt` function was applied to the 'text' column to:
    -   Convert text to lowercase.
    -   Remove square brackets, non-alphanumeric characters, URLs, HTML tags, punctuation, newline characters, and words containing digits.
6.  **Sampling**: The merged dataframe was sampled with `frac=3` and `replace=True` to create a larger, potentially more balanced dataset for training.
7.  **Shuffling and Resetting Index**: The dataframe was shuffled and its index was reset.

## Methodology
The project employs a supervised machine learning approach to classify news articles. The following steps outline the methodology:

1.  **Data Splitting**: The preprocessed data was split into training and testing sets using `train_test_split` with a test size of 25%.
2.  **Text Vectorization**: `TfidfVectorizer` was used to convert the textual data into numerical features (TF-IDF scores), which machine learning models can process.
3.  **Model Training**: Four different classification models were trained on the TF-IDF vectorized training data:
    -   Logistic Regression (`LR`)
    -   Decision Tree Classifier (`DT`)
    -   Gradient Boosting Classifier (`GBC`)
    -   Random Forest Classifier (`RFC`)
4.  **Prediction**: Each trained model made predictions on the test set.
5.  **Evaluation**: The accuracy of each model was calculated using `accuracy_score`.
6.  **Manual Testing Function**: A function `manual_testing` was created to allow a user to input a news article and receive a classification (Fake News or Not A Fake News) based on a majority vote from the four trained models.

## Results
The accuracy scores for each trained model on the test set are as follows:

| Model                        | Accuracy |
| :--------------------------- | :------- |
| Logistic Regression          | 0.9950   |
| Decision Tree Classifier     | 0.9993   |
| Gradient Boosting Classifier | 0.9966   |
| Random Forest Classifier     | 0.9987   |

### Comparison of Model Accuracy Scores
```
# The bar plot generated in the notebook visually compares these accuracies.
```

## Insights
-   All four models demonstrated very high accuracy, indicating that TF-IDF features are highly effective for fake news detection on this dataset.
-   The Decision Tree Classifier achieved the highest accuracy, closely followed by the Random Forest Classifier, suggesting that tree-based models perform exceptionally well in this classification task.
-   Logistic Regression, while performing well, had a slightly lower accuracy compared to the ensemble and tree-based models.
-   The Gradient Boosting Classifier also showed strong performance.

## Conclusion
The project successfully developed a fake news detection system capable of classifying news articles with high accuracy using various machine learning models. The preprocessing steps and TF-IDF vectorization proved crucial for achieving these results. The manual testing function provides a practical way to test the models with new input, utilizing an ensemble (majority vote) approach for robustness.

## Author

Subhajit Das -
Data Science & Machine Learning Enthusiast

## License

This project is licensed under the MIT License....

MIT License

Copyright (c) 2026 Subhajit Das

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
