# Naveen Mukkatt's Submission for the ACM Research coding challenge (Fall 2022)

## 1 Motivation
Dataset: https://www.kaggle.com/datasets/chancev/carsforsale 

Upon seeing the dataset, I struggled to find a unique problem to solve. Analysis of data was nice, but I wanted to create something that had real-world applications and used the data to give feedback to a potential user. I did not have much experience with Python at all, and I had no experience with coding up machine learning. I was familiar with basic data cleaning and rudimentary plotting in R, but I knew Python was widely used. Therefore, I decided to go for a machine learning algorithm that would predict some aspect of the car based off of the data given, and implement it in Python, both of which were very new to me.

After thinking about what I wanted to predict, I saw a column named "DealType", which rated deals off of whether they were great, good, or fair. This gave me the idea to attempt to create a model that would use other features of the car to predict this value, since whether a car was a good deal or not probably came from these other features, and wasn't some independent variable of the car. With this in mind, I set out to create a model that would classify cars into these three types depending on other variables. After searching for a suitable model, I came across the K-Nearest Neighbors (KNN) algorithm, which classifies new data points based on other points that the model determines are similar to it. A supervised learning algorithm like KNN was perfect for what I wanted, and I decided that this would be a fair bit easier to implement, since I was still new to Python and ML. 

I needed to choose variables that I thought would impact the deal. Variables like VIN and Stock were out of the question, as those were long strings that were extremely varied and almost useless for the scale of this problem. In addition, I discarded state and zip codes, because as useful as that information could be (cost of living taken into account), it was extremely complex and I only had 9,000 rows of data, which would probably not be enough to provide sufficient data for every zip code.

Instead, I chose more standard variables, like ratings by consumers, MPG, fuel type, mileage, year of the car, etc. Of course, price was included, since that was the biggest factor in determining whether a certain sale was a good deal.

## 2 Data Cleaning
This was something I was more familiar with due to having taken STAT 3355. However, most of the cleaning was done in R in that class. Luckily, Python's pandas package imitated R quite nicely. My goal was to eliminate rows with unusable data (like NaN and missing values) and ensure that the rest of the variables I wanted to use were in categorical or numerical value. To this end, I standardized categorical variables, grouping multiple string that meant the same thing into one group ("front wheel drive", "front-wheel drive", "FWD"), and I eliminated nonsense data points in numerical data (0 MPG doesn't make any sense). Most importantly, I made sure that my DealType was categorical in nature, as the KNN algorithm requires categories for data points to be classified into. The three valid values DealType could be were "Great", "Good", and "Fair". 

## 3 Models
### 3.1 Heatmap
I wanted to see if there were any other numerical variables I should be using, but it was hard for me to tell how variables were linked. After some searching, I found the Seaborn heatmap, which allowed me to analyze correlations between different variables. I was very interested in the correlation of Price with other variables due to its outsized impact on DealType, and I found interesting correlations that were explained in the Python notebook.

### 3.2 KNN Pitfalls
Finally, it was time to go for KNN. But I quickly ran into a problem: I discovered that the KNN algorithm does not work very well when the determining variables are themselves categorical. After looking around, I came across one-hot encoding, which essentially converted the categorical data into numerical data. But here there was yet another problem. I realized that KNN operated by finding the distance between every point, but the values of some features were incredibly large. Some values for MPG were over 100, while all of the one-hot encoding turned features into 1s and 0s, so MPG would have a massively disproportionate impact. Again, I was able to find a transformation of the data that would rectify this problem: scikit came with a StandardScaler that would scale data down to a more reasonable number. After concatenating my categorical and numerical data, 

### 3.3 Implementation and Evaluation
Here, I was very lucky to find a tutorial on how to implement KNN with the data. I was able to understand the motivation for train-test splits, and I chose a much larger range of k than was given, which paid off because it allowed me to see a clearer peak in accuracy followed by the dropoff when k became too large. I plotted the accuracy of each k-value, found that k performed the best around values of 20 to 40 (although not incredibly well - only around 61% compared to 55-58%, but still better than the default 33%). However, I wanted to see what the predictions were for each category. Luckily, I was familiar with the concept of a confusion matrix, and after searching around I found exactly what I was looking for in the documentation of scikit-learn. I was happy that it performed generally well on each category, predicting correctly for each category more than it predicted incorrectly.

## Sources

Ideas for Classification Algorithms:
https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501

Encoding and scaling of data:
https://towardsdatascience.com/categorical-variables-for-machine-learning-algorithms-d2768d587ab6 
https://www.kdnuggets.com/2021/05/deal-with-categorical-data-machine-learning.html

Applying KNN to a mixed dataset:
https://stackoverflow.com/questions/50335203/how-to-apply-knn-on-a-mixed-datasetnumerical-categorical-after-doing-one-hot

Merging two dataframes after applying transformers:
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html
https://stackoverflow.com/questions/47655296/pandas-merge-two-datasets-with-same-number-of-rows

Implementing KNN:
https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75

Metrics:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html

Ignoring warnings: 
https://stackoverflow.com/questions/9031783/hide-all-warnings-in-ipython


