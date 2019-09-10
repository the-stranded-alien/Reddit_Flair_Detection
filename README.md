# Reddit Flair Detector

A Reddit Flair Detection & Reddit Post's Statistical Analytics web application to detect flairs of India subreddit posts using Machine Learning and Deep Learning algorithms and to find some patterns within data. The application can be found live at https://flair-api-json.herokuapp.com/ .
For statistical results you can directly check the live website https://brownie-analytics.herokuapp.com/ .

### Directory Structure

The directory is a ***Web Application*** for hosting on *Heroku* servers. The description of files and folders can be found below:

  1. [Analysis](https://github.com/the-stranded-alien/Reddit_Flair_Detection/tree/master/Analysis) - Folder containing Python Code to do analytical study on entire data with results.
  2. [Brownie-Analytics](https://github.com/the-stranded-alien/Reddit_Flair_Detection/tree/master/Brownie-Analytics) - Folder having Code to Node-Js Web-App for Stats.
  3. [Clean_Data](https://github.com/the-stranded-alien/Reddit_Flair_Detection/tree/master/Clean_Data) - Contains the cleaned data in CSV format.
  4. [Models](https://github.com/the-stranded-alien/Reddit_Flair_Detection/tree/master/Models) - Folder containing all tried ML models and the final saved model.
  5. [MongoDB](https://github.com/the-stranded-alien/Reddit_Flair_Detection/tree/master/MongoDB) - Folder containing Python code to upload data to MongoDB and MongoDB instance.
  6. [Other_Jupyter_Notebooks](https://github.com/the-stranded-alien/Reddit_Flair_Detection/tree/master/Other_Jupyter_Notebooks) - Folder containing other Jupyter Notebooks to clean & merge data and a prediction tester tool. 
  7. [Prediction_API](https://github.com/the-stranded-alien/Reddit_Flair_Detection/tree/master/Prediction_API) - Folder having Flask Web-App for predicting Reddit Flairs.
  8. [Raw_Data](https://github.com/the-stranded-alien/Reddit_Flair_Detection/tree/master/Raw_Data) - Folder containing all flair data files fetched from Reddit API.
  
### Codebase

The code has been developed using Python and JavaScript programming languages. Utilizing Python's powerful text processing and machine learning modules certian ML and Deep Learning Models are made. The [Flair Prediction Web-application](https://flair-api-json.herokuapp.com/) has been developed using Flask web framework and the [Statistical Analysis Web-application](https://brownie-analytics.herokuapp.com/) has been made using NodeJs. Both of them are hosted on Heroku Servers.

### Project Execution

  1. Open the `Terminal`.
  2. Clone the repository by entering `git clone https://github.com/radonys/Reddit-Flair-Detector.git`.
  3. Ensure that `Python3` and `pip` is installed on the system.
  4. Create a `virtualenv` by executing the following command: `virtualenv -p python3 env`.
  5. Activate the `env` virtual environment by executing the follwing command: `source env/bin/activate`.
  6. Enter the cloned repository directory and execute `pip install -r requirements.txt`.
  7. Enter `python` shell and `import nltk`. Execute `nltk.download('stopwords')` and exit the shell.
  8. Now, execute the following command: `python manage.py runserver` and it will point to the `localhost` with the port.
  9. Hit the `IP Address` on a web browser and use the application.
  
### Dependencies

The following dependencies can be found in [requirements.txt](https://github.com/radonys/Reddit-Flair-Detector/blob/master/requirements.txt):

  1. [praw](https://praw.readthedocs.io/en/latest/)
  2. [scikit-learn](https://scikit-learn.org/)
  3. [nltk](https://www.nltk.org/)
  4. [Django](https://www.djangoproject.com/)
  5. [bs4](https://pypi.org/project/bs4/)
  6. [pandas](https://pandas.pydata.org/)
  7. [numpy](http://www.numpy.org/)
  
### Approach

Going through various literatures available for text processing and suitable machine learning algorithms for text classification, I based my approach using [[2]](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568) which described various machine learning models like Naive-Bayes, Linear SVM and Logistic Regression for text classification with code snippets. Along with this, I tried other models like Random Forest and Multi-Layer Perceptron for the task. I have obtained test accuracies on various scenarios which can be found in the next section.

The approach taken for the task is as follows:

  1. Collect 100 India subreddit data for each of the 12 flairs using `praw` module [[1]](http://www.storybench.org/how-to-scrape-reddit-with-python/).
  2. The data includes *title, comments, body, url, author, score, id, time-created* and *number of comments*.
  3. For **comments**, only top level comments are considered in dataset and no sub-comments are present.
  4. The ***title, comments*** and ***body*** are cleaned by removing bad symbols and stopwords using `nltk`.
  5. Five types of features are considered for the the given task:
    
    a) Title
    b) Comments
    c) Urls
    d) Body
    e) Combining Title, Comments and Urls as one feature.
  6. The dataset is split into **70% train** and **30% test** data using `train-test-split` of `scikit-learn`.
  7. The dataset is then converted into a `Vector` and `TF-IDF` form.
  8. Then, the following ML algorithms (using `scikit-learn` libraries) are applied on the dataset:
    
    a) Naive-Bayes
    b) Linear Support Vector Machine
    c) Logistic Regression
    d) Random Forest
    e) MLP
   9. Training and Testing on the dataset showed the **Random Forest** showed the best testing accuracy of **77.97%** when trained on the combination of **Title + Comments + Url** feature.
   10. The best model is saved and is used for prediction of the flair from the URL of the post.
    
### Results

#### Title as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.6011904762      |
| Linear SVM                 | 0.6220238095      |
| Logistic Regression        | **0.6339285714**  |
| Random Forest              | 0.6160714286      |
| MLP                        | 0.4970238095      |

#### Body as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.2083333333      |
| Linear SVM                 | 0.2470238095      |
| Logistic Regression        | 0.2619047619      |
| Random Forest              | **0.2767857143**  |
| MLP                        | 0.2113095238      |

#### URL as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.3005952381      |
| Linear SVM                 | **0.3898809524**  |
| Logistic Regression        | 0.3690476190      |
| Random Forest              | 0.3005952381      |
| MLP                        | 0.3214285714      |

#### Comments as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.5357142857      |
| Linear SVM                 | 0.6190476190      |
| Logistic Regression        | **0.6220238095**  |
| Random Forest              | 0.6011904762      |
| MLP                        | 0.4761904762      |

#### Title + Comments + URL as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.6190476190      |
| Linear SVM                 | 0.7529761905      |
| Logistic Regression        | 0.7470238095      |
| Random Forest              | **0.7797619048**  |
| MLP                        | 0.4940476190      |


### Intuition behind Combined Feature

The features independently showed a test accuracy near to **60%** with the `body` feature giving the worst accuracies during the experiments. Hence, it was excluded in the combined feature set.

### References

1. [How to scrape data from Reddit](http://www.storybench.org/how-to-scrape-reddit-with-python/)
2. [Multi-Class Text Classification Model Comparison and Selection](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568)
