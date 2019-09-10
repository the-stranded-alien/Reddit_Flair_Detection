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
  2. Clone the repository by entering `git clone https://github.com/the-stranded-alien/Reddit_Flair_Detection`.
  3. Ensure that `Python3`, `pip` and `npm` is installed on the system.
  4. Set terminal to Prediction-API folder and type `python app.py` to run on localhost port 5000.
  5. Set terminal to Brownie-Analytics folder and type `npm init` and keep pressing `Enter` untill project is initialized.
  6. Type `npm install` to install all dependencies. 
  7. Run `server.js` file to run Stats website on localhost port 1611.
  
### Dependencies

These are the entire prpject's dependencies :

  1. [praw](https://praw.readthedocs.io/en/latest/)
  2. [scikit-learn](https://scikit-learn.org/)
  3. [nltk](https://www.nltk.org/)
  4. [Flask](https://palletsprojects.com/p/flask/)
  5. [pandas](https://pandas.pydata.org/)
  6. [numpy](http://www.numpy.org/)
  7. [Keras](https://keras.io/)
  8. [NodeJs](https://nodejs.org/en/)
  9. [ExpressJs](https://expressjs.com/)
  
### Approach (for Flair Detection)

After reading blogs and books about text classification, I concluded that in general cases Naive Bayes Classifier, Random Forests and Recurrent Neural Networks perform better for text classification compared to other models.

The approach taken for the task is as follows:

  1. Collected around 200 India subreddit data for each of the 12 flairs using `praw` module [[1]](http://www.storybench.org/how-to-scrape-reddit-with-python/).
  2. The data had *title*, *score*, *id*, *url*, *comms_num*, *body*, *author*, *comments*, *flair*, and *time-created*.
  3. For **comments**, only top level comments are considered in dataset and no sub-comments are present.
  4. The ***title***, ***comments*** and ***url*** are cleaned by removing bad symbols and stopwords using `nltk`.
  5. Feature considered for the the given task:
    -> Combining Title, Comments and Urls as one feature.
  6. The dataset is split into **70% train** and **30% test** data for Naive-Bayes and Random Forest and **90% train** and **10% test** data for RNN based model using `train-test-split` of `scikit-learn`.
  7. The dataset is then converted into a `Vector` and `TF-IDF` form for Naive-Bayes and Random Forest and into `tokens` form for RNN(LSTM).
  8. Then, the following ML algorithms (using `scikit-learn` libraries and `Keras`) are applied on the dataset:
    
    a) Naive-Bayes
    b) Random Forest
    c) RNN (LSTM)
   9. Training and Testing on the dataset showed the **Random Forest** showed the best testing accuracy of **69.6%** when trained on the combination of **Title + Comments + Url** feature.
   10. The best model is saved and is used for [prediction of the flair from the URL of the post](https://flair-api-json.herokuapp.com/).
      
### Approach (for Flair Detection)

Available statsitical information is visualized using Four types of Graphs:
  
  1.[Number of Posts collected for each flair](https://brownie-analytics.herokuapp.com/analytics/stat1.html):
  2.a)[Total Number of Comments for each flair](https://brownie-analytics.herokuapp.com/analytics/stat2.html):
    b)[Average Number of Comments for each flair](https://brownie-analytics.herokuapp.com/analytics/stat2.1.html):
  3.[Number of Posts of each flair in different Times of Day](https://brownie-analytics.herokuapp.com/analytics/stat3.html):
  4.[Average Score for each flair](https://brownie-analytics.herokuapp.com/analytics/stat4.html):
    
### Results

#### Title + Comments + URL as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.55109962      |
| Random Forest              | **0.69598965**  |
| RNN(LSTM)                  | 0.42248062      |


### Intuition behind Combined Feature

Title, Comments and URL all had meaningful text data related to posts which was useful to predict the post's flair. Body was not taken as a feature because half of the posts didn't had a body and it was making the feature unbalanced.

### References

1. [How to scrape data from Reddit](http://www.storybench.org/how-to-scrape-reddit-with-python/)
2. [Multi-Class Text Classification Model Comparison and Selection](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568)
3. [Multi-Class Text Classification With LSTM](https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17)
4. [Yash Srivastava's Project on Reddit Flair Prediction](https://github.com/radonys/Reddit-Flair-Detector) 
