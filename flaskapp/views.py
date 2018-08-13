from flask import render_template
from flask import request 
from flaskapp import app
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#%% Readability
from nltk.corpus import cmudict
from nltk.tokenize import sent_tokenize, word_tokenize
from random import random
from math import floor
not_punctuation = lambda w: not (len(w)==1 and (not w.isalpha()))
get_word_count = lambda text: len([x for x in word_tokenize(text) if not_punctuation(x)])
get_sent_count = lambda text: len(sent_tokenize(text))
prondict = cmudict.dict()
numsyllables_pronlist = lambda l: len(list(filter(lambda s: s.lower()[-1].isdigit(), l)))

def numsyllables(word):
    try:
        return list(set(map(numsyllables_pronlist, prondict[word.lower()])))
    except KeyError:
        return [0]

def text_statistics(text):
    word_count = get_word_count(text)
    sent_count = get_sent_count(text)
    syllable_count = sum(map(lambda w: max(numsyllables(w)), word_tokenize(text)))
    complex_count = sum(map(lambda w: max(numsyllables(w))>2, word_tokenize(text)))
    return word_count, sent_count, syllable_count, complex_count

def gunning_fog(text):
    word_count, sent_count, syllable_count, complex_count = text_statistics(text)
    return 0.4*(word_count/sent_count + 100*(complex_count/word_count))

#%%Machine Learning
from sklearn.externals import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

analyser = SentimentIntensityAnalyzer()
clf = joblib.load('./models/clf.pkl')
qScl = joblib.load('./models/qscl.pkl')
X80 = pickle.load(open('./feature_ranks/X80.p', "rb" ))
X90 = pickle.load(open('./feature_ranks/X90.p', "rb" ))
feature_ranks = pickle.load(open('./feature_ranks/feature_ranks.p', "rb" ))

@app.route('/')
@app.route('/index')
def index():
    """Render the home page (index.html)"""
    # Create randomized links to prevent Flask from caching the CSS
    default_title = "Decision tree it is!"
    default_subtitle = "Description:"
    default_text = "A Decision tree  is a tree-based supervised learning algorithm used in the predictive analysis of data which helps one to get generalized conclusions. It decides the labels depending on given feature. Now, what is a label and feature here? A  Label  is something that you are attempting to predict or conclude.  Features  are the constraints or conditions that decide, to which the label or the set of given data belongs to. Example\xa0: Sam is a student. We need to check if Sam can finish her homework on time or not.\xa0:P It’s obvious that there are only two possible answers for this: YES or NO. So,  Labels  here are YES and NO, but to predict this we need  Features  like Sam’s writing speed, help from friends, number of pages to write etc This is a basic gist of what Labels and Features mean. To continue with the Decision Tree, we will use the same example. Let us take two feature for simplicity. One is help from her friend, Adi and the other is Sam’s writing speed. Types of Decision Trees ( CART—Classification And Regression Trees): Classification Trees:  These are used to classify things. Example: It can be used to predict if Sam will complete the homework or not. (YES / NO) Regression Trees:  These are usually used to predict values of anything happening in the future. Example: With what speed (pages/min) may Sam complete the homework. Graph Explanation: Let’s draw a two-dimensional graph for the example above first. Representation\xa0: Blue = YES Red = NO In general, the red dots to the lower left say that the average speed of Sam writing is low and Adi’s help is also less. So, it is  not possible  to complete the homework on time. Similarly, the red dots when Sam’s speed is low, but Adi’s helping percentage is high, is still  not possible  for Sam to complete her homework. The red dots when Sam’s speed is high, but Adi’s Help is very less, is  not possible  to complete the homework. But, when Adi’s help is high and Sam’s speed is fast as well, it  is possible  to complete the homework on time. This is represented by the blue dots. How Decision tree\xa0works: Let’s draw lines in the above graph as shown below. Lines need not be straight, but it should divide the graph such that  impurity in each part is reduced . X<7.8 & Y< 38 X<7.8 & Y>38 X>7.8 & Y< 38 X>7.8 & Y>38 In above graph, line  ab,  parallel to Y-axis, cuts X at 7.8(approximately) such that all the points left to the line  ab  are  Red  ( which implies that answer is NO) Similarly, the line  cd  divides the graph such that all points below it are  Red Note: You can see there are a couple of Blue dots which lie below  cd  so you might feel like drawing a line slightly below. But, doing that, you will put many  Red  dots above the line  cd , which causes more impurity in region X>7.3 & Y>10. And this is how decision tree works, it determines as to which part the data belongs to and then classifies them into sets, i.e if data given is, speed=7.5 pages/min and help= 5% then decision tree will classify it has (NO) Entropy: The simplest definition of entropy would be  ‘measure of impurity’.  Entropy lies in between 0 & 1. Mathematical Equation\xa0: E =  - 𝚺  p(xi )log (p( xi )) Example: p(yes) = 9/(9+5) = 9/14 = 0.642 p(no) = 5/(9+5) = 5/14 = 0.3514 Entropy(E) =\u200a—\u200a0.642 log(0.642)\u200a—\u200a0.3514 log(0.3514) = 0.9406 Lesser the entropy, lesser the impurity. Maximum value of entropy is 1 i.e p(YES)=p(NO)= 0.5. Thus, we can say that we have maximum impurity. Here is your decision tree: Information Gain: Information gain is a measure of entropy. The general formula goes like, You might be wondering what is the use of entropy here. Well, the above decision tree is great, but in real time, whether Sam can finish the homework also depends on the number of pages to write and Adi’s writing speed. Let’s take an example of an Automatic Car: You can see three features to decide whether the car should move fast or slow But the problem is we don’t know which feature has a priority over others. Features with the higher priority becomes the parent node And the priority of feature are identified using  information gain IG and priority are directly proportional to each other. Formula of Information Gain ( IG )\xa0: IG= E(parent)\u200a—\u200a𝚺 (weighted average )*E(child) We calculate the information gain for each feature and then arrange them. According to Grade: S is slow and F is fast IG= E(parent)- (0.75*E(left node)+ 0.25*E(right node)) (Weighted average at the left node\xa0: 3 out of 4 possible answers i.e 3/4 or 0.75) IG = 0.3112 Similarly the IG for remaining two features will be, IG(Bumpiness) = E(parent)\u200a—\u200a(0.5 E(bumpy) + 0.5 E(smooth)) = 0 IG(Speed limit) = E(parent)\u200a—\u200a(0.5 E(yes) + 0.5 E(no)) = 1 The Decision Tree would look like this: Root is speed limit with highest IG, 1 followed by Grade and then Bumpiness of the road. Branching termination: You might always wonder, when do we stop branching the tree. Answer: when  overfitting  occurs. Overfitting  is a type of an error in your model which results from an excessive number of training points. An overfitted model shows a curve with higher and lower points, while a properly fitted model shows a smooth curve or a linear regression. The opposite of overfitting is underfitting. Overfitting arises when “Training errors are small and test errors are large” whereas underfitting arises when “Training errors are large and test errors are small”. Pruning\xa0: Pruning  is a technique in machine learning which reduces the size of  decision trees  by removing a few sections of the  tree  that provide little power to classify instances. It reduces the complexity of the final classifier, and hence improves predictive accuracy by the reduction of overfitting. Pre-Pruning and Post-Pruning: Pre-pruning is stopping the growth of a tree before it is completely grown. Post-pruning is allowing the tree to grow with no size limit. After tree completion starts to prune the tree. Pruning reduces the complexity of the tree and also controls the unnecessary growth of the tree. This hence improves the accuracy in return. Pre-pruning is faster than post pruning as it need not wait for complete construction of the decision tree. Pros and Cons of a Decision Tree\xa0: Pros\xa0: →  Simple to analyze and interpret. → Construction of a DT is faster. → Fast prediction in most of the cases. It rather depends on the dataset. Cons\xa0: →  Unstable (A small change in data can lead to a huge difference in results of the model) → Calculations may get tedious if a lot of variable values are present in the data. → If any new scenario comes into picture, it is hard to modify the tree and predict the outputs again. I.e. loss of invention — Written by  Samyuktha Prabhu  and  Aditya Shenoy. Decision tree it is! A powerful building block of random forest algorithms. Keep reading, in the next 5 minutes you will know what is a label what is a feature and how prediction with trees work." 

    css_link = '../static/css/custom.css?q=' + str(random())
    return render_template('index.html', 
                           article_text = default_text,
                           article_title = default_title,
                           article_subtitle = default_subtitle,
                           css_hash=css_link, message = ' ')

@app.route('/result', methods=['GET', 'POST'])
def result():    
    """Render the results page (result.html)"""

    # Create randomized links to prevent Flask from caching the results image
    # and CSS
    image_location = '../static/images/features.png'
    image_link = image_location+'?q=' + str(random())
    css_link = '../static/css/custom.css?q=' + str(random())

    article_title = request.form.get('article_title')
    article_subtitle = request.form.get('article_subtitle')
    article_text = request.form.get('text_Content')

    link_count = int(request.form.get('link_count')[0])
    image_count = int(request.form.get('image_count')[0])
    publication_followers = int(request.form.get('publication_followers'))

    grade_level = gunning_fog(article_text)
    question_count = article_text.count('?')
    snt = analyser.polarity_scores(article_text)
    #%% Machine Learning
    article_word_count = get_word_count(article_text)
    title_word_count = get_word_count(article_title) + get_word_count(article_subtitle)
    #read_time = 0.2*image_count + article_word_count/275
    read_time = 0.2*image_count + article_word_count/275
    X_raw = []
    X_raw.append(image_count)
    X_raw.append(link_count)
    X_raw.append(publication_followers)
    X_raw.append(read_time)
    X_raw.append(grade_level)
    X_raw.append(snt['compound'])
    X_raw.append(snt['neu'])
    X_raw.append(question_count)
    X_raw.append(title_word_count)

    #%%
    X = qScl.transform([X_raw])
    y_pred = clf.predict(X)
    prob = floor(100 * clf.predict_proba(X)[0, 1])
    if y_pred:
        message = 'Great, your post will be popular'
        color_choice = "#20C863"
    else:
        message = 'You should improve your post, see below'
        color_choice = "#CB3335"
    predictive_indexes = feature_ranks.sort_values(by='fval',ascending=False)['order'].values
    predictive_features = ['publication followers', 'read time', 'grade level',
                       'number of links', 'neutrality','sentiment','number of images','length title-subtitle','number of questions']
    # Compute the weighted score of the meta features 
    user_article_score = np.multiply(X,feature_ranks['fval'].values)
    top80_article_score = np.multiply(X80,feature_ranks['fval'].values)
    top90_article_score = np.multiply(X90,feature_ranks['fval'].values)
    messy = pd.DataFrame([user_article_score[0,predictive_indexes], top80_article_score[0,predictive_indexes],top90_article_score[0,predictive_indexes]],
                     index=['Your article', 'Top 20% articles', 'Top 10% articles']).T.reset_index()
    # Transform the combined data into tidy format
    tidy = pd.melt(
        messy,
        id_vars='index',
        value_vars=['Your article', 'Top 20% articles', 'Top 10% articles'],
        var_name=' '
    )
    # Draw a grouped bar plot of the weighted scores
    fontsize = 14
    plt.figure(figsize=(14,6))
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('legend', fontsize=fontsize)
    sns.factorplot(
        data=tidy,
        y='index',
        x='value',
        hue=' ',
        kind='bar',
        size=4,
        aspect=2,
        palette='Set1',
        orient='h',
        legend_out=False
    ).set(
        xlabel='score',
        ylabel='',
        xticks=[]
    )
    #%%
    plt.yticks(range(len(predictive_features)),predictive_features)
    plt.savefig('flaskexample/static/images/features.png', bbox_inches='tight', dpi=300);
    #Variables in result.html
    #article_title, color_choice, probability, message, topic, image_location-->
    return render_template(
        'result.html',
        article_title=article_title,
        color_choice=color_choice,
        probability=prob,
        message = message+" ",
        image_location=image_link,
        css_hash=css_link,
    )

#@app.route('/index', methods=['GET', 'POST'])
