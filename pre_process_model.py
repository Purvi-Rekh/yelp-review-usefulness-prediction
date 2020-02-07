#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:47:51 2020

@author: joshi.purvi
"""

# -*- coding: utf-8 -*-


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,f1_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score,classification_report
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,f1_score
import pandas as pd
import matplotlib.pyplot as plt
import gensim
import nltk

stopwords_ = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer() 
tf = CountVectorizer()

def read_data():
    ''' read data from file, return dataframe '''
    review = pd.read_csv("data/new_dataset_indian_resto_reiew.csv")
    return review

def view_dataFrame(review):
    ''' view head of data frame '''
    print(review.head(10))

def split_training_testing_data(review):
    
    ''' split dataset into training(to be used for training-validation test) and testing(unseen data), 
    return both dataframe '''
    train_set = review.sample(frac=0.75, random_state=0)
    test_set = review.drop(train_set.index)    
    return train_set,test_set

def find_useful(df): 
    ''' deciding useful not useful based on useful,cool and funny - 0 only if all of them are 0 '''
    if df[4] == 0 and df[7]==0 and df[8] ==0:
        return 0
    elif df[4]>=1:
        return df[4]
    else:
        return 1
    
def create_label(review):
    ''' assign label 0 or 1 based on useful,cool and funny '''
    review["useful2"] = review.apply(find_useful,axis=1)
    return review

def useful_not_useful(x):
    
    ''' thresold 1 for useful and not useful distribution '''
    
    if x <1:
        return 0
    elif x >=1:
        return 1


def label_count(review):
    ''' prints count of both types of labels '''
    
    print(review["label"].value_counts())

def remove_accents(input_str):
    ''' normlize text'''
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()

def find_length(x):
    '''find length of each review '''
    return len(x)

def pre_processing(india_review):
    
    ''' text pre-processing steps, normalize,tokenize,remove stop words '''
    
    india_review["normalizes_text"] = india_review["text"].apply(lambda x:remove_accents(x))
    #india_review["remove_html"] = india_review['normalizes_text'].apply(lambda x: strip_html_tags(x))
    india_review["remove_special"] = india_review["normalizes_text"].replace(r'[^A-Za-z0-9 ]+', '', regex=True)
    
    india_review["token_text"] = india_review["remove_special"].apply(lambda x:[word.lower() for word in x.split(" ")]) 
    #india_review["token_text"] = india_review["token_text"].apply(lambda x:[word for word in x if word not in common_words_list])
    
    india_review["remove_stop"] = india_review["token_text"].apply(lambda x:[word for word in x if word not  in stopwords_])
    india_review["lemitize_text"] = india_review["remove_stop"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x]))
    india_review["length"] = india_review["remove_stop"].apply(lambda x:find_length(x))
   # review["clean_list"] = review["lemitize_text"].apply(lambda x:[word for word in x.split(" ")])
    return india_review
     
def create_vector(new_review):
    
    ''' create vector using tf-idf  for train and test (unseen) data '''
    
    X = new_review["lemitize_text"]
    X.head()
    y=new_review["label"]
    tfidf = TfidfVectorizer(max_features=3000)
    X_mat=tfidf.fit_transform(X)
    
    final_X=new_test_review["lemitize_text"]
    final_y=new_test_review["label"]
    tfidf_test = TfidfVectorizer(max_features=3000)
    final_X_mat=tfidf_test.fit_transform(final_X)
    
    return X_mat,y,final_X_mat,final_y

def calculate_threshold_values(prob, y):
    '''
    Build dataframe of the various confusion-matrix ratios by threshold
    from a list of predicted probabilities and actual y values
    '''
    df = pd.DataFrame({'prob': prob, 'y': y})
    df.sort_values('prob', inplace=True)
    
    actual_p = df.y.sum()
    actual_n = df.shape[0] - df.y.sum()

    df['tn'] = (df.y == 0).cumsum()
    df['fn'] = df.y.cumsum()
    df['fp'] = actual_n - df.tn
    df['tp'] = actual_p - df.fn

    df['fpr'] = df.fp/(df.fp + df.tn)
    df['tpr'] = df.tp/(df.tp + df.fn)
    df['precision'] = df.tp/(df.tp + df.fp)
    df = df.reset_index(drop=True)
    return df

def plot_roc(ax, df, name):
    
    ''' plot roc curve'''
    ax.plot([1]+list(df.fpr), [1]+list(df.tpr), label=name)
#     ax.plot([0,1],[0,1], 'k', label="random")
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.set_title('ROC Curve - Model Comparison', fontweight='bold', fontsize=24)
    ax.legend(fontsize=14)
    plt.savefig("roc_useful_cool_funny_feature1000.jpg")

def predict_model(X,y,model):
    
    ''' predict model and print accuracy '''
    
    
    print("model is ",model)
    if model.__class__.__name__ == "LinearRegression":
        X_train,X_test,y_train,y_test = train_test_split(X_mat,y,test_size=0.20)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        print(mean_squared_error(y_test, y_pred))
        print(model.score(X, y))
    elif model.__class__.__name__ == "GaussianNB":
        X_train,X_test,y_train,y_test = train_test_split(X_mat,y,test_size=0.20)
        model.fit(X_train.toarray(),y_train)
        y_pred=model.predict(X_test.toarray())
    else:    
        X_train,X_test,y_train,y_test = train_test_split(X_mat,y,test_size=0.20)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        final_y_pred=model.predict(final_X_mat)
        final_precision = precision_score(final_y, final_y_pred)
        final_recall = recall_score(final_y, final_y_pred)
        final_accuracy = accuracy_score(final_y, final_y_pred)
        final_f1_accuracy = f1_score(final_y, final_y_pred)
        
        
        probs = model.predict_proba(X_test)[:,1]
        roc_auc = roc_auc_score(y_test, probs)
        thresh_df = calculate_threshold_values(probs, y_test)
        
        print("accuracy score = ",accuracy_score(y_test,y_pred))
        print("accuracy score = ",confusion_matrix(y_test,y_pred))
        print("precision = ",precision)
        print("recall = ",recall)
        print("f1 score = ",f1_score(y_test,y_pred))
        print(classification_report(y_test,y_pred))


        print("********Finall tets accuracy*****")

        print("accuracy score = ",final_accuracy)
        print("accuracy score = ",confusion_matrix(final_y, final_y_pred))
        print(classification_report(final_y,final_y_pred))

        print("precision = ",final_precision)
        print("recall = ",final_recall)
        print("f1 score = ",final_f1_accuracy)

    return (precision, recall, accuracy, thresh_df, roc_auc, model)
    
    
def carete_model_instance():
    
    ''' carete instances of supervised algorithms '''
    
    rf = RandomForestClassifier(n_estimators=50,max_depth=100,max_features=1000)
    MNB = MultinomialNB()
    BNB = BernoulliNB()
    lg = LogisticRegression()
    dt = DecisionTreeClassifier()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = [dt,lg,MNB,BNB,rf]
    res_list=[]
    for model in models:
        results=predict_model(X_mat,y,model)
        res_list.append(results)
        auc_score = results[4]
        plot_roc(ax, results[3], 
                 "{} AUC = {}".format(model.__class__.__name__, round(auc_score, 3)))
    plt.savefig('roc_curve_tfidf.png', dpi=256, bbox_inches='tight')

def main():
    
    ''' main function to call and execute all funtionality '''
    
    review = read_data()
    view_dataFrame(review)
    review,test_set = split_training_testing_data(review)
    review["label"]=review["useful2"].apply(lambda x:useful_not_useful(x))
    test_set["label"]=test_set["useful2"].apply(lambda x:useful_not_useful(x))
    new_review=pre_processing(review)
    X_mat,y,final_X_mat,final_y = create_vector(new_review)

if __name__ == "__main__":
    
    ''' starting point of module'''
    
    main()
