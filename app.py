import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
###### helper functions. Use them when needed #######

def get_title_from_index(index,df):
    #df = pd.read_csv("movie_dataset.csv")
    #df['title']=df['title'].str.lower()
    #df1=df[df.index == index]["title"]
    #return df1.values[0]
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title,df):
    #df = pd.read_csv("movie_dataset.csv")
    #df['title']=df['title'].str.lower()
    #df1=df[df.title == title]["index"]
    #return df1.values[0]
	return df[df.title == title]["index"].values[0]

def create_sim():
##Step 1: Read CSV File
    df = pd.read_csv('movie_dataset.csv')
#print df.columns
##Step 2: Select Features
    df['title']=df['title'].str.lower()
    features = ['keywords','cast','genres','director']
##Step 3: Create a column in DF which combines all selected features
    for feature in features:
        df[feature] = df[feature].fillna('')
    df["combined_features"] = df.apply(combine_features,axis=1)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(count_matrix) 
    return cosine_sim,df
#movie_user_likes = "Avatar"
def combine_features(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
		print ("Error:", row)	
def rcmd(m):
    m = m.lower()
    cosine_sim,df=create_sim()
    movie_index = get_index_from_title(m,df)
    similar_movies =  list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)
    sorted_similar_movies=sorted_similar_movies[1:11]
    l=[]
    for element in sorted_similar_movies:
        l.append(get_title_from_index(element[0],df))
        
    return l    
#for rating
def rating():
    df = pd.read_csv('movie_dataset.csv')
    df[['title']].head()
    df1=df.sort_values('vote_average',ascending=False)
    df2 = df1[df1['vote_count'] > 200]
    df2[['title','vote_count','vote_average']]
    first=df2[1:11]
    l=[]
    l.extend(first['title'].tolist())
    return l

#for popularity
def popularity():
    df = pd.read_csv('movie_dataset.csv')
    df3=df.sort_values('popularity',ascending=False)
    second=df3[:11]
    l1=[]
    l1.extend(second['title'].tolist())
    return l1

#for weighted average
def weighted_average():
     df = pd.read_csv('movie_dataset.csv')
     df4=df.copy()
     df4.columns
     v=df['vote_count']
     R=df['vote_average']
     C=df['vote_average'].mean()
     m=df['vote_count'].quantile(0.70)
     df4['weighted_average']=((R*v)+ (C*m))/(v+m)
     df5=df4.sort_values('weighted_average',ascending=False)
     second=df5[:11]
     l2=[]
     l2.extend(second['title'].tolist())
     return l2

#for both weighted and popularity

def both():
    df = pd.read_csv('movie_dataset.csv')
    df4=df.copy()
    df4.columns
    v=df['vote_count']
    R=df['vote_average']
    C=df['vote_average'].mean()
    m=df['vote_count'].quantile(0.70)
    df4['weighted_average']=((R*v)+ (C*m))/(v+m)
    df5=df4.sort_values('weighted_average',ascending=False)
    scaling=MinMaxScaler()
    df6=df5.copy()
    df7=scaling.fit_transform(df6[['weighted_average','popularity']])
    df8=pd.DataFrame(df7,columns=['weighted_average','popularity'])
    df6[['normalized_weight_average','normalized_popularity']]= df8
    df6.columns
    df6['score'] = df6['normalized_weight_average'] * 0.5 + df6['normalized_popularity'] * 0.5
    df9 = df6.sort_values(['score'], ascending=False)
    df9.columns
    df9[['score']].head(10)
    third=df9[:11]
    l3=[]
    l3.extend(third['title'].tolist())
    return l3


app = Flask(__name__)
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')
@app.route("/recommend2")
def recommend2():
    r=rating()
    return render_template('recommend2.html',r=r)

@app.route("/recommend3")
def recommend3():
    r=popularity()
    return render_template('recommend3.html', r=r)

@app.route("/recommend4")
def recommend4():
    r=weighted_average()
    return render_template('recommend4.html', r=r)

@app.route("/recommend5")
def recommend5():
    r=both()
    return render_template('recommend5.html', r=r)


if __name__ == '__main__':
    app.run()
