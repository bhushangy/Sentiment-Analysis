from flask import Flask, redirect, render_template, request, url_for, session
import time
import re
import os



app = Flask(__name__)
app.secret_key = "secret"

@app.route("/", methods=['GET', 'POST'])
def index():
	
		list=[]
		try:
			list=session["list"]
			five=session["five"]
		except:
			session["list"]=list=[]
			session["five"]=five=5	
		if request.method == "GET":
			return render_template("index.html")

		if request.method == 'POST':
			inp = request.form['data']
			if inp == '':
				msg='Sorry..could u please repeat!!!!'
				return render_template("index.html",msg=msg,score="",advise="",five="Please submit 5 more sentences for score and advise")
        
			else:
				import pandas as pd
				import numpy as np
				import spacy
				nlp=spacy.load('en_core_web_sm')
				import nltk
				from sklearn.model_selection import train_test_split
				inp = request.form['data']
				

				df4=pd.read_csv('cleaned_data.csv')
			
				X = df4['cleaned_sentence']
				y = df4['emotion']

				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


# In[4]:


				from sklearn.pipeline import Pipeline
				from sklearn.feature_extraction.text import TfidfVectorizer
				from sklearn.svm import LinearSVC

				text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
				])

# Feed the training data through the pipeline
				text_clf.fit(X_train.values.astype('U'), y_train) 

				import nltk
				nltk.download('vader_lexicon')
				from nltk.sentiment.vader import SentimentIntensityAnalyzer

				sid = SentimentIntensityAnalyzer()
				scores=sid.polarity_scores(inp)
				list.append(scores['compound'])
				session["list"]=list
				corpus=[]
				sentence=re.sub('[^a-zA-Z]', ' ',inp)
				sentence=sentence.lower()
				sentence=sentence.split()
   
				sentence=[s for s in sentence if not nlp.vocab[s].is_stop]
				sentence=' '.join(sentence)
				sent=nlp(sentence)   
				sent2=[s.lemma_ for s in sent ]
				sentence2=' '.join(sent2)
				inp=sentence2   
				z=pd.Series(inp)
				predictions = text_clf.predict(z)
				five-=1
				session["five"]=five
				out=predictions[0]
				
				if len(session["list"])==5:
					avg=sum(list)/5
					session["list"]=[]
					if(avg>=0 and avg<=0.2):
						session.clear()
						return render_template("index.html",msg=out,score="Compound Score :"+str(avg),advise="Stay Happy !!") 
					elif(avg>0.2 and avg<=0.35):
						session.clear()
						return render_template("index.html",msg=out,score="Compound Score :"+str(avg),advise="Carry on the smile ")	
					elif (avg>0.35 and avg<=0.45):
						session.clear()
						return render_template("index.html",msg=out,score="Compound Score :"+str(avg),advise="So good to see that you are very happy !")	
					elif (avg>0.45):
						session.clear()
						return render_template("index.html",msg=out,score="Compound Score :"+str(avg),advise="Overwhelming !")
					elif(avg<0 and avg>=-0.2):
						session.clear()
						return render_template("index.html",msg=out,score="Compound Score :"+str(avg),advise="You are sad ")					
					elif(avg<-0.2 and avg>=-0.35):
						session.clear()
						return render_template("index.html",msg=out,score="Compound Score :"+str(avg),advise="You are down a little bit .Nothing to worry about much . Stay happy !!")

					elif(avg<-0.35 and avg>=-0.45):	
						session.clear()
						return render_template("index.html",msg=out,score="Compound Score :"+str(avg),advise="Take Care of yout health. Be calm and enjoy each and every bit of life.Visit a psychologist")
					elif(avg<-0.45):	
						session.clear()
						return render_template("index.html",msg=out,score="Compound Score :"+str(avg),advise="Immediately consult a psychologist !")
				elif(session["five"]==1):	
					return render_template("index.html",msg=out,score="",advise="",five="Please submit "+str(session["five"])+ " more sentence for score and advise")
				else:	
					return render_template("index.html",msg=out,score="",advise="",five="Please submit "+str(session["five"])+ " more sentences for score and advise")	

@app.route("/newlog",methods=["GET","POST"])
def newlog():
	session.clear()
	return redirect(url_for("index"))
@app.route("*",methods=["GET","POST"])
def newlog():
	return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=True,port = int(os.environ.get('PORT', 33507)))
