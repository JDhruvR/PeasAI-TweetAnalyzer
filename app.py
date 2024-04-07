import customtkinter

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
import transformers

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import pipeline

from scipy.special import softmax

class EntryFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.currentString=''

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        self.TweetEntry = customtkinter.CTkEntry(self, placeholder_text="Enter Tweet for analysis")
        self.TweetEntry.grid(row=0, column=0, padx=(10, 10), pady=(10, 10), sticky="nwe")

        self.EntrySubmitButton = customtkinter.CTkButton(self, text="Submit", command=self.submitButtonAction)
        self.EntrySubmitButton.grid(row=0, column=1, padx=(0, 10), pady=(10, 10), sticky="e")

    def submitButtonAction(self):
        self.currentString = self.TweetEntry.get()

        tweetClass=classAnalysis(self.currentString)
        
        if(sentimentAnalysis(self.currentString)==2):
            app.positiveFrame.frameList.append(TweetFrame(app.positiveFrame, text=self.currentString, tweetClass=tweetClass, sectionName='Positive', sectionRow=app.positiveFrame.row))
            app.positiveFrame.showFrame(-1)

        elif(sentimentAnalysis(self.currentString)==1):
            app.neutralFrame.frameList.append(TweetFrame(app.neutralFrame, text=self.currentString, tweetClass=tweetClass, sectionName='Neutral', sectionRow=app.neutralFrame.row))
            app.neutralFrame.showFrame(-1)

        else:
            app.negativeFrame.frameList.append(TweetFrame(app.negativeFrame, text=self.currentString, tweetClass=tweetClass, sectionName='Negative', sectionRow=app.negativeFrame.row))
            app.negativeFrame.showFrame(-1)

        self.TweetEntry.delete(0, len(self.currentString))

class PositiveFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master)

        self.frameList = []
        self.row = 1

        self.configure(height=300)

        self.grid_columnconfigure(0, weight=2)
        
        self.label = customtkinter.CTkLabel(self, text="Positive Tweets")
        self.label.grid(row=0, column=0, padx=20, sticky="w")
        self.label = customtkinter.CTkLabel(self, text="Class                             Sentiment")
        self.label.grid(row=0, column=0, padx=35, sticky="e")

    def showFrame(self, row):
        self.grid_columnconfigure(0, weight=1)
        self.frameList[row].grid(row=self.row, padx=(10, 10), pady=(0, 5), sticky="nwes")
        
        self.row+=1

class NeutralFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master)

        self.frameList = []
        self.row = 1

        self.configure(height=300)

        self.label = customtkinter.CTkLabel(self, text="Neutral Tweets")
        self.label.grid(row=0, column=0, padx=20, sticky="w")

    def showFrame(self, row):
        self.grid_columnconfigure(0, weight=1)
        self.frameList[row].grid(row=self.row, padx=(10, 10), pady=(0, 5), sticky="nwes")
        self.row+=1

class NegativeFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master)

        self.frameList = []
        self.row = 1

        self.configure(height=300)

        self.label = customtkinter.CTkLabel(self, text="Negative Tweets")
        self.label.grid(row=0, column=0, padx=20, sticky="w")
    
    def showFrame(self, row):
        self.grid_columnconfigure(0, weight=1)
        self.frameList[row].grid(row=self.row, padx=(10, 10), pady=(0, 5), sticky="nwes")
        self.row+=1

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("PeasAI - TweetAnalyzer")
        self.geometry("1600x1000")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)

        self.checkboxFrame = EntryFrame(self)
        self.checkboxFrame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nwe")

        self.positiveFrame = PositiveFrame(self)
        self.positiveFrame.grid(row=1, column=0, padx=10, pady=(10,10), stick="nwe")

        self.neutralFrame = NeutralFrame(self)
        self.neutralFrame.grid(row=2, column=0, padx=10, pady=(0,10), stick="we", )

        self.negativeFrame = NegativeFrame(self)
        self.negativeFrame.grid(row=3, column=0, padx=10, pady=(0,10), stick="swe")

class TweetFrame(customtkinter.CTkFrame):
    def __init__(self, master, text, tweetClass, sectionName, sectionRow):
        super().__init__(master)

        self.sectionName=sectionName
        self.sectionRow=sectionRow
        self.text=text

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self._border_width=2

        self.classes = ['Cancelled Flight', 'Customer Service', 'Bad Flight',
                        'Late Flight', 'Customer Service Issue', 'No Issues',
                        'Refund Issue', 'Flight Delays', 'Flight Attendant Complaints',
                        'Lost Luggage', 'No issue', 'Bad Experience', 'App Issue',
                        'Flight Booking Problems', 'Food Issues', 'Long Lines',
                        'Damaged Luggage']
        self.tweetClass=tweetClass
        self.sentiments=['Positive', 'Neutral', 'Negative']

        self.tweet = customtkinter.CTkLabel(self, text=text)
        self.tweet.grid(row=0, column=0, padx=(10, 10), pady=(5, 5), stick="w")

        self.classDropDown = customtkinter.CTkOptionMenu(self, values=self.classes, command=self.changeClass, hover=True)
        self.classDropDown.set(value=tweetClass)
        self.classDropDown.grid(row=0, column=1, padx=(10, 10), pady=(5,5), stick="e")

        self.sentimentsDropDown = customtkinter.CTkOptionMenu(self, values=self.sentiments, command=self.changeSentiment, width=90, hover=True)
        self.sentimentsDropDown.set(value=self.sectionName)
        self.sentimentsDropDown.grid(row=0, column=2, padx=(10, 10), pady=(5,5), stick="e")

    def changeClass(self, choice):
        self.tweetClass=choice

    def changeSentiment(self, choice):
        if(choice==self.sectionName):
            pass
        elif(self.sectionName=='Positive'):
            if(choice=='Neutral'):
                app.neutralFrame.frameList.append(TweetFrame(app.neutralFrame, text=self.text, sectionName='Neutral', tweetClass=self.tweetClass, sectionRow=app.neutralFrame.row ))
                app.neutralFrame.showFrame(-1)

                self.destroy()
            else:
                app.negativeFrame.frameList.append(TweetFrame(app.negativeFrame, text=self.text, sectionName='Negative', tweetClass=self.tweetClass, sectionRow=app.negativeFrame.row ))
                app.negativeFrame.showFrame(-1)

                self.destroy()
        elif(self.sectionName=='Neutral'):
            if(choice=='Positive'):
                app.positiveFrame.frameList.append(TweetFrame(app.positiveFrame, text=self.text, sectionName='Positive', tweetClass=self.tweetClass, sectionRow=app.positiveFrame.row ))
                app.positiveFrame.showFrame(-1)

                self.destroy()
            else:
                app.negativeFrame.frameList.append(TweetFrame(app.negativeFrame, text=self.text, sectionName='Negative', tweetClass=self.tweetClass, sectionRow=app.negativeFrame.row ))
                app.negativeFrame.showFrame(-1)

                self.destroy()
        else:
            if(choice=='Positive'):
                app.positiveFrame.frameList.append(TweetFrame(app.positiveFrame, text=self.text, sectionName='Positive', tweetClass=self.tweetClass, sectionRow=app.positiveFrame.row ))
                app.positiveFrame.showFrame(-1)

                self.destroy()
            else:
                app.neutralFrame.frameList.append(TweetFrame(app.neutralFrame, text=self.text, sectionName='Neutral', tweetClass=self.tweetClass, sectionRow=app.neutralFrame.row ))
                app.neutralFrame.showFrame(-1)

                self.destroy()

def preprocess(text): #Removes links and @username and rids all tweets of biases in username and links.
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

labels = {0:'Negative', 1:'Neutral', 2:'Positive'} #Label dictionary to convert numerical predictions to text

modelName = "cardiffnlp/twitter-roberta-base-sentiment-latest" #The Huggingface transformer used for Sentimental Analysis - twitter-roberta-base-sentiment-latest

tokz = AutoTokenizer.from_pretrained(modelName)

modelForSentiment = AutoModelForSequenceClassification.from_pretrained("D:\Downloads\FineTunedV2-20240402T132841Z-001\FineTunedV2") #Loading a finetuned version of the Huggingface model used.

classifier = pipeline("zero-shot-classification",
                      model="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33")

candidateClassLabels = ['Cancelled Flight', 'Customer Service', 'Bad Flight',
       'Late Flight', 'Customer Service Issue', 'No Issues',
       'Refund Issue', 'Flight Delays', 'Flight Attendant Complaints',
       'Lost Luggage', 'No issue', 'Bad Experience', 'App Issue',
       'Flight Booking Problems', 'Food Issues', 'Long Lines',
       'Damaged Luggage']

def sentimentAnalysis(tweet):
    output = modelForSentiment(**tokz(preprocess(tweet), return_tensors='pt')) #Predicts using the model

    scores = output[0][0].detach().numpy()
    result = np.argmax(softmax(scores))
    
    return result

def classAnalysis(tweet):
    return classifier(preprocess(tweet), candidateClassLabels)['labels'][0]

app = App()
app.mainloop()
