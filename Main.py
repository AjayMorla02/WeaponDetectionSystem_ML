from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt

# Importing necessary machine learning and data processing libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
import pandas as pd
from imblearn.over_sampling import SMOTE
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Convolution2D
from keras.models import Sequential
import pickle

# Global variables
filename = ""
X, Y = None, None
classifier = None
dataset = None
X_train, X_test, y_train, y_test = None, None, None, None
accuracy, precision, recall, fscore = [], [], [], []
le = None

# Initializing GUI window
main = tkinter.Tk()
main.title("Suicidal Tendency Detection")  # Set window title
main.geometry("1300x1200")  # Set window size

# Function to upload dataset
def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")  # Open file dialog to select dataset
    text.insert(END, filename + " loaded\n\n")
    dataset = pd.read_csv(filename)  # Load dataset using pandas
    text.insert(END, "Dataset before applying machine translation\n\n")
    text.insert(END, str(dataset.head()))

# Function to preprocess dataset
def processDataset():
    global dataset
    text.delete('1.0', END)
    dataset.fillna(0, inplace=True)  # Replace missing values with 0
    text.insert(END, "All missing values are replaced with 0\n")
    text.insert(END, "Total processed records found in dataset : " + str(dataset.shape[0]) + "\n\n")
    dataset.groupby('attempt_suicide').size().plot(kind="bar")  # Plot suicide attempts distribution
    plt.show()

# Function to encode categorical features and prepare data for training
def translation():
    global X_train, X_test, y_train, y_test, X, Y, le, dataset
    text.delete('1.0', END)
    
    # Drop unnecessary columns
    dataset.drop(['time', 'income'], axis=1, inplace=True, errors='ignore')
    
    # Encode categorical values
    le = LabelEncoder()
    cols = ['gender', 'sexuallity', 'race', 'bodyweight', 'virgin', 'prostitution_legal', 'pay_for_sex',
            'social_fear', 'stressed', 'what_help_from_others', 'attempt_suicide', 'employment', 'job_title',
            'edu_level', 'improve_yourself_how']
    for col in cols:
        if col in dataset.columns:
            dataset[col] = le.fit_transform(dataset[col].astype(str))
    
    # Extract features and labels
    Y = dataset['attempt_suicide'].values
    dataset.drop(['attempt_suicide'], axis=1, inplace=True)
    X = dataset.values
    
    # Handle class imbalance using SMOTE
    sm = SMOTE(random_state=42)
    X, Y = sm.fit_resample(X, Y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    text.insert(END, "Total records used for training: " + str(X_train.shape[0]) + "\n")
    text.insert(END, "Total records used for testing: " + str(X_test.shape[0]) + "\n")

# Function to train CNN model
def trainCNN():
    global X_train, X_test, y_train, y_test, classifier, accuracy, precision, recall, fscore
    
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text.delete('1.0', END)
    
    # Reshape data for CNN
    XX = X.reshape(X.shape[0], X.shape[1], 1, 1)
    YY = to_categorical(Y)
    X_train1 = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
    X_test1 = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    
    # Define CNN architecture
    classifier = Sequential()
    classifier.add(Convolution2D(32, 1, 1, input_shape=(X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dense(units=y_train1.shape[1], activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    classifier.fit(XX, YY, batch_size=16, epochs=70, shuffle=True, verbose=2)

# Function to train Random Forest classifier
def RFTraining():
    global X_train, X_test, y_train, y_test
    rf = RandomForestClassifier(class_weight='balanced')
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    accuracy.append(accuracy_score(y_test, predict) * 100)

# Function to visualize results
def graph():
    df = pd.DataFrame([
        ['CNN', 'Accuracy', accuracy[0]],
        ['Random Forest', 'Accuracy', accuracy[1]]
    ], columns=['Algorithm', 'Metric', 'Value'])
    df.pivot(index='Metric', columns='Algorithm', values='Value').plot(kind='bar')
    plt.show()

# GUI components
title = Label(main, text='Suicidal Tendency Detection', bg='dark goldenrod', fg='white', font=('times', 16, 'bold'), height=3, width=120)
title.place(x=0, y=5)

text = Text(main, height=30, width=110, font=('times', 12, 'bold'))
text.place(x=10, y=100)

uploadButton = Button(main, text="Upload Dataset", command=uploadDataset, bg='#ffb3fe', font=('times', 13, 'bold'))
uploadButton.place(x=900, y=100)

processButton = Button(main, text="Preprocess Dataset", command=processDataset, bg='#ffb3fe', font=('times', 13, 'bold'))
processButton.place(x=900, y=150)

main.config(bg='RoyalBlue2')
main.mainloop()
