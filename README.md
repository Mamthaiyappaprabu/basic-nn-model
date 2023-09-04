# DEVELOPING A NEURAL NETWORK REGRESSION MODEL:

## AIM:

To develop a neural network regression model for the given dataset.

## THEORY:

A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Here the basic neural network model has been created with one input layer, one hidden layer and one output layer.The number of neurons(UNITS) in each layer varies the 1st input layer has 16 units and hidden layer has 8 units and output layer has one unit.

In this basic NN Model, we have used "relu" activation function in input and hidden layer, relu(RECTIFIED LINEAR UNIT) Activation function is a piece-wise linear function that will output the input directly if it is positive and zero if it is negative.

## NEURAL NETWORK MODEL :

![Screenshot 2023-09-04 201805](https://github.com/Mamthaiyappaprabu/basic-nn-model/assets/119393563/88f0b1e1-950c-477b-86f7-e19496990a7a)


## DESIGN STEPS:

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM:

NAME : MAMTHA I

REG NO : 212222230076
```
        
### To Read CSV file from Google Drive :

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

## To train and test :

from sklearn.model_selection import train_test_split

## To scale :

from sklearn.preprocessing import MinMaxScaler

## To create a neural network model:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### Authenticate User:

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

### Open the Google Sheet and convert into DataFrame :

worksheet = gc.open('My Dataset').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns = rows[0])


df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})

df

x=dataset1[['INPUT']].values
y=dataset1[['OUTPUT']].values

x
y

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.4, random_state =35)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
X_train1 = Scaler.transform(x_train)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
ai_brain = Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop' , loss = 'mse')
ai_brain.fit(X_train1 , y_train,epochs = 2005)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

X_test1 =Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1=[[4]]
X_n1_1=Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```

## DATA INFORMATION:

![OUT1](https://github.com/Mamthaiyappaprabu/basic-nn-model/assets/119393563/6e2af7b6-0da1-4901-8614-2a398f52c687)

## OUTPUT:

### TRAINING LOSS VS ITERATION PLOT :

![OUT2](https://github.com/Mamthaiyappaprabu/basic-nn-model/assets/119393563/1b1eead3-788a-4e66-9522-401e79e40067)


### TEST DATA ROOT MEAN SQUARED ERROR :
![OUT3](https://github.com/Mamthaiyappaprabu/basic-nn-model/assets/119393563/97158085-415f-421d-ad75-ac90504bfd3d)


### NEW SAMPLE DATA PREDICTION :

![OUT4](https://github.com/Mamthaiyappaprabu/basic-nn-model/assets/119393563/4a438c25-5ff5-463e-9a5b-a3a41e3cb342)


## RESULT:

Thus a neural network regression model for the given dataset is written and executed successfully
