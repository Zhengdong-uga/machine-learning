## Importing the dataset


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
```

## Data collection & Pre-processing


```python
#Importing the csv data
#The data is from Kaggle; The link:
#https://www.kaggle.com/datasets/venky73/spam-mails-dataset
raw_data = pd.read_csv("/Users/a1561035045/Desktop/DATA_full/spam_ham_dataset.csv")
```


```python
raw_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>label</th>
      <th>text</th>
      <th>label_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>605</td>
      <td>ham</td>
      <td>Subject: enron methanol ; meter # : 988291\r\n...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2349</td>
      <td>ham</td>
      <td>Subject: hpl nom for january 9 , 2001\r\n( see...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3624</td>
      <td>ham</td>
      <td>Subject: neon retreat\r\nho ho ho , we ' re ar...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4685</td>
      <td>spam</td>
      <td>Subject: photoshop , windows , office . cheap ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2030</td>
      <td>ham</td>
      <td>Subject: re : indian springs\r\nthis deal is t...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
raw_data.isnull().value_counts()
```




    Unnamed: 0  label  text   label_num
    False       False  False  False        5171
    dtype: int64




```python
#replacing the null values to string(necessary or not)
#where function to gater conditions
data = raw_data.where((pd.notnull(raw_data)),'')
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>label</th>
      <th>text</th>
      <th>label_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>605</td>
      <td>ham</td>
      <td>Subject: enron methanol ; meter # : 988291\r\n...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2349</td>
      <td>ham</td>
      <td>Subject: hpl nom for january 9 , 2001\r\n( see...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3624</td>
      <td>ham</td>
      <td>Subject: neon retreat\r\nho ho ho , we ' re ar...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4685</td>
      <td>spam</td>
      <td>Subject: photoshop , windows , office . cheap ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2030</td>
      <td>ham</td>
      <td>Subject: re : indian springs\r\nthis deal is t...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5166</th>
      <td>1518</td>
      <td>ham</td>
      <td>Subject: put the 10 on the ft\r\nthe transport...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5167</th>
      <td>404</td>
      <td>ham</td>
      <td>Subject: 3 / 4 / 2000 and following noms\r\nhp...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5168</th>
      <td>2933</td>
      <td>ham</td>
      <td>Subject: calpine daily gas nomination\r\n&gt;\r\n...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5169</th>
      <td>1409</td>
      <td>ham</td>
      <td>Subject: industrial worksheets for august 2000...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5170</th>
      <td>4807</td>
      <td>spam</td>
      <td>Subject: important online banking alert\r\ndea...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5171 rows × 4 columns</p>
</div>




```python
#remember spam mail=1; and ham mail=0;
```


```python
#separate the dataframe to X=text and Y=label_num
X = data['text']
Y = data['label_num']
```


```python
print(X)
```

    0       Subject: enron methanol ; meter # : 988291\r\n...
    1       Subject: hpl nom for january 9 , 2001\r\n( see...
    2       Subject: neon retreat\r\nho ho ho , we ' re ar...
    3       Subject: photoshop , windows , office . cheap ...
    4       Subject: re : indian springs\r\nthis deal is t...
                                  ...                        
    5166    Subject: put the 10 on the ft\r\nthe transport...
    5167    Subject: 3 / 4 / 2000 and following noms\r\nhp...
    5168    Subject: calpine daily gas nomination\r\n>\r\n...
    5169    Subject: industrial worksheets for august 2000...
    5170    Subject: important online banking alert\r\ndea...
    Name: text, Length: 5171, dtype: object



```python
print(Y)
#the Y's dtype is already the int64, we don't need to transform it
#however, if the dtype is not int64, here is the way to transform the Y_train and Y_test
# Y_train = Y_train.astype('int')
# Y_test = Y_test.astype('int')
```

    0       0
    1       0
    2       0
    3       1
    4       0
           ..
    5166    0
    5167    0
    5168    0
    5169    0
    5170    1
    Name: label_num, Length: 5171, dtype: int64


## Spliting data in to training and testing


```python
#train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```


```python
print(X.shape)
print(X_train.shape)
print(X_test.shape)
```

    (5171,)
    (4136,)
    (1035,)


**Feature extraction**


```python
#transform the text data to feature vectors that can be used in the Logistic regression
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = 'True')
```


```python
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
```


```python
print(Y_train)
```

    5132    0
    2067    1
    4716    0
    4710    0
    2268    1
           ..
    4426    0
    466     0
    3092    1
    3772    0
    860     0
    Name: label_num, Length: 4136, dtype: int64



```python
print(X_train)
print(X_train_features)
```

## Training the model


```python
model = LogisticRegression()
```


```python
model.fit(X_train_features, Y_train)
```




    LogisticRegression()



**Evaluating the model**


```python
#prediction on the trained model
prediction_training = model.predict(X_train_features)
accuracy_training = accuracy_score(Y_train, prediction_training)
```


```python
print('accuracy on training data:', accuracy_training)
#accuracy > 80 should be a good model
```

    accuracy on training data: 0.9961315280464217


**Evaluating the test data**


```python
prediction_testing = model.predict(X_test_features)
accuracy_testing = accuracy_score(Y_test, prediction_testing)
```


```python
print('accuracy on test data:', accuracy_testing)
```

    accuracy on test data: 0.9903381642512077


## Building a predictive system


```python
#input mail content
#Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
input_mail1 = ["FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, ¬£1.50 to rcv"]
```


```python
input_mail2 = ["AI PROMPTER: AI tools like ChatGPT are only as useful as the humans asking them questions, so some companies have begun to hire “prompt engineers” for extracting the right info from chatbots. Finally, a job AI can never replace."]
```


```python
#convert this text to numeric
input_mail1_features = feature_extraction.transform(input_mail1)
input_mail2_features = feature_extraction.transform(input_mail2)
```


```python
print(input_mail1_features)
print()
print(input_mail2_features)
```

      (0, 44290)	0.45693818045632917
      (0, 43767)	0.3071832850784587
      (0, 43228)	0.21719445758584774
      (0, 36775)	0.21719445758584774
      (0, 30258)	0.2992343199746608
      (0, 26021)	0.19082412020076533
      (0, 21500)	0.28751953875747943
      (0, 19295)	0.3139603458554902
      (0, 13766)	0.4968208388027072
      (0, 2293)	0.22016927780894374
    
      (0, 41942)	0.20551139722373377
      (0, 40632)	0.1708111548713923
      (0, 35386)	0.12311025897040584
      (0, 34885)	0.19721998266799587
      (0, 33874)	0.0879908544042539
      (0, 33178)	0.18892856811225794
      (0, 26021)	0.09558515803886385
      (0, 24304)	0.16636830871277097
      (0, 23103)	0.12481169096747904
      (0, 21632)	0.22245226137805102
      (0, 18428)	0.1611135370235147
      (0, 17802)	0.24886108917465458
      (0, 11976)	0.13979525568375661
      (0, 7762)	0.19491303574624677
      (0, 6542)	0.1625197403156544
      (0, 5111)	0.7465832675239638



```python
#prediction
prediction1 = model.predict(input_mail1_features)
prediction2 = model.predict(input_mail2_features)
```


```python
#print(prediction1)
if prediction1[0] == 1:
    print('Spam mail')
else:
    print('Ham mail')
```

    Spam mail



```python
if prediction2[0] == 1:
    print('Spam mail')
else:
    print('Ham mail')
```

    Spam mail

