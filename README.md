# Tweet age predection using Machine Learning

- This project is for prediction user's age based on his tweets.
- Dataset available [here](https://drive.google.com/drive/folders/11_xqitTHNq0q4_shbMJkBsetF1bylTj9?usp=sharing)

## **Data preprocessing**:

After this section, I obtained two columns:

- Tweet as text: X
- Age : y

## **Data Description**:

```
                                            tweet  class
Things I want for my business cards but are to...  25-34
@username "and we get a free lunch" hahahaha\t\t   18-24
@username your new discussion layout is confus...  25-34
```

```
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   tweet   14166 non-null  object
 1   class   14166 non-null  object
dtypes: object(2)
memory usage: 332.0+ KB
```

## **Models Implementation**:

```
Using : TfidfVectorizer() to deal with text and extract features
from sklearn.feature_extraction.text
```

### Models:

**Naive bayes**

- MNB_clf = MultinomialNB()

**LogisticRegression**

- logreg = LogisticRegression(random_state=42)

**Support Vector Machine**

- classifier = svm.SVC(kernel="linear")

## **Validation**:

- Using k-fold cross validation:

```
skfold = StratifiedKFold(n_splits=10, random_state=100)
results_model = cross_val_score(model, train_vectors, y_train, cv=skfold)
```
