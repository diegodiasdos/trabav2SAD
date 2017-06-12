
# coding: utf-8

# ### Trabalho de SAD  
# ### Aluno: Diego Dias Dos Santos Martins.  matrícula: 0050005813.  
# ### Professor: Alex Salgado.
# * Preencha sua resposta no próprio arquivo do Jupyter e depois me envie no link que vou colocar no basecamp.

# # Tarefa simples de Classificação
# 
# Referência: https://github.com/justmarkham/scikit-learn-videos/blob/master/03_getting_started_with_iris.ipynb
# 
# 1 - importar os modulos python para machine learn e carregar o arquivo fruit_data_with_colors2.xlsx usando o método read_excel do pandas

# In[30]:

get_ipython().magic(u'matplotlib notebook')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
fruits = pd.read_excel('fruit_data_with_colors2.xlsx')


# 2 - Exibir os primeiros registros desta tabela

# In[31]:

fruits.head()


# ## Machine learning terminology
# Each row is an observation (also known as: sample, example, instance, record)
# Each column is a feature (also known as: predictor, attribute, independent variable, input, regressor, covariate)
# 
# 3.1 - Quantas observações têm nessa base de dados?  
# 3.2 - Quantas "features" têm nessa base de dados?

# In[32]:

##############################################################################################
#3.1 - 59 Observações\n"
#3.2 - 7 features"


# Each value we are predicting is the response (also known as: target, outcome, label, dependent variable)
# Classification is supervised learning in which the response is categorical
# Regression is supervised learning in which the response is ordered and continuous
# 
# 4.1 - Quantas respostas temos nessa base?  
# 4.2 - Para fazer nossas previsões em cima desta base, devemos usar algoritmo de Classificação ou Regressão?  

# In[33]:


##############################################################################################

#4.1 - 59 respostas\n",
#4.2 - Algoritmo de Classificação para efetuar previsões exatas a respeito das frutas."


# # store feature matrix in "X"
# X = ???
# 
# # store response vector in "y"
# y = ???
# 
# 5 - Como você pode gerar a matriz X de feature e o vetor y de respostas? Dica: use essas 3 features para X ['mass', 'width', 'height'] e a resposta deve ser ['fruit_label'].

# In[34]:

X = fruits[['mass','width','height']]
y = fruits['fruit_label']


# ## Training a machine learning model with scikit-learn
# 
# 
# K-nearest neighbors (KNN) classification  
# 6.1 - Usando o algoritmo de KNN (com 1 vizinho, k=1), qual a previsão? Adivinhe qual é a fruta com massa 20g, largura 4.3 cm, altura 5.5 cm, ou seja, com as seguintes features (mass = 20,	width=4.3,	height=5.5).
# 
# 
# Referência: https://github.com/justmarkham/scikit-learn-videos/blob/master/04_model_training.ipynb
# 
# 

# In[41]:

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
knn.predict([[20, 4.3, 5.5]])


# 

# In[44]:

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
knn.predict([[20, 4.3, 5.5]])


# e 6.3 - LogisticRegression 
# 
# 

# ## Evaluation procedure #1: Train and test on the entire dataset
# 7 - Usando o método de avaliação de acurácia (Treinar e testar na base de dados inteira), qual dos 3 métodos do item 6 é mais eficiente?
# 
# Referência: https://github.com/justmarkham/scikit-learn-videos/blob/master/05_model_evaluation.ipynb
# 

# 7.1 - Acurácia usando o algoritmo de KNN (com 1 vizinho, k=1)

# In[45]:

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))


# 7.2 - Acurácia usando o algoritmo de KNN (com 5 vizinho, k=5)

# In[46]:

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))


# 7.3 - Acurácia usando o algoritmo de LogisticRegression
# 

# ## Evaluation procedure #2: Train/test split
# Usando o método de avaliação de acurácia (Treinar e testar SEPARADAMENTE), qual dos 3 métodos do item 6 é mais eficiente?
# Referência: https://github.com/justmarkham/scikit-learn-videos/blob/master/05_model_evaluation.ipynb

# 8.1 - Acurácia usando o algoritmo de KNN (com 1 vizinho, k=1)

# In[51]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# 8.2 - Acurácia usando o algoritmo de KNN (com 5 vizinho, k=5)

# In[52]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# 8.3 - Acurácia usando o algoritmo de LogisticRegression

# In[55]:

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# ## Can we locate an even better value for K?
# Faça um programa python para testar diferentes valores de K entre 1 e 25 e diga através de gráfico, qual foi o melhor valor de K que obteve a melhor performance?
# 

# In[58]:

k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt


get_ipython().magic(u'matplotlib inline')

plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# Utilizando este seu último ajuste de K, faça novamente a seguinte previsão. Adivinhe qual é a fruta com massa 20g, largura 4.3 cm, altura 5.5 cm, ou seja, com as seguintes features (mass = 20, width=4.3, height=5.5

# In[60]:

knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(X, y)

knn.predict([[20,4.3,5.5]])


# In[ ]:



