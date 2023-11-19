import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Dados do arquivo CSV
df_train = pd.read_csv('train.csv')
w = df_train.shape
df_test = pd.read_csv('test.csv')
z = df_test.shape
print(w)
print(z)

# Normalizar os dados
def normalizar(database):
    features = database.drop(['label'], axis=1)
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(features)
    df_norm_data = pd.DataFrame(norm_data)
    df_norm_data['label'] = database['label']
    return df_norm_data

df_train_norm = normalizar(df_train)
df_test_norm = normalizar(df_test)


# Treinamento
x_train = df_train_norm.drop(['label'], axis=1)
y_train = df_train_norm['label']

x_test = df_test_norm.drop(['label'], axis=1)
y_test = df_test_norm['label']

classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Teste
predictions = classifier.predict(x_test)

print('\nLabels: ', end='')
classes = classifier.classes_
print(classes)

print('Quantidade por classe: ', end='')
classes_count = classifier.class_count_
print(classes_count)


# Métricas
print('\nMetricas: ')
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
confusion_matrix = metrics.confusion_matrix(y_test, predictions)

plt.title('Matriz de Confusão')
df_confusionMatrix = pd.DataFrame(confusion_matrix, range(10), range(10))
ax = sns.heatmap(df_confusionMatrix, annot=True, fmt='g')
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.xlabel('Classes Preditas')
plt.ylabel('Classes Verdadeiras')

print(metrics.classification_report(y_test, predictions))
 

# Taxa de Erro por Classe e Geral
class_error_rates = 1 - np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
overall_error_rate = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

print('\nTaxa de Erro:\n-------------------------------')
for i, error_rate in enumerate(class_error_rates):
    print(f'classe {i}: {error_rate}')

print(f'\ngeral: {overall_error_rate}')


# Visualização das médias por classe em um formato de imagem 28x28
class_media = classifier.theta_

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(class_media[i].reshape(28, 28), cmap='binary')
    ax.set_title(f'Class {i}')
    ax.axis('off')

plt.show()