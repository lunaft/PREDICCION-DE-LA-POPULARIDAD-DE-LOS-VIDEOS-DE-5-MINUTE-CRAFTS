# -*- coding: utf-8 -*-
"""train.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1y-di3avVlXtTcud2UnC0hhc7fybCUnH1
"""

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Cargar datos
five_minute_crafts = pd.read_csv('videos_classified_and_popular_words.csv')

# Dividir datos
X = five_minute_crafts.drop(['video_id','title','popularity_encoded','num_words','num_punctuation','num_words_uppercase',
                             'num_words_lowercase','num_stopwords','contain_digits','startswith_digits','hacks','life',
                             'ideas','DIY','crafts'], axis=1)
y = five_minute_crafts['popularity_encoded']

# Entrenar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_gb_model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=200, random_state=42)
best_gb_model.fit(X_train_scaled, y_train)

# Guardar el modelo
joblib.dump(best_gb_model, 'my_model.pkl')


# loaded_model = joblib.load('my_model.pkl')

# y_pred = loaded_model.predict(X_test_scaled)

# Calcular la precisión del modelo
# accuracy_gb = accuracy_score(y_test, y_pred)
# print("Gradient Boosting Accuracy:", accuracy_gb)