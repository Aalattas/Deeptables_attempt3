import pandas as pd
import numpy as np
from deeptables.models.deeptable import DeepTable, ModelConfig
from deeptables.models.deepnets import DeepFM
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


filename = 'Bot-IoT.csv'
df_BotIoT = pd.read_csv(filename)
print(df_BotIoT.shape)


features = df_BotIoT.drop(['label', 'tipo_ataque', 'ip_src', 'ip_dst', 'port_src', 'port_dst', 'protocols'], axis=1)
labels = df_BotIoT['tipo_ataque'].values


encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Model Configuration
conf = ModelConfig(
    nets=DeepFM,
    categorical_columns='auto',
    metrics=['AUC', 'accuracy'],
    auto_categorize=True,
    auto_discrete=False,
    embeddings_output_dim=20,
    embedding_dropout=0.3,
    earlystopping_patience=9999
)

dt = DeepTable(config=conf)
dt.fit(X_train, y_train, epochs=10)

preds_proba = dt.predict_proba(X_test)
preds_labels = np.argmax(preds_proba, axis=1)

# Calculate metrics
precision = precision_score(y_test, preds_labels, average='macro')
recall = recall_score(y_test, preds_labels, average='macro')
f1 = f1_score(y_test, preds_labels, average='macro')
accuracy = accuracy_score(y_test, preds_labels)
conf_matrix = confusion_matrix(y_test, preds_labels)

# Print results
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy Score:", accuracy)
