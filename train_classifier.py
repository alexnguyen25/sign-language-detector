import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# Load data
data_dict = pickle.load(open('data.pickle', 'rb'))

# FILTER: Keep only samples with exactly 42 coordinates
data = []
labels = []

for i in range(len(data_dict['data'])):
    if len(data_dict['data'][i]) == 42:  # Only keep correct size
        data.append(data_dict['data'][i])
        labels.append(data_dict['labels'][i])
    else:
        print(f"Skipping sample {i} with {len(data_dict['data'][i])} coordinates")

print(f"Kept {len(data)} out of {len(data_dict['data'])} samples")

# Now convert to numpy (all same size now!)
data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)  #Testing 20% and training 80% of data, x is the image and y is the letter key

model = RandomForestClassifier() #Model for analyzing landmarks
model.fit(x_train, y_train) #Training the model
y_predict = model.predict(x_test) ##Testing model on unseen data

score = accuracy_score(y_test, y_predict) #Comparing predictions to actual answers
print('{}% of samples were classified correctly !' .format(score*100))

f = open('model.p', 'wb') #Saving trained model
pickle.dump({'model': model}, f)
f.close()