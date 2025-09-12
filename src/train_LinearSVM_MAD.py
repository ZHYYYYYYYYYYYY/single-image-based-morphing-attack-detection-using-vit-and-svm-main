from sklearn import svm, metrics, calibration
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


save_dir = './trained_classifiers/LMA_FF/'

train_path_b = './extracted_features/LMA_FF/train/bonafide/features.npy'
train_path_a = './extracted_features/LMA_FF/train/morph/features.npy'
test_path_b = './extracted_features/LMA_FF/test/bonafide/features.npy'
test_path_a = './extracted_features/LMA_FF/test/morph/features.npy'



# label non-morph as 0, morph as 1
x_train_bonafide = list(np.load(train_path_b))
x_train_attack = list(np.load(train_path_a))
x_train = x_train_bonafide + x_train_attack
y_train = list(np.ones(len(x_train_bonafide))) + list(np.zeros(len(x_train_attack)))


x_test_bonafide = list(np.load(test_path_b))
x_test_attack = list(np.load(test_path_a))
x_test = x_test_bonafide + x_test_attack
y_test = list(np.ones(len(x_test_bonafide))) + list(np.zeros(len(x_test_attack)))



clf = svm.SVC(C=1,kernel='linear',probability=True)
clf.fit(x_train, y_train) 
print(clf.score(x_test, y_test))

print(metrics.confusion_matrix(y_test, clf.predict(x_test)))

s = pickle.dumps(clf)

save_path = os.path.join(save_dir)
os.makedirs(save_path,exist_ok=True)
f = open(os.path.join(save_path,'svm.pkl'), 'wb+')
f.write(s)
f.close()


