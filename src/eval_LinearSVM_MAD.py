from sklearn import svm, metrics, calibration
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


model_path = './trained_classifiers/LMA_FF/svm.pkl'

test_path_b = './extracted_features/LMA_FF/test/bonafide/features.npy'
filename_test_path_b = './extracted_features/LMA_FF/test/bonafide/filenames.csv'
test_path_a = './extracted_features/LMA_FF/test/morph/features.npy'
filename_test_path_a = './extracted_features/LMA_FF/test/morph/filenames.csv'

save_path = './mad_scores/test.txt'


# label non-morph as 0, morph as 1
x_test_b = list(np.load(test_path_b))
x_test_a = list(np.load(test_path_a))
x_test = x_test_b + x_test_a

y_test_b =list(np.ones(len(x_test_b)))
y_test_a = list(np.zeros(len(x_test_a)))
y_test =  y_test_b + y_test_a

f = open(model_path,'rb')
clf = pickle.load(f)
f.close()

print(clf.score(x_test, y_test))
print(metrics.confusion_matrix(y_test, clf.predict(x_test)))

s = pickle.dumps(clf)


y_hat_test_a_test_temp = clf.predict_proba(x_test_a)
y_hat_test_a = []
for y_hat in y_hat_test_a_test_temp:
    y_hat_test_a.append(y_hat[1])

y_hat_test_b_temp = clf.predict_proba(x_test_b)
y_hat_test_b = []
for y_hat in y_hat_test_b_temp:
    y_hat_test_b.append(y_hat[1])


with open(filename_test_path_b) as f:
    filename_test_b = f.readlines()
with open(filename_test_path_a) as f:
    filename_test_a = f.readlines()


with open(save_path, 'a') as output_file:
    for i in range(len(y_hat_test_a)):
        filename = filename_test_a[i].split('\n')[0]
        line = filename+','+str(y_hat_test_a[i])+','+str(y_test_a[i])
        output_file.write(line+'\n')
    for i in range(len(y_hat_test_b)):
        filename = filename_test_a[i].split('\n')[0]
        line = filename+','+str(filename_test_b[i])+','+str(y_test_b[i])
        output_file.write(line+'\n')


