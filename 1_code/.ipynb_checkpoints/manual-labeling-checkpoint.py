import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn import metrics
import seaborn as sns
import itertools
from statistics import mode
pd.set_option("display.max_colwidth", 500)
%config InlineBackend.figure_format = 'retina'

# helper functions
def plot_confusion_matrix(y_true, y_pred, normalize=False):
    """
    Input:
        y_true    : array of true binary labels. Eg: [0, 1, 0, 0, 1].
        y_pred    : array of predicted probabilities. Eg: [0.05, 0.55, 0.2, 0.8, 0.95].
        normalize : whether to show counts of proportion in a cell
    
    Output:
        confusion matrix
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = np.unique(y_true)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if(normalize):
        precision=4
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], precision)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    pass

fake_pattern = re.compile(".+_fake\.jpg$")
true_pattern = re.compile(".+_true\.jpg$")
def get_y_true(name):
    if fake_pattern.search(name):
        return 1
    elif true_pattern.search(name):
        return 0
    else:
        return -1
    pass

def y_to_int(label):
    if(label == 'fake'):
        return 1
    elif label == 'real':
        return 0
    else:
        return -1
    pass


# read data
soumya_labels = '../00_source_data/human-labelers/soumya.csv'
preeti_labels = '../00_source_data/human-labelers/preeti.csv'
jashandeep_labels = '../00_source_data/human-labelers/jashandeep.csv'
forum_labels = '../00_source_data/human-labelers/forum.csv'
#prabhjot_labels = '../00_source_data/human-labelers/prabhjot.csv'

soumya = pd.read_csv(soumya_labels); soumya.shape
preeti = pd.read_csv(preeti_labels); preeti.shape
jashandeep = pd.read_csv(jashandeep_labels); jashandeep.shape
forum = pd.read_csv(forum_labels); forum.shape
#prabhjot = pd.read_csv(prabhjot_labels); prabhjot.shape

# filter data
soumya = soumya[["Label", "External ID"]]
soumya = soumya[soumya.Label !="{}"]
soumya.shape

preeti = preeti[["Label", "External ID"]]
preeti = preeti[preeti.Label !="{}"]
preeti.shape

jashandeep = jashandeep[["Label", "External ID"]]
jashandeep = jashandeep[jashandeep.Label !="{}"]
jashandeep.shape

forum = forum[["Label", "External ID"]]
forum = forum[forum.Label !="{}"]
forum.shape

#prabhjot = prabhjot[["Label", "External ID"]]
#prabhjot = prabhjot[prabhjot.Label !="{}"]
#prabhjot.shape



# SOUMYA
# get true label
soumya_y_true = soumya["External ID"].apply(lambda name: get_y_true(name))
y_true = soumya_y_true

# get annotations
soumya_y_pred = soumya.Label.apply(lambda text: text[-9:-5])
soumya_y_pred = soumya_y_pred.apply(lambda label: y_to_int(label))

plot_confusion_matrix(soumya_y_true, soumya_y_pred, normalize=True)
print('Soumya was {:4f}% accurate.'.format(100*metrics.accuracy_score(soumya_y_true, soumya_y_pred)))

# time to plot
sns.boxplot(x=soumya['Seconds to Label'])
soumya['Seconds to Label'].describe()

# PREETI
# get true label
preeti_y_true = preeti["External ID"].apply(lambda name: get_y_true(name))

# get annotations
soumya_y_pred = soumya.Label.apply(lambda text: text[-9:-5])
soumya_y_pred = soumya_y_pred.apply(lambda label: y_to_int(label))

plot_confusion_matrix(preeti_y_true, preeti_y_pred, normalize=True)
print('Preeti was {:4f}% accurate.'.format(100*metrics.accuracy_score(preeti_y_true, preeti_y_pred)))


# time to plot
sns.boxplot(x=data['Seconds to Label'])
preeti['Seconds to Label'].describe()

# JASHANDEEP
# get true label
jashandeep_y_true = jashandeep["External ID"].apply(lambda name: get_y_true(name))

# get annotations
soumya_y_pred = soumya.Label.apply(lambda text: text[-9:-5])
soumya_y_pred = soumya_y_pred.apply(lambda label: y_to_int(label))

plot_confusion_matrix(jashandeep_y_true, jashandeep_y_pred, normalize=True)
print('Jashandeep was is {:4f}% accurate.'.format(100*metrics.accuracy_score(jashandeep_y_true, jashandeep_y_pred)))

# time to plot
sns.boxplot(x=jashandeep['Seconds to Label'])
jashandeep['Seconds to Label'].describe()


# FORUM
# get true label
forum_y_true = forum["External ID"].apply(lambda name: get_y_true(name))

# get annotations
soumya_y_pred = soumya.Label.apply(lambda text: text[-9:-5])
soumya_y_pred = soumya_y_pred.apply(lambda label: y_to_int(label))

plot_confusion_matrix(forum_y_true, forum_y_pred, normalize=True)
print('Forum was is {:4f}% accurate.'.format(100*metrics.accuracy_score(forum_y_true, forum_y_pred)))

# time to plot
sns.boxplot(x=forum['Seconds to Label'])
forum['Seconds to Label'].describe()


# PRABHJOT
# get true label
prabhjot_y_true = prabhjot["External ID"].apply(lambda name: get_y_true(name))

# get annotations
soumya_y_pred = soumya.Label.apply(lambda text: text[-9:-5])
soumya_y_pred = soumya_y_pred.apply(lambda label: y_to_int(label))

plot_confusion_matrix(prabhjot_y_true, prabhjot_y_pred, normalize=True)
print('Forum was is {:4f}% accurate.'.format(100*metrics.accuracy_score(prabhjot_y_true, prabhjot_y_pred)))

# time to plot
sns.boxplot(x=prabhjot['Seconds to Label'])
prabhjot['Seconds to Label'].describe()


# VOTING CLASSIFIER
y_preds = np.concatenate((soumya_y_pred, preeti_y_pred, jashandeep_y_pred, forum_y_pred), axis=0).T  #prabhjot_y_pred
y_pred = np.apply_along_axis(mode, 1, y_preds)

plot_confusion_matrix(y_true, y_pred, normalize=True)
print('Human baseline was is {:4f}% accurate.'.format(100*metrics.accuracy_score(y_true, y_pred)))
