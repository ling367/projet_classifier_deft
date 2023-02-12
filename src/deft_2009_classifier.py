from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def df_train_parsing(nom_fichier):
    """This function opens the xml file and do a parsing
    for df_train by looking all the textual values for <p> and
    the attribute 'valeur' for PARTI """
    with open (nom_fichier,'r') as f:
        file = f.read()
    soup = bs(file,'xml')
    texte = soup.find_all('texte')
    parti = soup.find_all('PARTI')
    data_train = []
    for i in range (0, len(texte)):
        rows = [texte[i].getText()[1:],parti[i].attrs['valeur']]
        data_train.append(rows)
    df_train = pd.DataFrame(data_train,columns =['texte','PARTI'])
    return df_train

def df_test_parsing(nom_test_xml,nom_test_txt):
    """ this function parses the xml file for df_test by looking all the textual
  values for <p>  and adds the column with the value of PARTI to df_test from the txt"""
    with open (nom_test_xml,'r') as f2:
        file2 = f2.read()
    soup2 = bs(file2,'xml')
    texte2 = soup2.find_all('texte')
    data_test = []
    for i in range (0, len(texte2)):
        rows = [texte2[i].getText()[1:]]
        data_test.append(rows)
    df_test = pd.DataFrame(data_test,columns =['Texte'])
    fichier_ref = pd.read_csv(nom_test_txt, sep = "\t", header = None)
    full = pd.concat([df_test, fichier_ref], axis=1)
    df_test = full.drop(columns=0, axis=1)
    df_test.columns = ['texte', 'PARTI']
    df_test.dropna(0,inplace=True)
    return df_test

def concat_dftrain_dftest(df_train, df_test):
  """ this function concats df_train and df_test """
  df_full= pd.concat([df_train,df_test])
  return df_full

def prepare_dataframes():
  """ this function is for dataframe preprocessing. It parses the files and returns
  all the dataframes """
  df_train = df_train_parsing('./data/deft09_parlement_appr_fr.xml')
  df_test = df_test_parsing('./data/deft09_parlement_test_fr.xml','./data/deft09_parlement_ref_fr.txt')
  df_full = concat_dftrain_dftest(df_train, df_test)
  return df_train, df_test, df_full
    
def data_visualization(y):
  """ this function is for data visualisation with bar plot to see the
  distribution of the classes. """
  dico={}
  parti=y.values.tolist()
  for el in parti:
      if el not in dico.keys():
          dico[el]=1
      else:
          dico[el]+=1
  print(dico)
  plt.bar(range(len(dico)), list(dico.values()), align='center')
  plt.xticks(range(len(dico)), list(dico.keys()))
  plt.show()
  
def fitGridmodel(grid, X_train, y_train, X_test, y_test):
  """ this function takes a grid, X_train, y_train, X_test, y_test as arguments. 
  The output is a f1-score given with the best parameters found by the gridSarchCV 
  function. """
  grid.fit(X_train, y_train)
  grid.score(X_test, y_test)
  y_pred_test = grid.predict(X_test)
  y_pred_train = grid.predict(X_train)
  best_Hyperparameters_rdf = grid.best_params_
  print ("Best Hyperparameters :", best_Hyperparameters_rdf )
  cm=confusion_matrix(y_test, y_pred_test)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)
  print('F1 score sur le train :', f1_score(y_train, y_pred_train, average='micro'))
  print(f"{grid} --> {grid.score(X_train, y_pred_train)}")
  print(classification_report(y_train, y_pred_train))
  print ("***********************\n")
  print('F1 score sur le test :',f1_score(y_test, y_pred_test, average='micro'))
  print(f"{grid} --> {grid.score(X_test, y_test)}")
  print(classification_report(y_test, y_pred_test))
  disp.plot()
  plt.show()

def fit_model(model, X_train, y_train, X_test, y_test):
  """ this function takes a model, X_train, y_train, X_test, y_test as arguments. 
  The output is a f1-score given with a classification report for train and test
  and a confusion matrix. """
  model.fit(X_train, y_train)
  model.score(X_test, y_test)
  y_pred_test = model.predict(X_test)
  y_pred_train = model.predict(X_train)
  cm=confusion_matrix(y_test, y_pred_test)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
  print('F1 score sur le train :', f1_score(y_train, y_pred_train, average='micro'))
  print(f"{model} --> {model.score(X_train, y_pred_train)}")
  print(classification_report(y_train, y_pred_train))
  print ("***********************\n")
  print('F1 score sur le test :',f1_score(y_test, y_pred_test, average='micro'))
  print(f"{model} --> {model.score(X_test, y_test)}")
  print(classification_report(y_test, y_pred_test))
  disp.plot()
  plt.show()

# def GridSearchModel(X_train, y_train, model, parameters, cv):
#   CV_model = GridSearchCV(estimator=model, param_grid=parameters, cv=5,verbose=2,n_jobs=-1)
#   CV_model.fit(X_train, y_train)
#   CV_model.cv_results_
#   model_best_score = CV_model.best_score_
#   model_best_params = CV_model.best_params_
#   return model_best_score,model_best_params

def val_curve(model, X , y, param_range, param_name):
  """ the function prints the validation curve for a given model and data. """
  train_scores, test_scores = validation_curve(estimator=model,
                                               X=X, y=y,
                                               cv=10, param_name=param_name, param_range=param_range, scoring="accuracy")
  train_mean = np.mean(train_scores, axis=1)
  test_mean = np.mean(test_scores, axis=1)
  plt.subplots(1, figsize=(10,10))
  plt.plot(param_range, train_mean, color='blue', label="Training Accuracy")
  plt.plot(param_range, test_mean, color='green', label="Validation Accuracy")
  plt.xlabel(f'Parameter {param_name}')
  plt.ylabel('Accuracy')
  plt.tight_layout()
  plt.show()
  return 'validation curve'

def learn_curve(model, X, y):
  """ the function prints the learning curve for a given model and data. """
  train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.05, 1.0, 120))
  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)
  test_mean = np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)
  plt.subplots(1, figsize=(10,10))
  plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
  plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
  plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
  plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
  plt.title(f"Learning Curve - {model}")
  plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
  plt.tight_layout()
  plt.show()
  return 'learning curve'
  
def main():
  df_train, df_test, df_full = prepare_dataframes()
  X_train, y_train = df_train["texte"], df_train["PARTI"]
  X_test, y_test = df_test["texte"], df_test["PARTI"]
  X_full, y_full  = df_full["texte"], df_full["PARTI"]

  sw = stopwords.words('french')
  
  #---VECTORIZATION part -----------------------------------------
  vec_tfidf = TfidfVectorizer(sublinear_tf = True, stop_words=sw, max_df=1.0)
  vec_count = CountVectorizer(max_df = 0.5)
  # X_vecCount_train = scaler.fit(vec_count.fit_transform(X_train))
  # X_vecCount_test = scaler.transform(vec_count.transform(X_test))
  # X_vecCount_full = scaler.fit_transform(vec_count.fit_transform(X_full))
  X_vecTfidf_train = vec_tfidf.fit_transform(X_train)
  X_vecTfidf_test = vec_tfidf.transform(X_test)
  X_vecTfidf_full = vec_tfidf.fit_transform(X_full)
  #----------------------------------------------------------------

  #--- RESAMPLING part --------------------------------------------
  random_underSampling = RandomUnderSampler(random_state=42, sampling_strategy = 'not minority' )
  # random_overSampling = RandomOverSampler(random_state=42, sampling_strategy = 'not majority')
  smote = SMOTE(random_state =42, sampling_strategy='all') # générer de nouveaux individus minoritaires qui ressemblent aux autres
  X_randomUnderSampling_train, y_randomUnderSampling_train = random_underSampling.fit_resample(X_vecTfidf_train, y_train)
  # X_randomOverSampling_train, y_randomOverSampling_train = random_overSampling.fit_resample(X_vecTfidf_train, y_train)
  X_smote_train, y_smote_train = smote.fit_resample(X_vecTfidf_train, y_train)
  #----------------------------------------------------------------

  # data_visualization(y_train)
  # data_visualization(y_test)
  # data_visualization(y_randomUnderSampling_train) 
  # data_visualization(y_randomOverSampling_train)
  # data_visualization(y_smote_train)
  # data_visualization(y_smoteen_train)
  
  #---- MODELS CREATION ----------------------------
  rforest = RandomForestClassifier(random_state=42)
  svc = SVC(random_state=42)
  lr = LogisticRegression(random_state=42, max_iter=1000)
  nb= MultinomialNB()
  gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
  svc2 = LinearSVC(C=50, dual=False, multi_class='ovr',random_state=42)
  # vc=VotingClassifier([('rforest', rforest), ('lr', lr), ("nb", nb)], voting="soft", weights=(1,1,2))
  # sgdc = make_pipeline(StandardScaler(with_mean=False), SGDClassifier(max_iter=1000, tol=1e-3))
  #--------------------------------------------------

  #---- HYPER-PARAMETERS GRIDS ----------------------
  param_grid_rforest = {
      'n_estimators': [100,200,500,700,1000], 'max_depth' : [7,9,11,20], 
      'criterion' :['gini', 'entropy'], 'max_features':['sqrt'],}
  param_grid_svc = {
      'C':[1,10,100], 'gamma':[1,0.1,0.001], 'kernel':['linear','rbf']}
  param_grid_lr = {
      'C':np.logspace(-3,3,7), 'penalty':['l1','l2']}
  param_grid_nb = {
      'alpha': [1,0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
  #--------------------------------------------------

  ### These models have to be tested with resampling methods for comparisons
  # grid = GridSearchCV(rforest, param_grid_rforest, cv=5, verbose=3, n_jobs=-1)
  # grid = GridSearchCV(svc, param_grid_svc, cv=5, verbose=3, n_jobs=-1)
  # grid = GridSearchCV(lr, param_grid_lr, cv=5, verbose=3, n_jobs=-1)
  # grid = GridSearchCV(nb, param_grid_nb, cv=5, verbose=3, n_jobs=-1)
  # fitGridmodel(grid, X_vecTfidf_train, y_train, X_vecTfidf_test, y_test)

  #--- RANDOM FOREST -------------------------------------  
  print('RANDOM FOREST')
  print("Results with undersampling : ")
  rforest_rus = RandomForestClassifier(random_state = 42, criterion='gini', max_depth=20, max_features='sqrt', n_estimators=1000)
  fit_model(rforest_rus, X_randomUnderSampling_train, y_randomUnderSampling_train, X_vecTfidf_test, y_test)
  print("Cross Validation Score Undersampling:", cross_val_score(rforest_rus, X_randomUnderSampling_train, y_randomUnderSampling_train).mean())
  print("\n\n\n")
  print("Results with SMOTE : ")
  rforest_smt = RandomForestClassifier(random_state=42,criterion='entropy', max_depth=20, max_features='sqrt', n_estimators=700)
  fit_model(rforest_smt,X_smote_train, y_smote_train, X_vecTfidf_test , y_test)
  print("Cross Validation Score SMOTE:", cross_val_score(rforest_smt,X_smote_train, y_smote_train).mean())
  # VALIDATION CURVE
  # param_range = np.arange(1, 11)
  # param_name = "max_depth"
  # val_curve(rforest, X_randomUnderSampling_train, y_randomUnderSampling_train, param_range, param_name)
  # LEARNING CURVE 
  # learn_curve(rforest, X_randomUnderSampling_train, y_randomUnderSampling_train)
  # print("\n\n\n")
  #-------------------------------------------------------

  #--- SVC ----------------------------------------------
  print('SVC')
  print("Results with undersampling : ")
  fit_model(svc2, X_randomUnderSampling_train, y_randomUnderSampling_train, X_vecTfidf_test, y_test)
  print("Cross Validation Score Undersampling:", cross_val_score(svc2, X_randomUnderSampling_train, y_randomUnderSampling_train).mean())
  print("\n\n\n")
  print("Results with SMOTE : ")
  fit_model(svc2, X_smote_train, y_smote_train, X_vecTfidf_test , y_test)
  print("Cross Validation Score SMOTE:", cross_val_score(svc2, X_smote_train, y_smote_train).mean())
  print("\n\n\n")
  # VALIDATION CURVE
  # param_range=np.logspace(-6, -1, 5)
  # param_name="gamma"
  # val_curve(svc,X_smote_train, y_smote_train, param_range, param_name)
  # LEARNING CURVE 
  # learn_curve(svc,X_smote_train, y_smote_train)
  # print("\n\n\n")
  #------------------------------------------------------- 

  #---VOTING CLASSIFIER -------------------------------------------------
  # print("SMOTE: ")
  # model(vc, X_smote_train, y_smote_train, X_vecTfidf_test, y_test)
  # print(cross_val_score(vc, X_smote_train, y_smote_train).mean())
  # print("\n\n\n")
  # model(svc, X_smote_train, y_smote_train, X_vecTfidf_test, y_test)
  # param_grid_vc = { }
  # model_best_score, model_best_params = GridSearchModel(X_smote_train, y_smote_train, model=RandomForestClassifier(), parameters= param_grid_rforest, cv=5)
  # grid_vc = GridSearchCV(vc, grid_vc, verbose=3, n_jobs=-1)
  # print("paramètres appliqués :", grid_svc.best_params_)
  # fit_model(grid_svc, X_randomUnderSampling_train, y_randomUnderSampling_train, X_vecTfidf_test, y_test)
  # VALIDATION CURVE
  # param_range=np.logspace(-6, -1, 5)
  # param_name="gamma"
  # val_curve(svc, X_smote_train, y_smote_train, param_range, param_name)
  # LEARNING CURVE 
  # learn_curve(svc, X_smote_train, y_smote_train)
  # print("Oversample: ")
  # model(vc, X_randomOverSampling_train, y_randomOverSampling_train, X_vecTfidf_test, y_test)
  # print(cross_val_score(vc, X_randomOverSampling_train, y_randomOverSampling_train).mean())
  #----------------------------------------------------------------------

  #--- LOGISTIC REGRESSION ----------------------------------------------
  print('LOGISTIC REGRESSION')
  print("Results with undersampling : ")
  lr_rus= LogisticRegression(random_state=42, max_iter=1000, C=1.0, penalty='l2')
  fit_model(lr_rus,X_randomUnderSampling_train, y_randomUnderSampling_train, X_vecTfidf_test, y_test)
  print("Cross Validation Score Undersampling:", cross_val_score(lr_rus, X_randomUnderSampling_train, y_randomUnderSampling_train).mean())
  print("\n\n\n")
  print("Results with SMOTE : ")
  lr_smote= LogisticRegression(random_state=42, max_iter=1000, C=10, penalty= 'l2')
  fit_model(lr_smote, X_smote_train, y_smote_train, X_vecTfidf_test, y_test)
  print(cross_val_score("Cross Validation Score SMOTE:", lr_smote, X_smote_train, y_smote_train).mean())
  #----------------------------------------------------------------------

  #--- NAIVE BAYES ----------------------------------------------
  print('NAIVE BAYES')
  print("Results with undersampling : ")
  nb_rus= MultinomialNB(alpha= 0.5)
  fit_model(nb_rus, X_randomUnderSampling_train, y_randomUnderSampling_train, X_vecTfidf_test, y_test)
  print("Cross Validation Score Undersampling :", cross_val_score(nb_rus, X_randomUnderSampling_train, y_randomUnderSampling_train).mean())
  print("\n\n\n")
  print("Results with SMOTE : ")
  nb_smote = MultinomialNB(alpha= 1e-05)
  fit_model(nb_smote, X_smote_train, y_smote_train, X_vecTfidf_test, y_test)
  print("Cross Validation Score SMOTE :",cross_val_score(nb_smote, X_smote_train, y_smote_train).mean())
  print("\n\n\n")
  #--------------------------------------------------------------

   

if __name__ == "__main__":
  main()
