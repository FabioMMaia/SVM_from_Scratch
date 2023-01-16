from modules import SVM
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

class SVM_one_vs_all():

    def __init__(self,kernel_grid, n_rounds = None, kernel_type='rbf', verbose=False):
        assert isinstance(kernel_grid, dict), 'kernel_grid deve ser um dicionário com chaves C e sigma'
        self.kernel_grid = kernel_grid
        self.kernel_type = kernel_type
        self.n_rounds=n_rounds
        self.kernel_params = kernel_grid
        self.verbose=verbose

    def fit(self, X_train, X_test, y_train, y_test):
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        assert len(np.unique(y_train))>2, 'Necessário se ter mais de duas classe na base de treino'

        best_model_classe = {}

        for classe in np.unique(y_train):
            y_train_trns = (y_train==classe).astype(int)
            y_train_trns = y_train_trns*2 - 1 
            y_test_trns = (y_test==classe).astype(int)
            y_test_trns = y_test_trns*2 - 1

            best_model_classe[classe] = None
            best_accuracy_test = -1
            best_f1_test = -1
            # best_recall_micro = -1
            iteration = 0

            if self.verbose:
                print('Iniciando testes dos {} modelos para a classe {}\n'.format(len(self.kernel_grid['sigma'])*len(self.kernel_grid['C']), classe))

            for sigma in self.kernel_grid['sigma']:
                for C in self.kernel_grid['C']:
                    classifier = SVM.SVM(C=C, kernel_type=self.kernel_type, kernel_params={'sigma':sigma}, verbose=False)
                    
                    # X_train_resampled_pos = X_train[(y_train_trns==1).flatten(),:]
                    # y_train_resampled_pos = y_train_trns[y_train_trns==1]
                    # m_pos = len(y_train_resampled_pos)

                    # X_train_resampled_neg = X_train[(y_train_trns==-1).flatten(),:][:3*m_pos, :]
                    # y_train_resampled_neg = y_train_trns[y_train_trns==-1][:3*m_pos]

                    # X_train_resampled = np.concatenate((X_train_resampled_pos, X_train_resampled_neg),axis=0)
                    # y_train_resampled = np.concatenate((y_train_resampled_pos, y_train_resampled_neg),axis=0)

                    # X_train_resampled= pd.DataFrame(X_train_resampled)
                    # y_train_resampled= pd.DataFrame(y_train_resampled)

                    classifier.fit(X_train, y_train_trns)

                    ypred_train = classifier.predict(X_train)
                    ypred_test = classifier.predict(X_test)

                    acc_train = np.mean(y_train_trns.flatten()==ypred_train.flatten())
                    acc_test = np.mean(y_test_trns.flatten()==ypred_test.flatten())

                    f1_score_train = f1_score(y_train_trns.flatten(), ypred_train.flatten(), average='weighted',zero_division=1)
                    f1_score_test = f1_score(y_test_trns.flatten(), ypred_test.flatten(), average='weighted',zero_division=1)


                    # recall_micro_test = (
                        
                    #     3*np.mean(
                    #         (ypred_test.flatten()[(ypred_test==1).flatten()]).flatten()
                    #          ==(y_test_trns.flatten()[(ypred_test==1).flatten()]).flatten()
                    #          ) 
                    #          + 1*np.mean
                    #          (
                    #             (ypred_test.flatten()[(ypred_test==-1).flatten()]).flatten()
                    #          ==(y_test_trns.flatten()[(ypred_test==-1).flatten()]).flatten())
                    #          )/4

                    # recall_micro_train = (
                        
                    #     3*np.mean(
                    #         (ypred_train.flatten()[(ypred_train==1).flatten()]).flatten()
                    #          ==(y_train_trns.flatten()[(ypred_train==1).flatten()]).flatten()
                    #          ) 
                    #          + 1*np.mean
                    #          (
                    #             (ypred_train.flatten()[(ypred_train==-1).flatten()]).flatten()
                    #          ==(y_train_trns.flatten()[(ypred_train==-1).flatten()]).flatten())
                    #          )/4

                    if f1_score_test>best_f1_test:
                        best_model_classe[classe]= {'model':classifier, 'acc_train': acc_train, 'acc_test': acc_test, 'f1_train':f1_score_train, 'f1_test':f1_score_test }
                        best_f1_test = f1_score_test
                        if self.verbose:
                            print('it:{} | melhor modelo para classe {} atualizado C:{} sigma:{} acc_train:{} acc_test:{} f1_train:{} f1_test:{}\n'.format(iteration,classe, C, sigma, acc_train, acc_test,f1_score_train,f1_score_test ))
                    if round(f1_score_test,3)==1:
                        break
                    
                    # if acc_train>best_accuracy_test:
                    #     best_model_classe[classe]= {'model':classifier, 'acc_train': acc_train, 'acc_test': acc_test}
                    #     best_accuracy_test = acc_test
                    #     if self.verbose:
                    #         print('it:{} | melhor modelo para classe {} atualizado C:{} sigma:{} acc_train:{} acc_test:{}\n'.format(iteration,classe, C, sigma, acc_train, acc_test))

                    # if recall_micro_train>best_recall_micro:
                    #     best_model_classe[classe]= {'model':classifier, 'acc_train': acc_train, 'acc_test': acc_test}
                    #     best_recall_micro = recall_micro_train
                    #     if self.verbose:
                    #         print('it:{} | melhor modelo para classe {} atualizado C:{} sigma:{} acc_train:{} acc_test:{}\n'.format(iteration,classe, C, sigma, acc_train, acc_test))


                    iteration+=1

        self.best_model_classe=best_model_classe
        return best_model_classe

    def predict_one_vs_all(self, X):
        
        if isinstance(X,pd.DataFrame):
            X = np.array(X)

        output_all = []

        for point in X:
            best_conf = -np.inf
            for classe_d, values_d in self.best_model_classe.items():
                (label, conf) = values_d['model'].predict(point.reshape(1,-1), confidence=True)
                
                if conf>best_conf:
                    best_conf = conf
                    output = classe_d

            output_all.append(output)
        
        return np.array(output_all).reshape(-1,1)





        






