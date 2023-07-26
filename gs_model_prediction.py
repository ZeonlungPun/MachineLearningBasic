import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten, Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras import Model
from xgboost import *
import catboost as cb
import lightgbm as lgb
from sklearn.svm import SVR
import matplotlib.pyplot as plt
"""
create CNN model for prediction
"""
def create_conv_NN(output_shape,nSNP):
    """
    :param output_shape:  single output is 1 or multi_output depend on the output shape of Y
    :param nSNP:    the length of SNP
    :return:  a complied model
    """
    nStride = 3  # stride between convolutions
    nFilter = 64  # filters
    model_cnn = Sequential()
    # add convolutional layer with l1 and l2 regularization
    model_cnn.add(
        Conv1D(nFilter, kernel_size=5, strides=nStride, input_shape=(nSNP, 1), kernel_regularizer='l1_l2'))
    model_cnn.add(Conv1D(nFilter, kernel_size=3, activation='relu'))
    # add pooling layer: takes maximum of two consecutive values
    model_cnn.add(MaxPooling1D(pool_size=2))
    # Solutions above are linearized to accommodate a standard layer
    model_cnn.add(Flatten())
    model_cnn.add(Dense(64))
    # activation layer
    model_cnn.add(Activation('relu'))
    model_cnn.add(Dense(32))
    model_cnn.add(Activation('relu'))
    model_cnn.add(Dense(output_shape))
    # Model Compiling
    model_cnn.compile(loss='mean_squared_error', optimizer='adam')
    return model_cnn

"""
create MLP(NN) model for prediction
"""
class MLP(Model):
    def __init__(self,output_shape):
        super(MLP,self).__init__()
        self.d1=Dense(64,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.d2=Dense(32,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.d4=Dense(output_shape,activation=None)
    def call(self, inputs,training=None, mask=None):
        x=self.d1(inputs)
        x=self.d2(x)
        y=self.d4(x)
        return y


"""
select the models to fit the genomic selection data (regression) 
the data has been imputed without any missing values 
this class is used to do 1 time experiment
"""

class MLModels(object):
    def __init__(self,dataset,target,selecting_model,test_ratio=0.2,multi_output=False,onehot_encoding=True,plot=False):

        """
        :param selecting_model: these models for selections: RandomForest ,
        GradientBoosting, CNN, MLP,KNN,XGBoost,lightGBM,CatBoost,
        :param test_ratio:the ratio of testing set
        :param multi_output:  using multi_output or not ; if true ,only CNN, MLP, CNN,XGBoost are available
        dataset: input variables with pandas dataframe format
        target: output variables with pandas dataframe format
        plot : if true ,plot the loss function step of trainning set and validation set ,ONLY FOR CNN and MLP
        """
        self.model_name=selecting_model
        self.multi_output=multi_output
        self.test_ratio=test_ratio
        self.dataset=dataset
        self.target=target
        self.onehot=onehot_encoding
        self.plot=plot

    def data_preprocess(self):
        #one hot encoding
        if self.onehot:
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse=False)
            self.onehot_input = encoder.fit_transform(self.dataset)
        else:
            self.onehot_input= self.dataset
        # Scaled data
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        np_scaled_x = min_max_scaler.fit_transform(self.onehot_input)
        X = np.array(np_scaled_x)

        if not self.multi_output:
            target = self.target.reshape((-1, 1))
        else:
            target=self.target

        np_scaled_y = min_max_scaler.fit_transform(target)
        Y = np.array(np_scaled_y)
        # Split data
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=self.test_ratio)


    def create_model(self):
        print("-------------------------------------------------------------------------------------------------")
        print("the model {} is beginning to train".format(self.model_name))
        if self.model_name == 'RandomForest':
            self.model = RandomForestRegressor(n_estimators=800)
        elif self.model_name == 'GradientBoosting':
            self.model = GradientBoostingRegressor(n_estimators=800)
        elif self.model_name == 'KNN':
            self.model = KNeighborsRegressor(n_neighbors=15)
        elif self.model_name == 'CNN':
            if self.multi_output:
                self.model = create_conv_NN(self.target.shape[1],self.X_train.shape[1])
            else:
                self.model = create_conv_NN(1)
        elif self.model_name == 'MLP':
            if self.multi_output:
                self.model = MLP(self.target.shape[1])
            else:
                self.model = MLP(1)
        elif self.model_name == "CatBoost":
            pool_train = cb.Pool(data=self.X_train, label=self.y_train)
            pool_valid = cb.Pool(data=self.X_test, label=self.y_test)
            iterations = 800
            early_stopping_rounds = 80
            self.model = cb.CatBoostRegressor(
                iterations=iterations,
                early_stopping_rounds=early_stopping_rounds)
            self.model.fit(pool_train, eval_set=pool_valid, plot=True)

        elif self.model_name =='lightGBM':
            lgb_train = lgb.Dataset(self.X_train, self.y_train)
            lgb_eval = lgb.Dataset(self.X_test, self.y_test, reference=lgb_train)
            params = {'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression'}
            callback = [lgb.early_stopping(stopping_rounds=20, verbose=True),
                        lgb.log_evaluation(period=20, show_stdv=True)]
            self.model = lgb.train(params, lgb_train, num_boost_round=1000,
                           valid_sets=[lgb_train, lgb_eval], callbacks=callback)

        elif self.model_name=='XGBoost':
            self.model= XGBRegressor(n_estimators=800)

        elif self.model_name=='BaggingRegressor':
            rf=RandomForestRegressor(n_estimators=200)
            self.model = BaggingRegressor(n_estimators=800,bootstrap=True,base_estimator=rf)

        elif self.model_name=='SVM':
            self.model = SVR(kernel='rbf')

        else:
            raise Exception("Invalid model name")

        # USE model to fit the data

        if self.model_name =='lightGBM' or self.model_name == "CatBoost":
            pass
        elif self.model_name =='MLP':
            self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics='mse')
            self.histroy = self.model.fit(self.X_train, tf.cast(np.array(self.y_train), tf.float32),
                                batch_size=100, epochs=10,
                                validation_data=(self.X_test, tf.cast(np.array(self.y_test), tf.float32)  )   )
            if self.plot:
                self.PlotDeepLearning()

        elif self.model_name == 'CNN':
            X2_train = np.expand_dims(self.X_train, axis=2)
            X2_test = np.expand_dims(self.X_test, axis=2)
            self.X_test=X2_test
            self.histroy =self.model.fit(X2_train, tf.cast(np.array(self.y_train), tf.float32),
                              batch_size=100, epochs=1,
                              validation_data=(X2_test, tf.cast(np.array(self.y_test), tf.float32)),
                              callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='min')])
            if self.plot:
                self.PlotDeepLearning()

        else:
            self.model.fit(self.X_train,self.y_train)

    @staticmethod
    def cal_correlation(pred, y_test, target):
        pred = pred.reshape((-1, 1))
        target_edit = target.reshape((-1, 1))
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        np_scaled = min_max_scaler.fit_transform(target_edit)
        target_pred = min_max_scaler.inverse_transform(pred)
        target_orig = min_max_scaler.inverse_transform(y_test)
        target_orig = target_orig[:, 0]
        target_orig = pd.Series(target_orig)
        target_pred = target_pred[:, 0]
        target_pred = pd.Series(target_pred)
        cor = target_orig.corr(target_pred, method='pearson')
        return cor

    def estimate_model(self):
        #use trained model to predict in testing set
        print('model {} begin to estimate'.format(self.model_name))
        pred=self.model.predict(self.X_test)

        if self.multi_output:
            scores=[]
            for i in range(self.target.shape[1]):
                score=self.cal_correlation(pred[:,i],self.y_test[:,i].reshape((-1,1)),self.target[:,i])
                scores.append(score)
            return scores
        else:
            return self.cal_correlation(pred, self.y_test, self.target)

    def PlotDeepLearning(self):
        acc = self.histroy.history['mse']

        val_acc = self.histroy.history['val_mse']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training mse')
        plt.plot(epochs, val_acc, 'b', label='Validation mse')
        plt.title('Training and validation mse')
        plt.xlabel('Epochs')
        plt.ylabel('mse')
        plt.legend()
        plt.savefig('loss.png')


def run_model(times,selected_model,input,output,test_ratio,multi_output,one_hot,plot):
    """
    this function use to run the algorithm above
    :param times: experiment times
    :param selected_model: these models for selections: RandomForest ,
        GradientBoosting, CNN, MLP,KNN,XGBoost,lightGBM,CatBoost
    :param input:  X
    :param output: Y
    :param test_ratio:   test set ratio
    :param multi_output:   using multi_output or not ; if true ,only CNN, MLP, CNN,XGBoost are available
    :param one_hot:   one-hot encoding or not ;default :true
    :param plot:  if true ,plot the loss function step of trainning set and validation set ,ONLY FOR CNN and MLP
    :return: pearson's correlation coefficent of ture Y and predicted Y
    """

    # instantialize the MLModels class
    model = MLModels(selecting_model=selected_model, dataset=input, target=output, test_ratio=test_ratio, multi_output=multi_output,
                     onehot_encoding=one_hot, plot=plot)

    scores = []
    for time in range(times):
        # prepare the data
        model.data_preprocess()
        # train the model
        model.create_model()
        score = model.estimate_model()
        print('the result of model {} in round {} is {}'.format(selected_model,time,score))
        scores.append(score)

    print('all results are:',scores)

    return scores


class CalculateImportance(MLModels):
    """
    example:
    imp=CalculateImportance(dataset=x,target=y,selecting_model='SVM',ExplainModel='CatBoost',EvaluatedModel='SVM')
    imp.data_preprocess()
    scores=imp.EstimateImportance()
    imp.PlotImportance(feature_num=[10,20,30],times=10)
    """
    
    def __init__(self,ExplainModel,EvaluatedModel,dataset,target,selecting_model,test_ratio=0.2,multi_output=False,onehot_encoding=True,plot=False):
        super(CalculateImportance,self).__init__(dataset,target,selecting_model,test_ratio=0.2,multi_output=False,onehot_encoding=True,plot=False)
        """
        this class use to give the importance of features and plot the model's accuracy trend when choosing different number of most vital features
        :param ExplainModel: give the feature importance:only when  RandomForest ,GradientBoosting,XGBoost,lightGBM, CatBoost are available
        :param EvaluatedModel:   use important features to fit the data
        """
        self.ExplainModel=ExplainModel
        self.EvaluatedModel=EvaluatedModel
    def EstimateImportance(self):
        """
        variable selection  function
        :return: feature importance score
        """
        if self.ExplainModel =='RandomForest':
            model=RandomForestRegressor(n_estimators=500)
            model.fit(self.X_train,self.y_train)

        elif self.ExplainModel=='GradientBoosting':
            model=GradientBoostingRegressor(n_estimators=500)
            model.fit(self.X_train, self.y_train)

        elif self.ExplainModel =='XGBoost':
            model=XGBRegressor(n_estimators=500)
            model.fit(self.X_train, self.y_train)

        elif self.ExplainModel=='lightGBM':
            lgb_train = lgb.Dataset(self.X_train, self.y_train)
            lgb_eval = lgb.Dataset(self.X_test, self.y_test, reference=lgb_train)
            params = {'task': 'train',
                      'boosting_type': 'gbdt',
                      'objective': 'regression'}
            callback = [lgb.early_stopping(stopping_rounds=20, verbose=True),
                        lgb.log_evaluation(period=20, show_stdv=True)]
            model = lgb.train(params, lgb_train, num_boost_round=1000,
                                   valid_sets=[lgb_train, lgb_eval], callbacks=callback)

        elif self.ExplainModel== 'CatBoost':
            pool_train = cb.Pool(data=self.X_train, label=self.y_train)
            pool_valid = cb.Pool(data=self.X_test, label=self.y_test)
            iterations = 800
            early_stopping_rounds = 80
            model = cb.CatBoostRegressor(
                iterations=iterations,
                early_stopping_rounds=early_stopping_rounds)
            model.fit(pool_train, eval_set=pool_valid, plot=True)
        else:
            raise ValueError('this model does not have feature importance attribute')
        return model.feature_importances_

    def PlotImportance(self,feature_num,times):
        """
        plot the accuracy trend along with increase of feature num
        :param feature_num: a list ,such as [10,20,30]  will choose respectively 10,20,30 most important features to fit the model
        :return:NONE
        """
        #pick specific impotant variables to fit and estimate the model
        y=[]
        imp=self.EstimateImportance()
        for num in feature_num:
            for time in range(times):
                print('estimate the model {} with number of {} features in round {}'.format(self.model_name,num,time))
                self.data_preprocess()
                max_index1 = imp.argsort()[-num::]
                x_train1, x_test1 = self.X_train[:, max_index1], self.X_test[:, max_index1]
                score=run_model(1,self.EvaluatedModel,self.dataset,self.target,test_ratio=0.2,multi_output=False,one_hot=True,plot=False)
                y.append(score[0])

        #plot the boxplot
        x = [i for i in feature_num for _ in range(times)]
        df = pd.DataFrame({'x': x, 'y': y})
        fig, ax = plt.subplots()
        # Boxplot
        df.boxplot(column='y', by='x', ax=ax, grid=False)

        # Median line and points
        medians = df.groupby('x')['y'].median()
        ax.plot(range(1, len(feature_num)+1), medians, color='red', linestyle='--', label='Median')
        ax.scatter(range(1, len(feature_num)+1), medians, color='red', marker='o')
        ax.set_xlabel('the most important features used')
        ax.set_ylabel('Pearson correlation coefficient')
        ax.set_title('Prediction accuracy trend by important features used ')
        plt.legend()
        plt.savefig('importance.png')





if __name__ == '__main__':
    #load the dataset
    blue = pd.read_csv('D:\\forest\\forestDataset\\forest_BLUE.csv')
    snp=pd.read_csv('D:\\forest\\forestDataset\\SNPdata_67168_012_renamed.csv')

    #merge the data based on Forest
    data=pd.merge(snp,blue,how='inner',on='Tree')
    y=data.iloc[:,-3::]
    x=data.iloc[:,1:-3]
    y=y.iloc[:,1]
    print(y.shape,x.shape)

    x=np.array(x)
    #find the null value
    index=np.where((x!=0) & (x!=1) & (x!=2))
    #conver the value to none
    x[index]=None
    x=pd.DataFrame(x)

    #calculate the missing value
    missing_num=x.isnull().sum(axis=0)
    #find the column with too many missing value
    drop_index=missing_num[(missing_num>=100)].index.tolist()
    print(drop_index)
    #drop the column with too many missing value
    x.drop(drop_index,axis=1,inplace=True)
    #check again
    missing_num=x.isnull().sum(axis=0)
    print(y.shape,x.shape)

    #imputed the missing value with KNN
    from sklearn.impute import KNNImputer
    imputer=KNNImputer(n_neighbors=10)
    imputed_x=imputer.fit_transform(x)
    imputed_x=np.round(imputed_x)
    y=np.array(y).reshape((-1,1))

    #scores=run_model(12,selected_model='SVM',input=imputed_x,output=y,test_ratio=0.2,multi_output=False,one_hot=True,plot=False)
    imp=CalculateImportance(dataset=x,target=y,selecting_model='SVM',ExplainModel='CatBoost',EvaluatedModel='SVM')
    imp.data_preprocess()
    scores=imp.EstimateImportance()
    print(scores )
    imp.PlotImportance(feature_num=[10,20,30],times=10)







