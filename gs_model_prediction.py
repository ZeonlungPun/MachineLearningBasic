import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Flatten, Conv1D, MaxPooling1D,Dropout,BatchNormalization,LSTM,GRU
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
from sklearn.linear_model import Ridge,LinearRegression
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
class WarmupExponentialDecay(Callback):
    def __init__(self,lr_base=0.0002,lr_min=0.0,decay=0,warmup_epochs=0):
        self.num_passed_batchs = 0   #一个计数器
        self.warmup_epochs=warmup_epochs
        self.lr=lr_base #learning_rate_base
        self.lr_min=lr_min #最小的起始学习率,此代码尚未实现
        self.decay=decay  #指数衰减率
        self.steps_per_epoch=0 #也是一个计数器
    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.steps_per_epoch==0:
            #防止跑验证集的时候呗更改了
            if self.params['steps'] == None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr,
                        self.lr*(self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.lr*((1-self.decay)**(self.num_passed_batchs-self.steps_per_epoch*self.warmup_epochs)))
        self.num_passed_batchs += 1
    def on_epoch_begin(self,epoch,logs=None):
    #用来输出学习率的,可以删除
        print("learning_rate:",K.get_value(self.model.optimizer.lr))


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
    #model_cnn.add( Conv1D(nFilter, kernel_size=3 , input_shape=(nSNP, 1),padding='same',))
    model_cnn.add(GRU(32, input_shape=(nSNP, 1) ))
    # model_cnn.add(Dropout(0.2))
    # model_cnn.add(BatchNormalization())
    # model_cnn.add(Conv1D(nFilter, kernel_size=3,padding='same', ))
    # model_cnn.add(Dropout(0.3))
    # model_cnn.add(BatchNormalization())
    # model_cnn.add(Conv1D(nFilter, kernel_size=3, padding='same',))
    # model_cnn.add(BatchNormalization())
    model_cnn.add(Flatten())
    model_cnn.add(Dense(3))
    # activation layer
    model_cnn.add(Activation('relu'))
    #model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(output_shape))
    # Model Compiling
    model_cnn.compile(loss='mean_squared_error', optimizer='sgd')
    return model_cnn

"""
create MLP(NN) model for prediction
"""
class MLP(Model):
    def __init__(self,output_shape):
        super(MLP,self).__init__()
        self.d1=Dense(64,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.05))
        self.d2=Dense(32,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.05))
        self.d3 = Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))
        self.bn=BatchNormalization()
        self.dropout=Dropout(0.1)
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(0.1)
        self.d4=Dense(output_shape,activation=None)
    def call(self, inputs,training=None, mask=None):
        x=self.d1(inputs)
        # x=self.bn(x)
        #x=self.dropout(x)
        #x=self.d2(x)
        # x = self.bn2(x)
        # x = self.dropout2(x)
        # x= self.d3(x)
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
        from sklearn.preprocessing import MinMaxScaler
        self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        np_scaled_x = self.min_max_scaler.fit_transform(self.onehot_input)
        X = np.array(np_scaled_x)

        if not self.multi_output:
            target = self.target.reshape((-1, 1))
        else:
            target=self.target

        np_scaled_y = self.min_max_scaler.fit_transform(target)
        Y = np.array(np_scaled_y)
        # Split data
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=self.test_ratio)

    def create_model_with_specific_data(self,train_x,train_y,test_x,test_y):
        print("-------------------------------------------------------------------------------------------------")
        print("the model {} is beginning to train".format(self.model_name))
        if self.model_name == 'RandomForest':
            self.model = RandomForestRegressor(n_estimators=200)
        elif self.model_name == 'GradientBoosting':
            self.model = GradientBoostingRegressor(n_estimators=800)
        elif self.model_name == 'KNN':
            self.model = KNeighborsRegressor(n_neighbors=20)
        elif self.model_name == 'CNN':
            if self.multi_output:
                self.model = create_conv_NN(train_y.shape[1], train_x.shape[1])
            else:
                self.model = create_conv_NN(nSNP=train_x.shape[1], output_shape=1)
        elif self.model_name == 'MLP':
            if self.multi_output:
                self.model = MLP(train_y.shape[1])
            else:
                self.model = MLP(1)
        elif self.model_name == "CatBoost":
            pool_train = cb.Pool(data=train_x, label=train_y)
            pool_valid = cb.Pool(data=test_x, label=test_y)
            iterations = 800
            early_stopping_rounds = 80
            self.model = cb.CatBoostRegressor(
                iterations=iterations,
                early_stopping_rounds=early_stopping_rounds)
            self.model.fit(pool_train, eval_set=pool_valid, plot=True)

        elif self.model_name == 'lightGBM':
            lgb_train = lgb.Dataset(train_x, train_y)
            lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)
            params = {'task': 'train',
                      'boosting_type': 'gbdt',
                      'objective': 'regression'}
            callback = [lgb.early_stopping(stopping_rounds=20, verbose=True),
                        lgb.log_evaluation(period=20, show_stdv=True)]
            self.model = lgb.train(params, lgb_train, num_boost_round=1000,
                                   valid_sets=[lgb_train, lgb_eval], callbacks=callback)

        elif self.model_name == 'XGBoost':
            self.model = XGBRegressor(n_estimators=800)

        elif self.model_name == 'BaggingRegressor':
            rf = RandomForestRegressor(n_estimators=200)
            self.model = BaggingRegressor(n_estimators=800, bootstrap=True, base_estimator=rf)

        elif self.model_name == 'SVM':
            self.model = SVR(kernel='rbf')

        elif self.model_name == 'rrBLUP':
            self.model= Ridge()
        elif self.model_name == 'BLUP':
            self.model =LinearRegression()

        else:
            raise Exception("Invalid model name")

        # USE model to fit the data

        if self.model_name == 'lightGBM' or self.model_name == "CatBoost":
            pass
        elif self.model_name == 'MLP':
            self.model.compile(optimizer='RMSprop', loss=tf.keras.losses.MeanSquaredError(), metrics='mse')
            self.histroy = self.model.fit(train_x, tf.cast(np.array(train_y), tf.float32),
                                          batch_size=100, epochs=100,
                                          validation_data=(test_x, tf.cast(np.array(test_y), tf.float32)),
                                          callbacks=[
                                              EarlyStopping(monitor='val_mse', patience=10, verbose=2, mode='min',
                                                            restore_best_weights=True)])

        elif self.model_name == 'CNN':
            X2_train = np.expand_dims(train_x, axis=2)
            X2_test = np.expand_dims(test_x, axis=2)
            print(X2_test.shape)
            self.histroy = self.model.fit(X2_train, tf.cast(np.array(train_y), tf.float32),
                                          batch_size=100, epochs=30,
                                          validation_data=(X2_test, tf.cast(np.array(test_y), tf.float32)),
                                          callbacks=[
                                              EarlyStopping(monitor='val_mse', patience=8, verbose=2, mode='min',restore_best_weights=True)])
        else:
            self.model.fit(train_x,train_y)
        return self.model

    def create_model(self):
        print("-------------------------------------------------------------------------------------------------")
        print("the model {} is beginning to train".format(self.model_name))
        if self.model_name == 'RandomForest':
            self.model = RandomForestRegressor(n_estimators=200)
        elif self.model_name == 'GradientBoosting':
            self.model = GradientBoostingRegressor(n_estimators=800)
        elif self.model_name == 'KNN':
            self.model = KNeighborsRegressor(n_neighbors=20)
        elif self.model_name == 'CNN':
            if self.multi_output:
                self.model = create_conv_NN(self.target.shape[1],self.X_train.shape[1])
            else:
                self.model = create_conv_NN(nSNP=self.X_train.shape[1],output_shape=1)
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
        elif self.model_name == 'rrBLUP':
            self.model= Ridge()
        elif self.model_name == 'BLUP':
            self.model =LinearRegression()

        else:
            raise Exception("Invalid model name")

        # USE model to fit the data

        if self.model_name =='lightGBM' or self.model_name == "CatBoost":
            pass
        elif self.model_name =='MLP':
            self.model.compile(optimizer='Adam', loss=tf.keras.losses.MeanSquaredError(), metrics='mse')
            self.histroy = self.model.fit(self.X_train, tf.cast(np.array(self.y_train), tf.float32),
                                batch_size=100, epochs=100,
                                validation_data=(self.X_test, tf.cast(np.array(self.y_test), tf.float32)),
                                callbacks=[WarmupExponentialDecay(lr_base=0.0002,decay=0.00002,warmup_epochs=10)] )
                                #callbacks=[EarlyStopping(monitor='val_loss', patience=15, verbose=2, mode='min',restore_best_weights=True)])
            if self.plot:
                self.PlotDeepLearning()

        elif self.model_name == 'CNN':
            X2_train = np.expand_dims(self.X_train, axis=2)
            X2_test = np.expand_dims(self.X_test, axis=2)

            self.X_test=X2_test
            self.histroy =self.model.fit(X2_train, tf.cast(np.array(self.y_train), tf.float32),
                              batch_size=30, epochs=30,
                              validation_data=(X2_test, tf.cast(np.array(self.y_test), tf.float32)),
                              callbacks=[EarlyStopping(monitor='val_loss', patience=15, verbose=2, mode='min',restore_best_weights=True)])
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

    def single_train(self):
        self.data_preprocess()
        self.create_model()
        score = self.estimate_model()
        print('the result of model is {}'.format( score))

    def single_predict(self,x_test,min_max_scale=False):
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse=False)
        if self.onehot:
            x_test=encoder.fit_transform(x_test.reshape((-1,1)) )
        pred=self.model.predict(x_test)
        if min_max_scale:
            pred=self.min_max_scaler.inverse_transform(pred.reshape((-1,1)))
        return pred



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

def run_model_with_train_test_data(selected_model,train_x,train_y,test_x,test_y):
    model = MLModels(selecting_model=selected_model, dataset=train_x, target=train_y, test_ratio=0,
                     multi_output=False,onehot_encoding=False, plot=False)
    model=model.create_model_with_specific_data(train_x,train_y,test_x,test_y)
    if selected_model=='CNN':
        test_x=np.expand_dims(test_x, axis=2)
    print(test_x.shape)
    pred=model.predict(test_x).reshape((-1,))
    testy=test_y.reshape((-1,))
    score=np.corrcoef(pred,testy)[0,1]

    return score



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
            model=RandomForestRegressor(n_estimators=20)
            model.fit(self.X_train,self.y_train)

        elif self.ExplainModel=='GradientBoosting':
            model=GradientBoostingRegressor(n_estimators=20)
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
    #vcf_path="/mnt/soysnp50k.vcf"
    vcf_path='N2000.vcf'
    with open(vcf_path, "r") as f:
        # if the line does not start as "##", this line is the header line
        header = next((i for i, line in enumerate(f) if not line.startswith('##')), 0)
    # read imthe file from header row
    df = pd.read_csv(vcf_path, header=header, sep='\t', dtype=str)
    df = df.iloc[0:-2, :]

    columns = []
    columns.append(df.columns[2])
    columns.extend(df.columns[9:])
    # concat the chromosome name and position as new ID for SNPs
    #df['ID'] = df['#CHROM'].str.cat(df['POS'], sep='_')
    # set ID column as index and drop the useless columns
    df = df[columns].set_index('ID')
    df.dropna(axis=0, how='any')

    # clear values of DataFrame
    # sometimes, values might be with annotations, for example, '1/1(some annotation)', which should be deleted
    for column in columns[1:]:
        # only the first three characters could be kept
        #df[column] = df[column].str.slice(0, 3)
        # create a temporary series, which is replaced by rules

        temp_series = df[column].replace({r'1/1': '1', r'1|1': '1', r'0/1': '2', r'0|1': '2',r'0/0':'0',r'0|0':'0',r'1|0':'2'})
        # # replace column of DataFrame with assigned series
        df[column] = temp_series.astype(np.int32)
        # except:
        #     df.drop(column,axis=1)

    dfall=df.transpose()
    df=dfall
    #random_number=np.random.permutation(dfall.shape[0])
    #df=dfall.iloc[random_number[0:2000],:]

    oil = pd.read_csv("grin_oil.csv", sep=',', header=0)
    # oil=pd.read_csv("/home/kingargroo/gs/grin_FLWRCOLOR.csv")
    oil_index1 = oil.iloc[:, 1].astype(str)
    oil_index2 = oil.iloc[:, 2].astype(str)
    oil_index3 = oil.iloc[:, 3].fillna("").astype(str)
    oil.index = oil_index1 + oil_index2 + oil_index3

    intersect_df = pd.merge(oil, df, left_index=True, right_index=True, how='inner')
    y = np.array(intersect_df.loc[:, 'observation_value'])
    y =np.log(y+10)
    x = np.array(intersect_df.iloc[:, 26::])

    # z=intersect_df.iloc[:, 26::]
    # counts_per_column = z.apply(z.value_counts)
    # nan_column_id = np.where(np.sum(counts_per_column.isnull()) != 0)
    # sub_data=np.array(counts_per_column)[0:2,nan_column_id].squeeze(1)
    # p_per_column_sub=sub_data/np.sum(sub_data,axis=0)
    # p_2nd_per_column_sub=np.sort(p_per_column_sub, axis=0)[1, :]
    # one_minu_p2nd_per_column_sub=1-p_2nd_per_column_sub
    # p_per_column=counts_per_column/np.sum(np.array(counts_per_column),0)
    # p_2nd_per_column=np.sort(p_per_column, axis=0)[1, :]
    # p_2nd_per_column[nan_column_id]=p_2nd_per_column_sub
    # one_minu_p2nd_per_column=1-p_2nd_per_column
    # one_minu_p2nd_per_column[nan_column_id]=one_minu_p2nd_per_column_sub
    # denominator=np.sum(2*p_2nd_per_column*one_minu_p2nd_per_column)
    # G=x@x.T /denominator








    # blue=pd.read_csv('predict.csv')
    #
    #
    # f1=pd.read_csv('F1.GenoType.20231205_012.txt',sep='\t')
    # data=pd.merge(left=f1,right=blue,on='IID')
    # y=data.loc[:,'predicted.value']
    # x=data.iloc[:,7:-23]
    # x,y=np.array(x),np.array(y)

    import shap,xgboost

    model = xgboost.train({"learning_rate": 0.001,"subsample": 0.5,}, xgboost.DMatrix(x, label=y), 200)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    shap_values = np.mean(np.abs(shap_values), axis=0)
    non_zero_id=np.where(shap_values>0)[0]
    #trainx,testx=trainx[:,non_zero_id],testx[:,non_zero_id]
    x_=x[:,non_zero_id]
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=x.shape[0])
    # pca.fit(x)
    # x_ = pca.transform(x)
    # x_=pd.DataFrame(x_)
    # x_.to_csv("input_file.csv")


    # score=run_model_with_train_test_data(selected_model='SVM', train_x=trainx, train_y=trainy, test_x=testx, test_y=testy)
    # print(score)
    run_model(times=5, selected_model='MLP', input=x_, output=y, test_ratio=0.2, multi_output=False, one_hot=True, plot=False)

    # model = MLModels(selecting_model='SVM', dataset=x, target=y, test_ratio=0.2,
    #                  multi_output=False,
    #                  onehot_encoding=True, plot=False)
    #
    # model.single_train()
    # pred=model.single_predict(x)
    #
    # error=y.reshape((-1,1))-pred
    # steer=np.std(error)

    # true=pd.DataFrame(y,index=range(len(y)))
    # pred=pd.DataFrame(pred)
    # error=pd.DataFrame(error,index=range(len(error)))
    # import seaborn as sns
    # pred_true_data=pd.concat([true,pred],axis=1)
    # pred_true_data.columns=['true','predict']
    # #regression plot
    # sns.lmplot(data=pred_true_data,x='true',y='predict')
    # plt.title('YLD14_XX 22 cor=0.19')
    #
    #
    # #殘差圖
    # plt.scatter(true,error)
    # plt.axhline(y=0, color='r', linestyle='--')  # 添加水平虛線，表示殘差為0
    # plt.xlabel('true Values')
    # plt.ylabel('Residuals')
    # plt.title('YLD14_XX 22 cor=0.19')
    # # 顯示圖形
    # plt.show()

    #scores=run_model(10,selected_model='XGBoost',input=x,output=y,test_ratio=0.2,multi_output=False,one_hot=True,plot=False)
    # imp=CalculateImportance(dataset=x,target=y,selecting_model='KNN',ExplainModel='XGBoost',EvaluatedModel='SVM')
    # imp.data_preprocess()
    # scores=imp.EstimateImportance()
    # print(scores)
    # imp.PlotImportance(feature_num=[400,600,800,1000,1500,2500],times=5)







