import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier,plot_importance 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

def read_data():
    df = pd.read_csv('./insurance_claims.csv')
    df.drop('_c39',axis=1,inplace=True)
    df['authorities_contacted'] = df['authorities_contacted'].fillna("Other")
    return df 

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def encoding_label(df):
    #lets do Lable enconding coding to make more features 
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in df:
        if df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(df[col].unique())) <= 2:
                # Train on the training data
                le.fit(df[col])
                # Transform both training and testing data
                df[col] = le.transform(df[col])
                # Keep track of how many columns were label encoded
                le_count += 1
            
    print('%d columns were label encoded.' % le_count)
    return df

def data_profiling(df):
    profile = ProfileReport(df, title="Profiling Report", explorative=True,
    correlations={
        "pearson": {"calculate": True},
        "spearman": {"calculate": True},
        "kendall": {"calculate": True},
        "cramers": {"calculate": True},
        "phi_k": {"calculate": True},
    },)
    profile.to_file("insurance_claims.html")
    #heatmap
    correlation_heatmap(df)

def correlation_heatmap(df):
    #data = pd.get_dummies(data)
    #print('Training Features shape: ', data.shape)
    sns.set(style="white")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(18, 15))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(df.corr(numeric_only=True), cmap=cmap, vmax=.3, center=0,annot=True,fmt='.2g',
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

def get_unique(data):
    colum_name =[]
    unique_value=[]
    # Iterate through the columns
    for col in data:
        if data[col].dtype == 'object':
            # If 2 or fewer unique categories
            colum_name.append(str(col)) 
            unique_value.append(data[col].nunique())
    table= pd.DataFrame()
    table['Col_name'] = colum_name
    table['Value']= unique_value
                
    table=table.sort_values('Value',ascending=False)
    return table

def show_fraud_reported(colum_name,data):
    f, ax = plt.subplots(figsize=(20, 20))
    sns.countplot(x=colum_name,hue='fraud_reported',data=data)
    plt.show()

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

def run_lgb(X_train, X_test, y_train, y_test):
    lgbm = LGBMClassifier(learning_rate = 1)
    lgbm.fit(X_train, y_train)
    # accuracy score, confusion matrix and classification report of lgbm classifier
    lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))

    print(f"Training Accuracy of LGBM Classifier is {accuracy_score(y_train, lgbm.predict(X_train))}")
    print(f"Test Accuracy of LGBM Classifier is {lgbm_acc} \n")

    pred_test_y =lgbm.predict(X_test)
    print(f"{confusion_matrix(y_test, pred_test_y)}\n")
    print(classification_report(y_test, pred_test_y))
    print(roc_auc_score(y_test,pred_test_y))
    show_auc_score(y_test,pred_test_y)
    print('Plot feature importances...')
    ax = plot_importance(lgbm, max_num_features=10)
    plt.show()

def show_auc_score(y_test, pred_test):
    from sklearn import metrics
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred_test)
    roc_auc = metrics.auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(10, 10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def check_multicolinearity(df):
    # checking for multicollinearity

    plt.figure(figsize = (18, 12))

    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype = bool))

    sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g', linewidth = 1)
    plt.show()

def plot_data(X):
    plt.figure(figsize = (25, 20))
    plotnumber = 1
    for col in X.columns:
        if plotnumber <= 24:
            ax = plt.subplot(5, 5, plotnumber)
            sns.distplot(X[col])
            plt.xlabel(col, fontsize = 15)
            
        plotnumber += 1
        
    plt.tight_layout()
    plt.show()

def outlier_detection(X):
    plt.figure(figsize = (20, 15))
    plotnumber = 1

    for col in X.columns:
        if plotnumber <= 24:
            ax = plt.subplot(5, 5, plotnumber)
            sns.boxplot(X[col])
            plt.xlabel(col, fontsize = 15)
        
        plotnumber += 1
    plt.tight_layout()
    plt.show()

def train(X,y):
    # splitting data into training set and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    #print(X_train.head())
    num_df = X_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]
    
    # Scaling the numeric values in the dataset
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(num_df)
    scaled_num_df = pd.DataFrame(data = scaled_data, columns = num_df.columns, index = X_train.index)
    #print(scaled_num_df.head())
    X_train.drop(columns = scaled_num_df.columns, inplace = True)
    X_train = pd.concat([scaled_num_df, X_train], axis = 1)
    print(X_train.head())
    return X_train,y_train,X_test,y_test

def predict_and_compare_models(X_train,y_train,X_test,y_test):
    import svm
    import knn
    import decision_tree_classifier
    import random_forest_classifier
    import ada_boost_classifier
    import gradient_boosting_classifier
    import stochastic_gradient_boosting_classifier
    import cat_boost_classifier
    import extra_trees_classifier 
    import LGBM_classifier
    import Xgboost_classifier
    import vooting_classifier
    
    svc,svc_test_acc=svm.predict(X_train,y_train,X_test,y_test)
    knn,knn_test_acc=knn.predict(X_train,y_train,X_test,y_test)
    dtc,dtc_test_acc=decision_tree_classifier.predict(X_train,y_train,X_test,y_test)
    rand_clf,rand_clf_test_acc=random_forest_classifier.predict(X_train,y_train,X_test,y_test)
    ada,ada_test_acc=ada_boost_classifier.predict(X_train,y_train,X_test,y_test)
    xgb,xgb_test_acc=Xgboost_classifier.predict(X_train,y_train,X_test,y_test)
    gb,gb_acc = gradient_boosting_classifier.predict(X_train,y_train,X_test,y_test)
    sgb,sgb_acc = stochastic_gradient_boosting_classifier.predict(X_train,y_train,X_test,y_test)
    cat,cat_acc=cat_boost_classifier.predict(X_train,y_train,X_test,y_test)
    etc,etc_acc=extra_trees_classifier.predict(X_train,y_train,X_test,y_test)
    lgbm,lgbm_acc = LGBM_classifier.predict(X_train,y_train,X_test,y_test)
    vc,vc_test_acc=vooting_classifier.predict(svc,knn,dtc,rand_clf,ada,xgb,gb,sgb,cat,etc,lgbm,
    X_train,y_train,X_test,y_test)

    models = pd.DataFrame({
        'Model' : ['SVC', 'KNN', 'Decision Tree', 'Random Forest','Ada Boost', 'Gradient Boost', 'SGB', 'Cat Boost', 'Extra Trees', 'LGBM', 'XgBoost', 'Voting Classifier'],
        'Score' : [svc_test_acc, knn_test_acc, dtc_test_acc, rand_clf_test_acc, ada_test_acc, gb_acc, sgb_acc, cat_acc, etc_acc, lgbm_acc, xgb_test_acc, vc_test_acc]
    })
    models.sort_values(by = 'Score', ascending = False)
    px.bar(data_frame = models, x = 'Score', y = 'Model', color = 'Score', template = 'plotly_dark', 
       title = 'Models Comparison')
    plt.show()

def preprocessing(df):
    #outlier detection
    #outlier_detection(X)

    to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year']

    df.drop(to_drop, inplace = True, axis = 1)
    #print(df.head())
    #check_multicolinearity(df)
    
    # high correlation between age and months_as_customer
    df.drop(columns = ['age', 'total_claim_amount'], inplace = True, axis = 1)
    #print(df.info())

    # separating the feature and target columns
    X = df.drop('fraud_reported', axis = 1)
    y = df['fraud_reported']

    # extracting categorical columns
    cat_df = X.select_dtypes(include = ['object'])
    #print(cat_df)
    cat_df = pd.get_dummies(cat_df, drop_first = True)
    # extracting the numerical columns
    num_df = X.select_dtypes(include = ['int64'])
    # combining the Numerical and Categorical dataframes to get the final dataset
    X = pd.concat([num_df, cat_df], axis = 1)
    print(X.head())
    return X,y

if __name__=="__main__":
    df=read_data()
    #print(df)
    #data_profiling(df)
    X,y=preprocessing(df)
    X_train,y_train,X_test,y_test=train(X,y)
    predict_and_compare_models(X_train,y_train,X_test,y_test)

    