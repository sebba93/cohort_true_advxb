# Libraries
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline

##################################################################
#                    Funciones parte 1                           #
##################################################################

# This function generates a plot illustrating the relationship between the relevant
# variables for the XGBoost and Random Forest models.
# taken from the library
# https://github.com/kartikay-bagla/bump-plot-python/blob/master/bumpplot.py
# we made modifications to the code
def bumpchart(df, show_rank_axis= True, rank_axis_distance= 1.1,
              ax= None, scatter= False, holes= False,
              line_args= {}, scatter_args= {}, hole_args= {}):

    if ax is None:
        left_yaxis= plt.gca()
    else:
        left_yaxis = ax

    # Creating the right axis.
    right_yaxis = left_yaxis.twinx()

    axes = [left_yaxis, right_yaxis]

    # Creating the far right axis if show_rank_axis is True
    if show_rank_axis:
        far_right_yaxis = left_yaxis.twinx()
        axes.append(far_right_yaxis)

    for col in df.columns:
        y = df[col]
        x = df.index.values
        # Plotting blank points on the right axis/axes
        # so that they line up with the left axis.
        for axis in axes[1:]:
            axis.plot(x, y, alpha= 0)

        left_yaxis.plot(x, y, **line_args, solid_capstyle='round')

        # Adding scatter plots
        if scatter:
            left_yaxis.scatter(x, y, **scatter_args)

            #Adding see-through holes
            if holes:
                bg_color = left_yaxis.get_facecolor()
                left_yaxis.scatter(x, y, color= bg_color, **hole_args)


    # Configuring the axes so that they line up well.
    axes[0].invert_yaxis()
    axes[0].set_yticks(df.loc['rf',:].sort_values(ascending=True))
    axes[1].invert_yaxis()
    axes[1].set_yticks(df.loc['xgb',:].sort_values(ascending=True))


    # Sorting the labels to match the ranks.
    left_labels = df.loc['rf',:].sort_values(ascending=True).index
    right_labels =  df.loc['xgb',:].sort_values(ascending=True).index

    left_yaxis.set_yticklabels(left_labels)
    right_yaxis.set_yticklabels(right_labels)

    # Setting the position of the far right axis so that it doesn't overlap with the right axis
    if show_rank_axis:
        far_right_yaxis.spines["right"].set_position(("axes", rank_axis_distance))

    return axes

# Run the Random Forest with a number of trees 'c_a', maximum depth 'depth',
# and a seed 'r_s'.
# Returns the average f1_score.
def evaluate(X, y, c_a, depth, r_s, cv):
    # Create the Random Forest Classifier model
    model = RandomForestClassifier(n_estimators=c_a, max_depth=depth, random_state=r_s)
    model.fit(X, y)
    # Perform cross-validation with 'cv' folds and obtain predictions
    y_pred = cross_val_predict(model,X,y,cv=cv)
    median=f1_score(y,y_pred,average='macro')
    return median,model,y_pred

# Returns the convergence matrix with the f1-score metric
def convergence(X, y, path, r_s, num_folds):
    depths = list(range(3, 8))
    quantities = list(range(100, 1000, 100))
    f1_list = []
    depth_list = []
    quantity_list = []
    stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=False)
    best_median=0
    best_params=[0,0]
    for depth in depths:
        for c_a in quantities:
            f1,model,y_pred= evaluate(X, y, c_a, depth, r_s, stratified_kfold)
            if f1 >= best_median:
                best_y_pred=y_pred
                best_median=f1
                best_model=model
                best_params[0]=c_a
                best_params[1]=depth
            f1_list.append(f1)
            depth_list.append(depth)
            quantity_list.append(c_a)
    # DataFrame with the convergence analysis
    result = pd.DataFrame(f1_list)
    result.loc[:, 'tree_quantity'] = quantity_list
    result.loc[:, 'depths'] = depth_list
    result.columns = ['f1_score', 'tree_quantity', 'depths']
    result.sort_values(by=['tree_quantity'], ascending=True, inplace=True)

    # Plot
    pivot_table = result.pivot(index='tree_quantity', columns='depths', values='f1_score')
    # Create a heatmap using seaborn
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm')

    # Customize the axes and title of the graph
    plt.xlabel('Depth')
    plt.ylabel('NumTrees')
    plt.title('F1 score in function of parameters\n category ' + str(y.name))
    plt.savefig(path + 'convergence_' + str(y.name) + '.png')
    #plt.show()
    return result,best_params,best_model,best_y_pred

# Run the model with the parameters c_a (number of trees) and prof (maximum depth)
# Returns the importances of the top (max_features) features.
# Saves the importance plot at the specified (path)

def importances(model, X, y, y_pred, path):

    n_groups = len(y.unique())
    if n_groups > 2:
        roc_curve_multiclass(model, X, y, path)
    else:
        roc_curve_binary(y, y_pred, path)

    # Performance graphs of the model
    confusion_matrix_plot(y, y_pred, path)
    classification_report_plot(y, y_pred, path)

    model.fit(X, y)
    # Calculate feature importances
    feature_importance = pd.DataFrame({'col_name': X.columns, 'feature_importance_vals': model.feature_importances_})

    # Sort the DataFrame by feature importance values in descending order
    feature_importance.sort_values(by='feature_importance_vals', ascending=False, inplace=True)
    return feature_importance


# Generates and saves the classification report
def classification_report_plot(y_true, y_pred, path):
    report = classification_report(y_true, y_pred)

    # Configure the figure size if needed
    plt.figure(figsize=(8, 6))

    # Use plt.text to add the report to the plot
    plt.text(0.1, 0.5, report, fontsize=12)

    # Configure the axes
    plt.axis('off')

    # Save the plot as an image (e.g., PNG)
    plt.savefig(path + 'classification_report_' + str(y_true.name) + '.png', bbox_inches='tight')

    # Show the plot if you want to view it on the screen
    #plt.show()



# Generates and saves ROC curves for each class in a multiclass classification problem
def roc_curve_multiclass(model, X, y, path):
    # Create a one-vs-rest classifier
    classifier = OneVsRestClassifier(model)

    # Train the classifier
    classifier.fit(X, y)

    # Predict probabilities for each class
    y_score = classifier.predict_proba(X)
    label_binarizer = LabelBinarizer().fit(y)

    # Binarize labels to compute the multiclass ROC curve
    y_bin = label_binarizer.transform(y)
    classes = y.unique()

    for class_id in classes:
        RocCurveDisplay.from_predictions(
            y_bin[:, class_id],
            y_score[:, class_id],
            name=f"Group {class_id} vs the rest",
            color="darkorange"
        )
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"One-vs-Rest ROC curves:\n Group {class_id} vs the rest")
        plt.legend()
        plt.savefig(path + f'roc_curve_class_{class_id}_' + str(y.name) + '.png')
        #plt.show()

# Generates and saves a ROC curve for a binary classification problem
def roc_curve_binary(y, y_pred, path):
    # Calculate ROC-AUC score
    roc_score = roc_auc_score(y, y_pred)

    # False Positive Rate, True Positive Rate, and thresholds
    fprs, tprs, thresholds = roc_curve(y, y_pred)

    # Plot ROC-AUC curve
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle="-", c="k", alpha=0.2, label="ROC-AUC=0.5")
    plt.plot(fprs, tprs, color="orange", label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Fill the area under the ROC-AUC curve
    y_zeros = [0 for _ in tprs]
    plt.fill_between(fprs, y_zeros, tprs, color="orange", alpha=0.3, label="ROC-AUC")
    plt.text(0.6, 0.2, f'ROC-AUC Score: {roc_score:.2f}', fontsize=12, color='black')
    plt.legend()
    plt.savefig(path + 'roc_curve_' + str(y.name) + '.png')
    #plt.show()


# Generates and saves a normalized confusion matrix plot
def confusion_matrix_plot(y, y_pred, path):
    # Get the confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)

    # Normalize the confusion matrix
    normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

    # Create a matplotlib figure
    plt.figure(figsize=(8, 6))

    # Create a heatmap with seaborn
    sns.heatmap(normalized_conf_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=False)

    # Add labels and title
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")

    # Save the plot to a file
    plt.savefig(path + 'confusion_matrix_' + str(y.name) + '.png')
    #plt.show()


# Plot the feature importances and save the plot
def plot_importances(importances, category, method, path):
    # Create a horizontal bar plot
    variable_names = importances.loc[:, 'col_name']
    importances_vals = importances.loc[:, 'feature_importance_vals']

    plt.figure(figsize=(10, 6))  # Figure size
    bars = plt.barh(variable_names, importances_vals, color='dodgerblue')
    plt.xlabel('Importance')  # x-axis label
    plt.ylabel('Variable')  # y-axis label
    plt.title('Variable Importance for Category ' + category)  # Plot title
    plt.gca().invert_yaxis()  # Invert y-axis to show the most important variable on top

    # Add labels to the bars
    for bar, val in zip(bars, importances_vals):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', ha='left', va='center', fontsize=10, color='black')

    plt.savefig(path + 'importances_category_' + category + '_' + method + '.png', bbox_inches='tight')

    # Show the plot
    #plt.show()

# Format parameters as a string
def format_params(params):
    return ', '.join([f"{key}={val}" for key, val in params.items()])

# Plot the top 10 parameter combinations for the model
def plot_grid_search(grid_search, path, category):
    # Extract results from grid_search
    results = grid_search.cv_results_
    params = results['params']
    mean_f1_scores = results['mean_test_score']

    # Get indices of the top 10 combinations
    top_10_indices = np.argsort(mean_f1_scores)[-10:]

    # Get parameters and scores of the top 10 combinations
    top_10_params = [params[i] for i in top_10_indices]
    top_10_scores = [mean_f1_scores[i] for i in top_10_indices]

    # Create a plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(10), top_10_scores, 'o-', label='Average F1 Score')
    plt.xticks(range(10), [str(i) for i in range(1, 11)])  # Use numbers from 1 to 10
    for i, txt in enumerate(top_10_scores):
        plt.annotate(f'{txt:.3f}', (i, top_10_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Top 10 Parameter Combinations')
    plt.ylabel('Average F1 Score')
    plt.title(f'Top 10 Parameter Combinations for {category}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Adjust coordinates to place the legend slightly below the x-axis
    legend_text = [f"Combination {i+1}: {format_params(param)}" for i, param in enumerate(top_10_params)]
    legend_text = '\n'.join(legend_text)
    plt.figtext(0.1, -0.15, legend_text, wrap=True, horizontalalignment='left', verticalalignment='center', fontsize=10)
    plt.savefig(path + 'hyperparameters_' + str(category) + '_plot.png')
    #plt.show()

# Sequential Feature Selection
def pre_selection(X,y,n_folds,best_model):
    # XGBoost classifier
    xgb_classifier=best_model
    # F1-score
    scorer = make_scorer(f1_score,average= 'macro')

    # Cross validation
    stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=False)

    # Create a SequentialFeatureSelector object
    sfs = SequentialFeatureSelector(xgb_classifier, k_features=15, forward=False, floating=True, verbose=2, scoring=scorer, cv=stratified_kfold)

    # Apply SFS to the training set
    sfs = sfs.fit(X, y)

    # Display the selected features
    selected_features = list(X.columns[list(sfs.k_feature_idx_)])
    print("Selected features:", selected_features)

    # Train the model with the selected features
    xgb_classifier.fit(X[selected_features], y)

    # Make predictions with cross-validation
    y_pred =cross_val_predict(xgb_classifier,X[selected_features],y,cv=stratified_kfold)

    # Evaluate the model performance
    f1_metric = f1_score(y,y_pred)
    print("Model accuracy with selected features:", f1_metric)

    return selected_features

def grid(X,y,n_folds):
    # XGBoost Classifier
    model = xgb.XGBClassifier(objective='binary:logistic',random_state=13)


    # Hiperparámetros a ajustar
    param_grid = {
        'n_estimators': list(range(10,200,10)),  # Puedes ajustar estos valores
        'learning_rate':[0.001,0.05,0.1,0.7,0.9], # [0,1] valores bajos evitan el sobreajuste
        'gamma':[0,0.001,0.05,1,5,10], # [0, +∞) Regula las divisiones. Valor alto evita sobreajuste
        'max_depth':[0,4,5,6],# Profundidad del árbol. 0: no hay límite
        'lambda':[0.01,0.1,1,10,100] #regularización L2 en los pesos del modelo
    }

    # Evaluation metric: macro f1-score
    scorer = make_scorer(f1_score, average='macro')
    # Cross-validation
    stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=False)
    # Grid search for the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=stratified_kfold, scoring=scorer, n_jobs=-1, error_score='raise')
    grid_search.fit(X, y)
    # Print the best parameters found
    print("Best parameters found:", grid_search.best_params_)
    return grid_search.best_estimator_,grid_search

def display_bumpchart(importances_rf, importances_xgb, category,path):
    """
    Display a bumpchart comparing the variable importance rankings between Random Forest (RF) and XGBoost (XGB).

    Parameters:
    - importances_rf (DataFrame): DataFrame containing Random Forest variable importances.
    - importances_xgb (DataFrame): DataFrame containing XGBoost variable importances.
    - category (str): Category label for the instrumental ranking.

    Returns:
    None
    best
    """

    # Extract 'col_name' columns from each DataFrame
    columns_rf = importances_rf.loc[:, 'col_name']
    columns_xgb = importances_xgb.loc[:, 'col_name']

    # Create a dictionary to store variable rankings for RF and XGB
    dic = {}
    union = set(columns_rf).union(columns_xgb)
    variables = list(union)

    for var in variables:
        id_rf = columns_rf[columns_rf.values == var].index[0] if var in columns_rf.values else None
        id_xgb = columns_xgb[columns_xgb.values == var].index[0] if var in columns_xgb.values else None
        dic[var] = [id_rf, id_xgb]

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(dic, index=['rf', 'xgb'])

    # Plot the bumpchart
    plt.figure()
    bumpchart(df, show_rank_axis=False, scatter=True, holes=False)
    plt.title('Ranking of Relevant Variables\n'+str(category)+ ' category')
    # Save the figure
    plt.savefig(path + 'ranking_importance_rf_xgb_'+str(category)+'.png', bbox_inches='tight')

    # Display the plot
    #plt.show()

##################################################################
#                    Funciones parte 2                           #
##################################################################


#Este es el onehot encoder
def preprocess_data(data):
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    encoded_data = one_hot_encoder.fit_transform(data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(categorical_cols))

    # Drop the original categorical columns and concatenate the encoded columns
    data = data.drop(categorical_cols, axis=1)
    data = pd.concat([data, encoded_df], axis=1)
    return data


def replace_numbers_in_excel(file_path, sheet_name=0):
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # List of columns to be excluded from replacement
    excluded_columns = [
        'depression_total_score',
        'sociodemo_years_marital_status',
        'sociodemo_years_since_current_em',
        'frag_marcha_4_5',
        'moca_final5',
        'sociodemo_idp_anio_nacimiento'
    ]

    # Function to replace numbers
    def replace_number(num):
        if pd.isna(num):
            return num  # If the number is NaN, return it as is
        if num < 0:
            return 'X'
        elif 0 <= num <= 20:
            return chr(65 + int(num))  # Convert 0-20 to 'A'-'K'
        else:
            return 'X'  # Return number as is if > 10

    # Apply the replacement function to each column except the excluded ones
    for column in df.columns:
        if column not in excluded_columns:
            df[column] = df[column].apply(replace_number)

    return df

##################################################################
#                    Data Reading                                #
##################################################################
file_path = 'Datos/datos_originales.xlsx'
new_df = replace_numbers_in_excel(file_path)

method='quantile'
n_groups=2
# Read the data for independent variables
#data=pd.read_excel('C:\\Users\\MariaVelez\\OneDrive - Universidad EAFIT\\Práctica_Investigativa\\Resultados_sfs\\Datos\\Datos_limpios\\datos_originales.xlsx')
data=new_df
data=data.iloc[:,1:]

#Onehot enconding
data = preprocess_data(new_df)



# Read the data for the categories
categories=pd.DataFrame()
#path='C:\\Users\\MariaVelez\\OneDrive - Universidad EAFIT\\Práctica_Investigativa\\Resultados_sfs\\Datos\\Categorias_discretizadas\\'+str(n_groups)+'_grupos\\'
path='Categorias_discretizadas/'+str(n_groups)+'_grupos/'
if method=='quantile':
  categories.loc[:,'tadlqi_instrumental']=pd.read_excel(path+'instrumental_discreto_quantile.xlsx')
  categories.loc[:,'tadlqi_advanced']=pd.read_excel(path+'advanced_discreto_quantile.xlsx')
  categories.loc[:,'tadlqi_total_adv_inst']=pd.read_excel(path+'total_discreto_quantile.xlsx')
elif method=='kmeans':
  categories.loc[:,'tadlqi_instrumental']=pd.read_excel(path+'instrumental_discreto_kmeans.xlsx')
  categories.loc[:,'advanced']=pd.read_excel(path+'advanced_discreto_kmeans.xlsx')
  categories.loc[:,'tadlqi_total_adv_inst']=pd.read_excel(path+'total_discreto_kmeans.xlsx')
elif method=='naturalbreaks':
  categories.loc[:,'tadlqi_instrumental']=pd.read_excel(path+'instrumental_discreto_naturalbreaks.xlsx')
  categories.loc[:,'advanced']=pd.read_excel(path+'advanced_discreto_naturalbreaks.xlsx')
  categories.loc[:,'tadlqi_total_adv_inst']=pd.read_excel(path+'total_discreto_naturalbreaks.xlsx')
# Sort the categories
id_instr=categories.sort_values(by=['tadlqi_instrumental'],ascending=True).index
id_adv=categories.sort_values(by=['tadlqi_advanced'],ascending=True).index
id_total=categories.sort_values(by=['tadlqi_total_adv_inst'],ascending=True).index


##################################################################
#                    Data Procesing                              #
##################################################################


# Variable that controls whether sequential feature selection is performed
# or if classification reports are generated for the model without pre-selecting features
bool_selected_features = True


# If sequential feature selection is not performed
if bool_selected_features == False:
    # CROSS VALIDATION
    y=categories.loc[id_adv,'tadlqi_instrumental']
    X=data.loc[id_adv,:]
    # Set the path for storing results

    #path = 'C:\\Users\\MariaVelez\\OneDrive - Universidad EAFIT\\Práctica_Investigativa\\Resultados_sfs\\Resultados_RF_individual\\' + str(n_groups) + 'grupos\\reales_' + str(method) + '\\'
    path = 'Resultados_RF_individual/' + str(n_groups) + 'grupos/reales_' + str(method) + '/'


    result,best_params,best_model,best_y_pred=convergence(X,y,path,r_s=13,num_folds=10)

    # Compute importances and generate plots
    importances_instrumental = importances(best_model, X, y, best_y_pred, path)
    plot_importances(importances_instrumental, category='tadlqi_instrumental', method=method, path=path)

    # Sort importances
    importances_instrumental.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

    # Save importances

    #path = 'C:\\Users\\MariaVelez\\OneDrive - Universidad EAFIT\\Práctica_Investigativa\\Resultados_sfs\\Resultados_RF_individual\\importancias\\' + str(n_groups) + 'grupos\\' + str(method) + '\\'
    path = 'Resultados_RF_individual/importancias/' + str(n_groups) + 'grupos/' + str(method) + '/'

    importances_instrumental.to_excel(path + 'importancias_instrumental_RF_Ind.xlsx', index=False)

# If sequential feature selection is performed
else:
    # Set the path for storing results
    path = 'Resultados_RF_individual_sfs/' + str(n_groups) + 'grupos/reales_' + str(method) + '/'

    # CROSS VALIDATION
    y=categories.loc[id_adv,'tadlqi_instrumental']
    X=data.loc[id_adv,:]
    result,best_params,best_model,best_y_pred=convergence(X,y,path,r_s=13,num_folds=10)
    # Perform feature pre-selection
    selected_features = pre_selection(X, y, n_folds=10, best_model=best_model)
    X = X.loc[id_adv, selected_features]

    # Perform grid search for the best estimator
    best_estimator, grid_search = grid(X, y, n_folds=10)

    # Cross-validation predictions
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=False)
    y_pred = cross_val_predict(best_estimator, X, y, cv=stratified_kfold)

    # Plot grid search results
    plot_grid_search(grid_search, path, category='tadlqi_instrumental')

    # Compute importances and generate plots
    importances_instrumental = importances(best_estimator, X, y, y_pred, path)
    plot_importances(importances_instrumental, category='tadlqi_instrumental', method=method, path=path)

    # Sort importances
    importances_instrumental.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

    # Save importances


    #path = 'C:\\Users\\MariaVelez\\OneDrive - Universidad EAFIT\\Práctica_Investigativa\\Resultados_sfs\\Resultados_RF_individual_sfs\\importancias\\' + str(n_groups) + 'grupos\\' + str(method) + '\\'
    path = 'Resultados_RF_individual_sfs/importancias/' + str(n_groups) + 'grupos/' + str(method) + '/'

    importances_instrumental.to_excel(path + 'importancias_instrumental_RF_Ind.xlsx', index=False)
