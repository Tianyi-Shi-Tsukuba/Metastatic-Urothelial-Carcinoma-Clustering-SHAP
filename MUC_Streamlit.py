import streamlit as st
import pandas as pd
import numpy as np
import itertools
import xgboost
import shap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from bioinfokit.analys import norm
from rpy2 import robjects
from rpy2.robjects import r,pandas2ri
from rpy2.robjects.packages import importr

   
###################  函数部分 ################### 
def SC_nSNN(A, k, sigma):
    data_size = A.shape[0]
    B = np.zeros((data_size, data_size)) 

    for i in range(data_size):
        for j in range(data_size):
            B[i, j] = np.exp(-np.sum((A[i, :] - A[j, :]) ** 2) / (2 * sigma ** 2))
            B[j, i] = B[i, j]

    temp = np.array([sorted(row, reverse=True) for row in B]) 
    I = np.argsort(-B, axis=1)  

    for i in range(k, data_size):
        temp[:, i] = 0

    E = np.zeros((data_size, data_size))
    for i in range(data_size):
        for j in range(k):
            E[i, I[i, j]] = temp[i, j]

    E[np.where(E != 0)] = 1 
    G = np.copy(E)

    W = np.zeros((data_size, data_size)) 

    for i in range(data_size):
        for j in range(i + 1, data_size):
            diff = np.sum(np.abs(G[i, :] - G[j, :])) / 2
            W[i, j] = k - diff
            if G[i, j] != 0 and G[j, i] != 0:
                W[i, j] += 1
            W[i, j] /= k
            W[j, i] = W[i, j]

    return W

def spectral_clustering(similarity_matrix, num_clusters):
    degrees = np.sum(similarity_matrix, axis=1)
    sqrt_degrees = np.sqrt(degrees)
    normalized_laplacian = np.diag(1.0 / sqrt_degrees) @ (np.diag(degrees) - similarity_matrix) @ np.diag(1.0 / sqrt_degrees)

    eigenvalues, eigenvectors = np.linalg.eig(normalized_laplacian)
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices[:num_clusters]]

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(np.real(sorted_eigenvectors))

    return clusters

def map_cluster_to_color(cluster):
    if cluster == 0:
        return "red"
    elif cluster == 1:
        return "blue"
    elif cluster == 2:
        return "purple"
    elif cluster == 3:
        return "green"
    elif cluster == 4:
        return "orange"
    elif cluster == 5:
        return "cyan"
    elif cluster == 6:
        return "pink"
    elif cluster == 7:
        return "brown"
    elif cluster == 8:
        return "yellow"
    elif cluster == 9:
        return "teal"
    elif cluster == 10:
        return "lime"
    else:
        return "gray" 

def display_shap_values(X, shap_values):
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    st.write(shap_df.head(20))

def train_xgboost_model(X, y, subsample, alpha, eta):
    model = xgboost.XGBClassifier(subsample=subsample, alpha=alpha, eta=eta)
    model.fit(X, y)
    return model

def kfold_cross_validation(X, y, model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_scores = []
    test_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        train_scores.append(train_score)
        test_scores.append(test_score)

    train_acc = np.mean(train_scores)
    test_acc = np.mean(test_scores)
    
    st.write("5折交叉验证 Train ACC：", train_acc)
    st.write("5折交叉验证 Test ACC：", test_acc)

def kfold_cross_validation_best(X, y):
    xgb_model = xgboost.XGBClassifier()
    param_combinations = list(itertools.product([0.6,0.7,1], [0.8,1.2,1.5], [0.01,0.05]))
    best_total_accuracy = 0
    best_params = None
  

    for subsample, alpha, eta in param_combinations:

        # 固定 random_state 为 42
        xgb_model.set_params(subsample=subsample, alpha=alpha, eta=eta)

        # Perform 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_scores = []
        test_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            xgb_model.fit(X_train, y_train)
            train_score = xgb_model.score(X_train, y_train)
            test_score = xgb_model.score(X_test, y_test)

            train_scores.append(train_score)
            test_scores.append(test_score)

        train_acc = np.mean(train_scores)
        test_acc = np.mean(test_scores)
        
        total_accuracy = train_acc + test_acc

        # 判断是否是最佳参数组合
        if total_accuracy > best_total_accuracy:
            best_total_accuracy = total_accuracy
            best_params = (subsample, alpha, eta)

    # Create a DataFrame for better table formatting
    st.write("Best Parameter Combination:")
    st.write("Subsample:", best_params[0])
    st.write("Alpha:", best_params[1])
    st.write("Eta:", best_params[2])
    st.write("5折交叉验证 Train ACC：", train_acc)
    st.write("5折交叉验证 Test ACC：", test_acc)
    return subsample, alpha, eta

def perform_shap_analysis(model, X, y):
    st.subheader("SHAP value")
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    display_shap_values(X,shap_values)
    shap_values_nr = shap_values.values[y == 0]
    shap_values_r = shap_values.values[y == 1]
    return shap_values.values, shap_values_nr, shap_values_r

def display_elbow_plot(pca_components):
    num_components = len(pca_components[0])
    std_deviation = np.std(pca_components, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(range(1, num_components + 1), std_deviation, color='black', marker='o')
    ax.set_title('Elbow Plot - Standard Deviation of Principal Components')
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Standard Deviation')
    ax.set_xticks(np.arange(1, num_components + 1))

    st.pyplot(fig)

def filter_principal_components(pca_components, threshold):
    std_deviation = np.std(pca_components, axis=0)
    selected_components = np.where(std_deviation > threshold)[0]
    selected_pca_components = pca_components[:, selected_components]
    return selected_pca_components, selected_components

def perform_clustering(selected_pca_components, k_value, cluster_number, original_data):
    similarity_matrix = SC_nSNN(selected_pca_components, k_value, 1.5)
    clusters = spectral_clustering(similarity_matrix, cluster_number)

    # Filter original_data based on the indices of selected_pca_components
    selected_indices = np.where(np.std(selected_pca_components, axis=1) > 0.0)[0]
    filtered_original_data = original_data.iloc[selected_indices]

    # Create a DataFrame with 'id' values and corresponding clusters
    result_df = pd.DataFrame({'id': filtered_original_data['id'], 'cluster': clusters, 'label': filtered_original_data['label']})

    # Map clusters to colors
    cluster_colors = [map_cluster_to_color(cluster) for cluster in clusters]
    result_df['color'] = cluster_colors
    
    # Perform PCA on selected_pca_components and add the first component to result_df
    num_components = 1
    pca = PCA(n_components=num_components)
    pca.fit(selected_pca_components)
    pca_1 = pca.transform(selected_pca_components)[:, 0]
    result_df['pca_1'] = pca_1

    sorted_result_df = result_df.sort_values(by=['cluster', 'label', 'pca_1'])

    return cluster_colors, sorted_result_df

def create_scatter_plot(pca_components, clusters, selected_components):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    x_index, y_index = selected_components
    for i in range(len(pca_components)):
        ax.scatter(pca_components[:, x_index][i], pca_components[:, y_index][i], c=clusters[i])
        ax.annotate(str(i), xy=(pca_components[:, x_index][i], pca_components[:, y_index][i]),
                    xytext=(pca_components[:, x_index][i], pca_components[:, y_index][i]), fontsize=10)

    title = f'PC{selected_components[0]+1} vs PC{selected_components[1]+1}'
    ax.set_title(title, fontsize=14)

    st.pyplot(fig)
    

def plot_clusters(pca_components, clusters, selected_components_to_show):
    st.subheader('Visualization')
    if selected_components_to_show == ('PC1', 'PC2'):
        create_scatter_plot(pca_components, clusters, (0, 1))
    elif selected_components_to_show == ('PC1', 'PC3'):
        create_scatter_plot(pca_components, clusters, (0, 2))
    else:  # ('PC2', 'PC3')
        create_scatter_plot(pca_components, clusters, (1, 2))

def decision_tree_analysis(X, clusters, selected_clusters):
    st.subheader("Decision Tree")

    selected_indices = [i for i, cluster in enumerate(clusters) if cluster in selected_clusters]
    selected_X = X.iloc[selected_indices, :]
    selected_clusters = [clusters[i] for i in selected_indices]

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(selected_X, selected_clusters)

    feature_importance_analysis(X, clf)
    
    return clf

def feature_importance_analysis(X, clf):
    feature_importances = clf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df['Importance'] = feature_importance_df['Importance'].apply(lambda x: f'{x:.4f}')
    sorted_importance = feature_importance_df.sort_values(by='Importance', ascending=False)

    st.subheader('Feature Importance:')
    st.dataframe(sorted_importance.head(10))

    pruned_tree_visualization(X, clf)

def pruned_tree_visualization(X, clf):
    st.subheader('Decision Tree Visualization')
    #length_number = st.slider("レイヤー数を選択してください", min_value=1, max_value=4, value=2)
    pruned_tree = DecisionTreeClassifier(max_depth=2)
    pruned_tree.fit(X, clf.predict(X))

    fig, ax = plt.subplots(figsize=(24, 16), dpi=300)
    plot_tree(pruned_tree, feature_names=X.columns, class_names=True, filled=True, fontsize=26, ax=ax)
    st.pyplot(fig)

def create_heatmap_streamlit(X, result_df, selected_features, sigmoid_size, num_display_features,x_label_font_size,y_label_font_size,cmap_choice):
    # Extract relevant information from result_df
    cluster_colors = result_df['color'].values
    cluster_labels = result_df['cluster'].values
    sample_order = result_df.sort_values(by=['cluster', 'label', 'pca_1']).index

    # Filter X based on selected features
    X_selected = X[selected_features].iloc[sample_order]
    
    sigmoid = sigmoid_size
    selected_X_return = 10**(X_selected + np.log10(0.00001)) - 0.00001
    selected_X_original  = (1 / (1 + np.exp(-(selected_X_return  * sigmoid))) - 0.5) * 2
    selected_X_original = np.log10(selected_X_original + 0.001) - np.log10(0.001)

    # Select the top num_display_features features
    selected_X_original = selected_X_original.iloc[:, :num_display_features]

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    heatmap = sns.heatmap(selected_X_original.T, cmap=cmap_choice, yticklabels=True, xticklabels=selected_X_original.index, cbar_kws={'label': 'Feature Value'}, ax=ax)

    # Set axis labels and title
    ax.set_xlabel('Sample')
    ax.set_ylabel('Feature')

    # Adjust font size for y-axis labels
    ytick_labels = ax.get_yticklabels()
    for label in ytick_labels:
        label.set_fontsize(y_label_font_size)

    # Adjust font size for x-axis labels
    xtick_labels = ax.get_xmajorticklabels()
    for label in xtick_labels:
        label.set_fontsize(x_label_font_size)

    # Show the plot
    st.pyplot(fig)

@st.cache_resource 
def DEG_analysis(cnt,metadata):
    edgeR = importr('edgeR')
    pandas2ri.activate()
    
    st.markdown("<h4 style='text-align: left; color: black;'></h4>", unsafe_allow_html=True)
    
    nm = norm()
    nm.tpm(df=cnt, gl='Length')
    tpm = nm.tpm_norm
    tpm = tpm[~tpm.index.duplicated()]
    tpm = tpm.dropna()
    tpm = tpm[tpm.mean(axis=1) >= 1]
    
    sam_1 = metadata[metadata["label"] == 1]["id"].tolist()
    sam_0 = metadata[metadata["label"] == 0]["id"].tolist()
    sam_all = sam_1 + sam_0
    tpm = tpm[sam_all]
    
    group = robjects.IntVector([1] * len(sam_1) + [0] * len(sam_0))
    y1 = edgeR.DGEList(counts=pandas2ri.DataFrame(tpm), group=group)
    y2 = edgeR.calcNormFactors(y1)
    y3 = edgeR.estimateCommonDisp(y2)
    y = edgeR.estimateTagwiseDisp(y3)
    et = edgeR.exactTest(y, pair=["1", "0"])
    
    result = edgeR.topTags(et,n=100000)
    alldiff = result.rx2("table")
    
    normlized_data = y.rx2('pseudo.counts')
    tpm_tmm = pd.DataFrame(normlized_data, index=tpm.index, columns=tpm.columns)
    
    return tpm, tpm_tmm, alldiff

def get_deg_data(tpm_tmm, alldiff, logFCcutoff_Thresh, FDR_Thresh, metadata):
        
    diff = alldiff.query('(FDR < @FDR_Thresh) & ((logFC > @logFCcutoff_Thresh) | (logFC < -@logFCcutoff_Thresh))')
    
    tpm_tmm_diff = tpm_tmm.loc[diff.index].T
    tpm_tmm_diff.reset_index(inplace=True)
    tpm_tmm_diff = tpm_tmm_diff.rename(columns={"index": "id"})
    merged_data = pd.merge(tpm_tmm_diff, metadata[['id', 'label']], on='id')
    tpm_tmm_diff['label'] = merged_data['label']
    
    st.write(f"{tpm_tmm_diff.shape[1]-2}個Geneが選ばれた。")
    
    data = {'logFC': alldiff['logFC'], 'logCPM': alldiff['logCPM'], 'PValue': alldiff['PValue'],'FDR': alldiff['FDR']}
    volcano_data = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.scatter(volcano_data['logFC'], -np.log10(volcano_data['FDR']), color='blue', alpha=0.7, label='All Genes')
    ax.scatter(diff['logFC'], -np.log10(diff['FDR']), color='red', alpha=0.7, label='Differentially Expressed Genes')

    for i, row in diff.iterrows():
        ax.annotate(i, (row['logFC'], -np.log10(row['FDR'])),
                    textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8, color='black')
    ax.axhline(y=-np.log10(FDR_Thresh), color='red', linestyle='--', label=f'FDR Threshold({FDR_Thresh})')
    ax.axvline(x=logFCcutoff_Thresh, color='green', linestyle='--', label=f'logFC Threshold ({logFCcutoff_Thresh})')
    ax.axvline(x=-logFCcutoff_Thresh, color='green', linestyle='--')

    ax.set_title(f'Number of DEGs: {len(diff)}')
    ax.set_xlabel('log2 Fold Change')
    ax.set_ylabel('-log10(FDR)')
    ax.legend()
    st.pyplot(fig)
    
    return tpm_tmm_diff

def xgboost_shap_analysis(X, y):
    ###################################################################################################
    # def train_xgboost_model
    subsample = st.slider("XGBoostのsubsample パラメーターの選択", 0.5, 1.0, 0.8)
    alpha = st.slider("XGBoostのalpha パラメーターの選択", 0.1, 1.5, 1.0)
    eta = st.slider("XGBoostのeta パラメーターの選択", 0.01, 0.3, 0.1)
    model = train_xgboost_model(X, y, subsample, alpha, eta)
    
    # 用户选择是否标准化的复选框
    should_best = st.checkbox("パラメーターを最適化する")

    # 根据用户选择展示数据
    if should_best:
        bestsubsample, bestalpha, besteta = kfold_cross_validation_best(X,y)
        model = train_xgboost_model(X, y, bestsubsample, bestalpha, besteta)
    else:
        # def kfold_cross_validation
        kfold_cross_validation(X, y, model)
        model = train_xgboost_model(X, y, subsample, alpha, eta)
     
    shap_values, shap_values_nr, shap_values_r = perform_shap_analysis(model, X, y)
    
    ###################################################################################################
    # def perform_shap_analysis
    st.write("---")  
    st.title("Step4: PCA + Clustering + Decision Tree + Heatmap")
    shap_values_choice = st.selectbox("サンプルを選択してください", ["All samples", "Label 0 samples", "Label 1 samples"])
    
    # def display_elbow_plot，filter_principal_components
    st.subheader("PCA")
    if shap_values_choice == "All samples":
        selected_shap_values = shap_values
    elif shap_values_choice == "Label 0 samples":
        selected_shap_values = shap_values_nr
    else:  # "shap_values_r"
        selected_shap_values = shap_values_r
        
    num_components = 20
    pca = PCA(n_components=num_components)
    pca.fit(selected_shap_values)
    pca_components = pca.transform(selected_shap_values)
    display_elbow_plot(pca_components)

    threshold = st.slider('閾値を選択してください', min_value=0.0, max_value=0.5, step=0.05,key = 'all')
    selected_pca_components, selected_components = filter_principal_components(pca_components, threshold)
    st.write(f"選択された閾値は {threshold} で、その閾値より大きい主成分の数は：{len(selected_components)}")

    # def perform_clustering，plot_clusters
    st.subheader("Clustering(Method: SC_nSNN)")
    k_value = st.slider("Neighbors数を選択してください", min_value=2, max_value=len(selected_shap_values), value=30)
    cluster_number = st.slider("クラスター数を選択してください", min_value=2, max_value=10, value=4)
    clusters, result_df = perform_clustering(selected_pca_components, k_value, cluster_number, df_cnt_degs_norm)
    
    st.subheader("Cluster Assignments")
    st.dataframe(result_df)
    
    selected_components_to_show = st.selectbox('主成分を選択してください', [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC2', 'PC3')],key = '1')
    plot_clusters(selected_pca_components, clusters, selected_components_to_show)

    # def perform_clustering，plot_clusters
    selected_clusters = st.multiselect('クラスターを選択してください:', list(set(clusters)), default=list(set(clusters)),key = '111')
    clf = decision_tree_analysis(X, clusters, selected_clusters)
    selected_features = X.columns[np.argsort(clf.feature_importances_)[::-1][:5]]
    
    # def create_heatmap_streamlit
    st.subheader('Heatmap')
    sigmoid_size = st.slider("Sigmoidのパラメーターを選択してください", 1/50, 10.0, 1.0)  # Allow user to choose sigmoid size
    num_display_features = st.slider("特徴数を選択してください", 1, 10, 3)  # Allow user to choose the number of features to display
    x_label_font_size = st.slider("ラベルFont Sizeを選択してください", 1, 20, 2)
    y_label_font_size = st.slider("特徴Font Sizeを選択してください", 1, 20, 8)
    cmap_choice = st.selectbox("ColorMapを選択してください", ["coolwarm", "viridis", "plasma", "inferno", "magma"],key = '11')
    create_heatmap_streamlit(X, result_df, selected_features, sigmoid_size, num_display_features,x_label_font_size,y_label_font_size,cmap_choice)

    ###################################################################################################
    st.write("---")  
    st.title("Step5: SubClustering + Decision Tree + Heatmap")
    st.subheader("Cluster Assignments")
    st.dataframe(result_df)
    selected_color = st.selectbox("分析したいsubclusterの色を選択してください", result_df['color'].unique())
    filtered_result_df = result_df[result_df['color'] == selected_color]
    st.subheader("Subcluster情報")
    st.dataframe(filtered_result_df)  
    selected_rows_shap_values = shap_values[filtered_result_df.index, :]
    
    st.subheader("PCA")
    num_components_sub = 20
    pca_sub = PCA(n_components=num_components_sub)
    pca_sub.fit(selected_rows_shap_values)
    pca_components_sub = pca_sub.transform(selected_rows_shap_values)
    display_elbow_plot(pca_components_sub)
    
    threshold_sub = st.slider('閾値を選択してください', min_value=0.0, max_value=0.5, step=0.05,key = 'allsub')
    selected_pca_components_sub, selected_components_sub = filter_principal_components(pca_components_sub, threshold_sub)
    st.write(f"選択された閾値は {threshold_sub} で、その閾値より大きい主成分の数は：{len(selected_components_sub)}")
    
    st.subheader("Clustering(Method: SC_nSNN)")
    k_value_sub = st.slider("Neighbors数を選択してください", min_value=2, max_value=len(selected_rows_shap_values), value=10)
    cluster_number_sub = st.slider("クラスター数を選択してください", min_value=2, max_value=10, value=2)
    clusters_sub, result_df_sub = perform_clustering(selected_pca_components_sub, k_value_sub, cluster_number_sub, df_cnt_degs_norm)
    
    st.subheader("Cluster Assignments of subcluster")
    st.dataframe(result_df_sub)
    
    selected_components_to_show_sub = st.selectbox('主成分を選択してください', [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC2', 'PC3')],key = '2')
    plot_clusters(selected_pca_components_sub, clusters_sub, selected_components_to_show_sub)
    
    selected_clusters_sub = st.multiselect('クラスターを選択してください:', list(set(clusters_sub)), default=list(set(clusters_sub)),key = '222')
    clf_sub = decision_tree_analysis(X, clusters_sub, selected_clusters_sub)
    selected_features_sub = X.columns[np.argsort(clf_sub.feature_importances_)[::-1][:5]]
    
    st.subheader('Heatmap')
    sigmoid_size_sub = st.slider("Sigmoidのパラメーターを選択してください", 1/50, 10.0, 1.5)  # Allow user to choose sigmoid size
    num_display_features_sub = st.slider("特徴数を選択してください", 1, 10, 2)  # Allow user to choose the number of features to display
    x_label_font_size_sub = st.slider("ラベルFont Sizeを選択してください", 1, 20, 4)
    y_label_font_size_sub = st.slider("特徴Font Sizeを選択してください", 1, 20, 9)
    cmap_choice_sub = st.selectbox("ColorMapを選択してください", ["coolwarm", "viridis", "plasma", "inferno", "magma"],key = '22')
    create_heatmap_streamlit(X, result_df_sub, selected_features_sub, sigmoid_size_sub, num_display_features_sub,x_label_font_size_sub,y_label_font_size_sub,cmap_choice_sub)
    
    
###################  网页部分  ################### 
st.set_page_config(
    page_title="Sakura",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.google.com/',
        'Report a bug': "https://www.google.com/",
        'About': "まだ開発中"
    }
)
st.markdown("<h1 style='text-align: center; color: black;'>MUC project</h1>", unsafe_allow_html=True)
#image_path = "/Users/shi/Documents/MUC/ss.png" 
#st.image(image_path, use_column_width=True)
data_format_table_cnt = pd.DataFrame({
    '': ['Gene 1', 'Gene 2', 'Gene 3', '...', 'Gene N'],
    'Length': ['5423', '8925', '94', '...', '804'],
    'sample 1': ['231', '15', '36', '...', '1578'],
    'sample 2': ['967', '25', '4531', '...', '5'],
    'sample 3': ['0', '825', '21', '...', '644'],
    '...': ['...', '...', '...', '...', '...'],
    'sample N': ['312', '0', '321', '...', '2'],
})
data_format_table_meta = pd.DataFrame({
    'id': ['sample 1', 'sample 2', 'sample 3', '...', 'sample N'],
    'label': ['0', '1', '1', '...', '0'],
})

st.markdown("<h3 style='text-align: left; color: black;'>Expected Data Format:</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Raw count data:</h3>", unsafe_allow_html=True)
st.table(data_format_table_cnt)
st.markdown("<h6 style='text-align: left; color: black;'>Data should be .csv of gene raw count data, and the second column should be length of every gene. </h8>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Meta data:</h3>", unsafe_allow_html=True)
st.table(data_format_table_meta)
st.markdown("<h6 style='text-align: left; color: black;'>Data should be .csv of Meta data. </h8>", unsafe_allow_html=True)
st.write("---")  
st.title("Step1: Data upload")
uploaded_file = st.file_uploader("Count dataをアップロードしてください", type="csv")
uploaded_file_meta = st.file_uploader("Meta dataをアップロードしてください", type="csv")
st.write("---")  

if uploaded_file is not None and uploaded_file_meta is not None:
    st.title("Step2: DEG analysisと前処理")
    cnt = pd.read_csv(uploaded_file, header=0, index_col=0)
    metadata = pd.read_csv(uploaded_file_meta, header=0)
  
    tpm, tpm_tmm, alldiff = DEG_analysis(cnt, metadata)
    
    st.subheader("Countデータ(一部)")
    st.write(cnt.head(20))
    st.subheader("TPMデータ(一部)")
    st.write(tpm.head(20))
    st.subheader("TransformしたTPMデータ(一部)")
    st.write(tpm_tmm.head(20))
    st.subheader("選ばれたTop ten genes")
    st.write(alldiff.head(10))
    
    st.subheader("Volcano Plot")
    logFCcutoff_Thresh = st.slider("logFC Thresholdを選択してください", 1, 10, 2)
    FDR_Thresh = st.slider("FDR Thresholdを選択してください", 0.01, 0.3, 0.05)
    

    df_cnt_degs_norm = get_deg_data(tpm_tmm, alldiff, logFCcutoff_Thresh, FDR_Thresh, metadata)
    
    st.subheader("DEGデータ")
    st.write(df_cnt_degs_norm)

    # 用户选择是否标准化的复选框
    should_normalize = st.checkbox("データを標準化する")
   
    # 根据用户选择展示数据
    if should_normalize: 
        # 参数选择
        eps_options = [0.00001, 0.000001, 0.0000001, 0.00000001]
        eps = st.selectbox('標準化のパラメーターを選択してください', options=eps_options)

        # 対数変換
        X_ = df_cnt_degs_norm.drop(['id', 'label'], axis=1)
        y = df_cnt_degs_norm['label']
        z = df_cnt_degs_norm['id']
        X_log = np.log10(X_ + eps) - np.log10(eps)
        X = pd.DataFrame(X_log, index=X_.index, columns=X_.columns)

        # 标准化的数据
        st.subheader("標準化後のDEGデータ")
        st.write(X.head(20))
    else:
        # 用户选择不标准化时的操作
        X = df_cnt_degs_norm.drop(['id', 'label'], axis=1)
        y = df_cnt_degs_norm['label']
        z = df_cnt_degs_norm['id']
        
        # 显示未标准化的数据
        st.subheader("標準化しないDEGデータ")
        st.write(X.head(20))
    
    st.write("---")  
    st.title("Step3: XGBoost + SHAP value")
    st.subheader("XGBoost")
    xgboost_shap_analysis(X, y)


