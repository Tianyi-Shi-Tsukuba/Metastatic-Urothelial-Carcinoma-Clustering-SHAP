import streamlit as st
import pandas as pd
import numpy as np
import itertools
import xgboost
import shap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

   
###################  Function ################### 
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

def display_shap_values(X, shap_values, sample_ids):
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    shap_df.insert(0, 'id', sample_ids)
    st.write(shap_df)


def train_xgboost_model(X, y, subsample, alpha, eta):
    model = xgboost.XGBClassifier(subsample=subsample, alpha=alpha, eta=eta)
    model.fit(X, y)
    return model

def single_split_evaluation(X, y, model, test_size=0.2):
    seeds = [0, 1, 2, 3, 4]
    acc_train_list = []
    acc_test_list = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        model.fit(X_train, y_train)
        acc_train = model.score(X_train, y_train)
        acc_test = model.score(X_test, y_test)

        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)

    st.write(f"Train ACC (avg of 5 seeds): {np.mean(acc_train_list):.4f}")
    st.write(f"Test ACC (avg of 5 seeds): {np.mean(acc_test_list):.4f}")


def perform_shap_analysis(model, X, y, z):
    st.subheader("SHAP values")
    st.markdown("<h6 style='text-align: left; color: black;'>*You can download the SHAP values for your own clustering and downstream analysis. </h8>", unsafe_allow_html=True)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    display_shap_values(X, shap_values, z.values)
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

    selected_indices = np.where(np.std(selected_pca_components, axis=1) > 0.0)[0]
    filtered_original_data = original_data.iloc[selected_indices]

    result_df = pd.DataFrame({'id': filtered_original_data['id'], 'cluster': clusters, 'label': filtered_original_data['label']})

    cluster_colors = [map_cluster_to_color(cluster) for cluster in clusters]
    result_df['color'] = cluster_colors
    
    num_components = 1
    pca = PCA(n_components=num_components)
    pca.fit(selected_pca_components)
    pca_1 = pca.transform(selected_pca_components)[:, 0]
    result_df['pca_1'] = pca_1

    sorted_result_df = result_df.sort_values(by=['cluster', 'label', 'pca_1'])

    return cluster_colors, sorted_result_df

def create_scatter_plot(pca_components, clusters, selected_components, sample_ids):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    x_index, y_index = selected_components

    unique_clusters = np.unique(clusters)

    for cluster in unique_clusters:
        mask = clusters == cluster
        ax.scatter(
            pca_components[mask, x_index],
            pca_components[mask, y_index],
            label=f'Cluster {cluster}',
            s=50
        )
        # 添加每个点的前5位 ID
        for i in np.where(mask)[0]:
            short_id = str(sample_ids[i])[:5]
            ax.annotate(short_id,
                        xy=(pca_components[i, x_index], pca_components[i, y_index]),
                        xytext=(pca_components[i, x_index], pca_components[i, y_index]),
                        fontsize=10)

    title = f'PC{selected_components[0]+1} vs PC{selected_components[1]+1}'
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f'PC{selected_components[0]+1}')
    ax.set_ylabel(f'PC{selected_components[1]+1}')
    ax.legend(loc='upper right', title='Cluster')

    st.pyplot(fig)



def plot_clusters(pca_components, clusters, selected_components_to_show, sample_ids):
    st.subheader('Visualization of clusters')
    # 传入 sample_ids
    if selected_components_to_show == ('PC1', 'PC2'):
        create_scatter_plot(pca_components, clusters, (0, 1), sample_ids)
    elif selected_components_to_show == ('PC1', 'PC3'):
        create_scatter_plot(pca_components, clusters, (0, 2), sample_ids)
    else:  # ('PC2', 'PC3')
        create_scatter_plot(pca_components, clusters, (1, 2), sample_ids)
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

def plot_cluster_label_proportions(result_df):
    st.subheader("Count of Each Label in Each Cluster")

    cluster_label_counts = result_df.groupby(['cluster', 'label']).size().unstack(fill_value=0)
    cluster_label_props = cluster_label_counts.div(cluster_label_counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_label_counts.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'lightsalmon'])

    for i, cluster in enumerate(cluster_label_counts.index):
        bottom = 0
        for j, label in enumerate(cluster_label_counts.columns):
            count = cluster_label_counts.loc[cluster, label]
            prop = cluster_label_props.loc[cluster, label]
            ax.text(i, bottom + count / 2, f"{prop * 100:.1f}%", ha='center', va='center', fontsize=10)
            bottom += count

    ax.set_title('Count of Responder and Non-responder in Each Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')
    ax.legend(title='Response', labels=['Non-responder', 'Responder'])
    st.pyplot(fig)


def pruned_tree_visualization(X, clf):
    st.subheader('Decision Tree Visualization')
    pruned_tree = DecisionTreeClassifier(max_depth=2)
    pruned_tree.fit(X, clf.predict(X))

    fig, ax = plt.subplots(figsize=(24, 16), dpi=300)
    plot_tree(pruned_tree, feature_names=X.columns, class_names=True, filled=True, fontsize=26, ax=ax)
    st.pyplot(fig)

def create_heatmap_streamlit(X, result_df, selected_features, num_display_features, x_label_font_size, y_label_font_size, cmap_choice):
    sample_order = result_df.index
    X_selected = X[selected_features].iloc[sample_order]
    X_zscore = (X_selected - X_selected.mean()) / X_selected.std()
    X_zscore = X_zscore.iloc[:, :num_display_features]

    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    sns.heatmap(
        X_zscore.T,
        cmap=cmap_choice,
        yticklabels=True,
        xticklabels=X_zscore.index,
        cbar_kws={'label': 'Z-score'},
        ax=ax
    )

    ordered_clusters = result_df['cluster'].values
    unique_clusters, counts = np.unique(ordered_clusters, return_counts=True)
    boundaries = np.cumsum(counts)
    for b in boundaries[:-1]:
        ax.axvline(b, color='black', linewidth=3)

    ax.set_xlabel('Sample')
    ax.set_ylabel('Feature')
    for label in ax.get_yticklabels():
        label.set_fontsize(y_label_font_size)
    for label in ax.get_xmajorticklabels():
        label.set_fontsize(x_label_font_size)

    st.pyplot(fig)


@st.cache_resource(show_spinner=True)
def single_split_cv_best_cached(X, y):
    return single_split_cv_best(X, y)

def single_split_cv_best(X, y, test_size=0.2):
    xgb_model = xgboost.XGBClassifier()
    param_combinations = list(itertools.product(
        [0.6, 0.7, 0.8 ,0.9, 1],
        [0, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    ))

    best_total_metric = -1
    best_result = {}
    seeds = [0, 1, 2, 3, 4]

    for subsample, alpha, eta in param_combinations:
        acc_train_list, acc_test_list = [], []

        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )

            model = xgboost.XGBClassifier(subsample=subsample, alpha=alpha, eta=eta, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)

            acc_train = model.score(X_train, y_train)
            acc_test = model.score(X_test, y_test)

            acc_train_list.append(acc_train)
            acc_test_list.append(acc_test)

        mean_acc_train = np.mean(acc_train_list)
        mean_acc_test = np.mean(acc_test_list)
        total_metric = mean_acc_train + mean_acc_test

        if total_metric > best_total_metric:
            best_total_metric = total_metric
            best_result = {
                'Subsample': subsample,
                'Alpha': alpha,
                'Eta': eta,
                'Train ACC (avg)': round(mean_acc_train, 4),
                'Test ACC (avg)': round(mean_acc_test, 4)
            }

    st.subheader("Best Parameter Combination (Average of 5 Splits)")
    st.dataframe(pd.DataFrame([best_result]))

    return best_result['Subsample'], best_result['Alpha'], best_result['Eta']


def xgboost_shap_analysis(X, y):
    subsample = st.slider("XGBoost subsample parameter", 0.5, 1.0, 1.0, step=0.1)
    alpha = st.slider("XGBoost alpha parameter", 0.0, 5.0, 0.0, step=0.1)
    eta = st.slider("XGBoost eta parameter", 0.01, 0.3, 0.3, step=0.01)
    model = train_xgboost_model(X, y, subsample, alpha, eta)
    
    optimize_params = st.checkbox("Automatically optimize parameters (Grid Search) *This may take a long time")

    if optimize_params:
        bestsubsample, bestalpha, besteta = single_split_cv_best_cached(X, y)
        model = train_xgboost_model(X, y, bestsubsample, bestalpha, besteta)
    else:
        single_split_evaluation(X, y, model)
        model = train_xgboost_model(X, y, subsample, alpha, eta)
     
    shap_values, shap_values_nr, shap_values_r = perform_shap_analysis(model, X, y, z)

    st.write("---")  
    st.title("Step 4: PCA + Clustering + Decision Tree + Heatmap")
    shap_values_choice = st.selectbox("Select sample type for PCA", ["All samples", "Label 0 samples", "Label 1 samples"])
    
    st.subheader("PCA")
    if shap_values_choice == "All samples":
        selected_shap_values = shap_values
    elif shap_values_choice == "Label 0 samples":
        selected_shap_values = shap_values_nr
    else:
        selected_shap_values = shap_values_r
        
    num_components = 20
    pca = PCA(n_components=num_components)
    pca.fit(selected_shap_values)
    pca_components = pca.transform(selected_shap_values)
    display_elbow_plot(pca_components)

    threshold = st.slider("Threshold for PCA selection", 0.0, 0.5, 0.1, step=0.05, key='all')
    selected_pca_components, selected_components = filter_principal_components(pca_components, threshold)
    st.write(f"Selected threshold: {threshold}, number of components above threshold: {len(selected_components)}. These components will be used in clustering.")

    st.subheader("Clustering (Method: Spectral clustering)")
    k_value = st.slider("Number of Neighbors", 2, len(selected_shap_values), 30)
    cluster_number = st.slider("Number of Clusters", 2, 10, 4)
    cluster_colors, result_df = perform_clustering(selected_pca_components, k_value, cluster_number, df_cnt_degs_norm)
    sample_ids = result_df['id'].values
    clusters   = result_df['cluster'].values
    st.subheader("Cluster Assignments")
    st.dataframe(result_df)
    
    selected_components_to_show = st.selectbox("Choose Principal Components for Scatter Plot", [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC2', 'PC3')], key='1')
    plot_clusters(selected_pca_components, clusters, selected_components_to_show, sample_ids)
    plot_cluster_label_proportions(result_df)
    
    selected_clusters = st.multiselect("Select Clusters for Decision Tree", list(set(clusters)), default=list(set(clusters)), key='111')
    clf = decision_tree_analysis(X, clusters, selected_clusters)
    selected_features = X.columns[np.argsort(clf.feature_importances_)[::-1][:5]]
    
    st.subheader("Heatmap")
    num_display_features = st.slider("Number of features to display", 1, 50, 3)
    x_label_font_size = st.slider("X-axis label font size", 1, 20, 2)
    y_label_font_size = st.slider("Y-axis label font size", 1, 20, 8)
    cmap_choice = st.selectbox("Color map", ["coolwarm", "viridis", "plasma", "inferno", "magma"], key='11')
    create_heatmap_streamlit(X, result_df, selected_features, num_display_features, x_label_font_size, y_label_font_size, cmap_choice)

###################  Page  ################### 
st.set_page_config(
    page_title="Sakura · LSC project",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Tianyi-Shi-Tsukuba/Metastatic-Urothelial-Carcinoma-Clustering-SHAP',
        'Report a bug': "https://github.com/Tianyi-Shi-Tsukuba/Metastatic-Urothelial-Carcinoma-Clustering-SHAP",
        'About': "まだ開発中"
    }
)
st.markdown("<h1 style='text-align: center; color: black;'>Interpretable Feature Extraction and Spectral Clustering on gene expression data</h1>", unsafe_allow_html=True)
#image_path = "/Users/shi/Documents/MUC/ss.png" 
#st.image(image_path, use_column_width=True)
data_format_table_cnt = pd.DataFrame({
    '': ['Gene 1', 'Gene 2', 'Gene 3', '...', 'Gene N'],
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
st.markdown("<h3 style='text-align: center; color: black;'>Gene expression data (Count/TPM/FPKM/RPKM/CPM):</h3>", unsafe_allow_html=True)
st.table(data_format_table_cnt)
st.markdown("<h6 style='text-align: left; color: black;'>The input data should be a .csv file containing gene expression values, where rows represent gene names and columns represent sample names. To ensure efficient computation and meaningful analysis, it is recommended to input a filtered set of genes, such as those identified by differential expression analysis (DEGs). </h8>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Meta data:</h3>", unsafe_allow_html=True)
st.table(data_format_table_meta)
st.markdown("<h6 style='text-align: left; color: black;'>The input data should be .csv of Meta data. Notice: Please avoid uploading files that contain personal information (e.g., names, dates of birth, addresses, or any other personally identifiable information). </h8>", unsafe_allow_html=True)

st.write("---")  
st.title("Step1: Data upload")
uploaded_file = st.file_uploader("Please upload the gene expression data", type="csv")
uploaded_file_meta = st.file_uploader("Please upload the meta data", type="csv")
st.write("---")  

if uploaded_file is not None and uploaded_file_meta is not None:
    st.title("Step2: Preprocessing")
    expr = pd.read_csv(uploaded_file, header=0, index_col=0)
    metadata = pd.read_csv(uploaded_file_meta, header=0)

    expr = expr[~expr.index.duplicated()]
    expr = expr.dropna()
    expr = expr[expr.mean(axis=1) >= 1]

    sam_1 = metadata[metadata["label"] == 1]["id"].tolist()
    sam_0 = metadata[metadata["label"] == 0]["id"].tolist()
    sam_all = sam_1 + sam_0
    expr = expr[sam_all]

    expr_T = expr.T
    expr_T.reset_index(inplace=True)
    expr_T = expr_T.rename(columns={"index": "id"})
    merged_data = pd.merge(expr_T, metadata[['id', 'label']], on='id')

    df_cnt_degs_norm = merged_data

    st.subheader("Uploaded Gene Expression Data")
    st.write(df_cnt_degs_norm)

    st.subheader("Normalization Selection")
    normalize_option = st.radio("Do you want to normalize the data?", ("Yes (Normalize)", "No (Use raw data)"))

    if normalize_option:
        if normalize_option == "Yes (Normalize)":
            norm_method = st.selectbox("Select normalization method", ["log2(data+1)", "log2(data)", "log10(data+1)", "log10(data)"])

            X_ = df_cnt_degs_norm.drop(['id', 'label'], axis=1)
            y = df_cnt_degs_norm['label']
            z = df_cnt_degs_norm['id']

            if norm_method == "log2(data+1)":
                X_log = np.log2(X_ + 1)
            elif norm_method == "log2(data)":
                X_log = np.log2(X_)
            elif norm_method == "log10(data+1)":
                X_log = np.log10(X_ + 1)
            else:  # log10
                X_log = np.log10(X_)

            X = pd.DataFrame(X_log, index=X_.index, columns=X_.columns)

            X_display = X.copy()
            X_display.insert(0, 'id', z.values)

            st.subheader("Normalized Data")
            st.write(X_display)
        else:
            X = df_cnt_degs_norm.drop(['id', 'label'], axis=1)
            y = df_cnt_degs_norm['label']
            z = df_cnt_degs_norm['id']

        st.write("---")  
        st.title("Step3: XGBoost + SHAP values")
        st.subheader("XGBoost")
        xgboost_shap_analysis(X, y)
    else:
        st.warning("Please select whether to normalize the data in order to proceed to Step3.")
