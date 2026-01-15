import numpy as np
import pandas as pd
import os
import pickle
import argparse
from scipy.io import loadmat
from tqdm import tqdm

from contrastive import CPCA
from sklearn.decomposition import PCA
from sparsecca import pmd


def load_data(data_root, fc_dir):
    # Load clinical data
    imputed_clinical_file = os.path.join(data_root, "MDD_imputed_1.xlsx")
    imputed_cli_df = pd.read_excel(imputed_clinical_file)
    imputed_cli_df.columns = imputed_cli_df.columns.str.strip()
    
    # Load covariate data (only for grouping subjects)
    covariate_file = os.path.join(data_root, "covariate_data.xlsx")
    cov_df = pd.read_excel(covariate_file)
    cov_df.columns = cov_df.columns.str.strip()

    # Get available .mat files
    mat_files = {os.path.splitext(f)[0]: os.path.join(fc_dir, f)
                 for f in os.listdir(fc_dir) if f.endswith(".mat")}

    # Load MDD patient data
    mdd_subjs_from_cov = cov_df[cov_df['group'] != 2]['subject_id'].tolist()
    available_mdd_subjs = [subj for subj in mdd_subjs_from_cov if subj in mat_files]
    
    mdd_subjs_list = []
    mdd_fcs_list = []
    for subj in available_mdd_subjs:
        fc = loadmat(mat_files[subj])["ROICorrelation_FisherZ"]
        if fc.shape == (100, 100, 1):
            fc = fc[:, :, 0]
        vec = fc[np.tril_indices(100, k=-1)]
        mdd_subjs_list.append(subj)
        mdd_fcs_list.append(vec)

    mdd_fc_df = pd.DataFrame({"subject_id": mdd_subjs_list, "w0_fc": mdd_fcs_list})

    # Load healthy control data
    all_hc_subjs = cov_df[cov_df['group'] == 2]['subject_id'].tolist()
    available_hc_subjs = [subj for subj in all_hc_subjs if subj in mat_files]

    hc_fcs_list = []
    for subj in available_hc_subjs:
        fc = loadmat(mat_files[subj])["ROICorrelation_FisherZ"]
        if fc.shape == (100, 100, 1):
            fc = fc[:, :, 0]
        vec = fc[np.tril_indices(100, k=-1)]
        hc_fcs_list.append(vec)

    hc_data = np.array(hc_fcs_list)

    # Align clinical data with available subjects
    cli_df = imputed_cli_df[imputed_cli_df['subject_id'].isin(available_mdd_subjs)].reset_index(drop=True)
    cli_df = cli_df.set_index('subject_id').reindex(available_mdd_subjs).reset_index()

    return hc_data, mdd_fc_df, cli_df


def get_v_top(mdl, alpha):
    n_components = mdl.n_components
    sigma = mdl.fg_cov - alpha * mdl.bg_cov

    w, v = np.linalg.eig(sigma)

    eig_idx = np.argpartition(w, -n_components)[-n_components:]
    eig_idx = eig_idx[np.argsort(-w[eig_idx])]
    v_top = v[:, eig_idx].astype(np.float32)

    try:
        weight = mdl.pca.components_ 
        v_top = weight.T @ v_top
    except:
        pass
    
    return v_top


def calculate_variance_explained_for_components(foreground, background, alpha, n_components):
    mdl = CPCA(n_components=n_components)
    mdl.fit(foreground, background, preprocess_with_pca_dim=99999)

    sigma = mdl.fg_cov - alpha * mdl.bg_cov
    eigenvalues = np.linalg.eigvals(sigma)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    
    if len(positive_eigenvalues) == 0:
        return 0.0
    
    n_useful = min(n_components, len(positive_eigenvalues))
    if n_useful == 0:
        return 0.0
        
    total_positive_variance = np.sum(positive_eigenvalues)
    explained_variance = np.sum(positive_eigenvalues[:n_useful]) / total_positive_variance
    
    return explained_variance


def get_args():
    parser = argparse.ArgumentParser(description='cPCA + sCCA Analysis')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing clinical data files')
    parser.add_argument('--fc_dir', type=str, required=True,
                        help='Directory containing functional connectivity .mat files')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--alpha', type=float, default=2.8,
                        help='Contrast parameter for cPCA')
    parser.add_argument('--penaltyu', type=float, default=0.8,
                        help='Sparsity penalty for brain patterns in sCCA')
    parser.add_argument('--penaltyv', type=float, default=0.9,
                        help='Sparsity penalty for clinical patterns in sCCA')
    parser.add_argument('--seed', type=int, default=2014,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_components', type=int, default=80,
                        help='Number of cPCA components')
    parser.add_argument('--cv_runs', type=int, default=10,
                        help='Number of cross-validation runs')
    parser.add_argument('--cv_folds', type=int, default=10,
                        help='Number of cross-validation folds')
    
    args = parser.parse_args()
    return args

# Main Analysis
args = get_args()
np.random.seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)

# Load data
hc_data, patient_EMBARC_df, cli_df = load_data(args.data_root, args.fc_dir)

# Prepare patient data
patient_EMBARC_df = patient_EMBARC_df.groupby('subject_id').apply(
    lambda l: np.sum(l)/len(l)).reset_index()

# Define clinical features
selected_features = [
    "S_AT", "S_rME", "S_pME", "S_EF", "O_EF", "O_AT", "O_PS", "O_ME", "HAMD" 
]

N_scales = len(selected_features)

# Merge datasets
full_patient_df = patient_EMBARC_df.merge(cli_df, on='subject_id')
full_patient_df = full_patient_df.sort_values('subject_id').reset_index(drop=True)

patient_fc = np.stack(full_patient_df['w0_fc'])
subject_id = full_patient_df['subject_id']
cli_data = full_patient_df[selected_features].to_numpy()

foreground = patient_fc
background = hc_data

print(f'Parameters: seed={args.seed}, alpha={args.alpha}, '
      f'penaltyu={args.penaltyu}, penaltyv={args.penaltyv}')

# Fit cPCA on fc data and PCA on clinical data
mdl = CPCA(n_components=args.n_components)

fg_mean = foreground.mean(axis=0)
fg_std = foreground.std(axis=0)
mdl.fit(foreground, background, preprocess_with_pca_dim=99999)

v_top_all = get_v_top(mdl, args.alpha)
cpcaed = ((foreground - fg_mean) / fg_std).dot(v_top_all)

# Prepare clinical data
cli_mean = cli_data.mean(axis=0)
cli_std = cli_data.std(axis=0)
cli_data_normed = (cli_data - cli_mean) / cli_std

pca_cli = PCA(n_components=len(selected_features))
cli_data_normed_PCA = pca_cli.fit_transform(cli_data_normed)

# Run sCCA
X = cpcaed
Z = cli_data_normed_PCA

U, V, D = pmd(
    X.T @ Z, 
    K=N_scales, 
    penaltyu=args.penaltyu, 
    penaltyv=args.penaltyv, 
    standardize=False
)

Rs_all = []
Us_all = []
Vs_all = []
Us_all_scores = []
Vs_all_scores = []

for component_i in range(N_scales):
    x_weights = U[:, component_i]
    z_weights = V[:, component_i]
    
    U_score = np.dot(x_weights, X.T)
    V_score = np.dot(z_weights, Z.T)
    
    corrcoef = np.corrcoef(U_score, V_score)[0, 1]
    Rs_all.append(corrcoef)
    
    Us_all.append(x_weights / np.max(np.abs(x_weights)))
    Vs_all.append(z_weights / np.max(np.abs(z_weights)))
    
    Us_all_scores.append(U_score)
    Vs_all_scores.append(V_score)

FC_loadings = np.array(Us_all) @ v_top_all.T
Vs_all = np.array(Vs_all) @ pca_cli.components_


# Cross-validation
reses = []
idx_tests = []
rows_test_scores = []
rows_U_test = []
rows_V_test = []

for run_i in range(args.cv_runs):
    print(f"Run {run_i + 1}/{args.cv_runs}")
    
    subj_uniq = np.unique(subject_id)
    n_subj = subj_uniq.shape[0]
    subj_shuffled = subj_uniq[np.random.permutation(n_subj)]
    
    Rs_train, Us_train, Vs_train = [], [], []
    Us_test, Vs_test = [], []
    FC_loadings_cv = []
    v_tops_cv = []
    
    for fold_i in tqdm(range(args.cv_folds), desc=f"Run {run_i + 1}"):
        subj_test = subj_shuffled[
            int(n_subj * fold_i / args.cv_folds):int(n_subj * (fold_i + 1) / args.cv_folds)]
        
        idx_train = ~np.isin(subject_id, subj_test)
        idx_test = np.isin(subject_id, subj_test)
        idx_tests.append(idx_test)
        
        foreground_train = foreground[idx_train]
        foreground_test = foreground[idx_test]
        
        fg_mean_train = foreground_train.mean(axis=0)
        fg_std_train = foreground_train.std(axis=0)
        
        # Fit cPCA on training data
        mdl_train = CPCA(n_components=args.n_components)
        mdl_train.fit(foreground_train, background, preprocess_with_pca_dim=99999)
        v_top_train = get_v_top(mdl_train, args.alpha)
        v_tops_cv.append(v_top_train)
        
        cpcaed_train = ((foreground_train - fg_mean_train) / fg_std_train).dot(v_top_train)
        cpcaed_test = ((foreground_test - fg_mean_train) / fg_std_train).dot(v_top_train)
        
        # Prepare clinical data
        cli_train = cli_data[idx_train]
        cli_test = cli_data[idx_test]
        
        cli_mean_train = cli_train.mean(axis=0)
        cli_std_train = cli_train.std(axis=0)
        cli_train_normed = (cli_train - cli_mean_train) / cli_std_train
        cli_test_normed = (cli_test - cli_mean_train) / cli_std_train
        
        pca_cli_fold = PCA(n_components=len(selected_features))
        cli_train_normed_PCA = pca_cli_fold.fit_transform(cli_train_normed)
        cli_test_normed = cli_test_normed @ pca_cli_fold.components_.T
        
        # Run sCCA on training data
        X_fold = cpcaed_train
        Z_fold = cli_train_normed_PCA
        
        U_fold, V_fold, D_fold = pmd(
            X_fold.T @ Z_fold, 
            K=N_scales, 
            penaltyu=args.penaltyu, 
            penaltyv=args.penaltyv, 
            standardize=False
        )
        
        Rs_train_fold = []
        Us_train_fold = []
        Vs_train_fold = []
        Us_test_fold = []
        Vs_test_fold = []
        
        for component_i in range(N_scales):
            x_weights_fold = U_fold[:, component_i]
            z_weights_fold = V_fold[:, component_i]
            
            corrcoef_fold = np.corrcoef(
                np.dot(x_weights_fold, X_fold.T),
                np.dot(z_weights_fold, Z_fold.T)
            )[0, 1]
            
            Rs_train_fold.append(corrcoef_fold)
            Us_train_fold.append(x_weights_fold / np.max(np.abs(x_weights_fold)))
            Vs_train_fold.append(z_weights_fold / np.max(np.abs(z_weights_fold)))
            
            Us_test_fold.append(np.dot(x_weights_fold, cpcaed_test.T))
            Vs_test_fold.append(np.dot(z_weights_fold, cli_test_normed.T))
        
        # Store test scores
        test_subjects = subject_id[idx_test].values
        for comp_i in range(N_scales):
            u_scores = Us_test_fold[comp_i]
            v_scores = Vs_test_fold[comp_i]
            
            for subj_id, u, v in zip(test_subjects, u_scores, v_scores):
                rows_test_scores.append({
                    "Run": run_i,
                    "Fold": fold_i,
                    "Subject_ID": subj_id,
                    "Variable": "U",
                    "Component": comp_i,
                    "Score": float(u)
                })
                rows_test_scores.append({
                    "Run": run_i,
                    "Fold": fold_i,
                    "Subject_ID": subj_id,
                    "Variable": "V",
                    "Component": comp_i,
                    "Score": float(v)
                })
                
                rows_U_test.append({
                    "Run": run_i,
                    "Fold": fold_i,
                    "Subject_ID": subj_id,
                    "Component": comp_i,
                    "U_score": float(u)
                })
                rows_V_test.append({
                    "Run": run_i,
                    "Fold": fold_i,
                    "Subject_ID": subj_id,
                    "Component": comp_i,
                    "V_score": float(v)
                })
        
        FC_loadings_fold = np.array(Us_train_fold) @ v_top_train.T
        Vs_train_fold = np.array(Vs_train_fold) @ pca_cli_fold.components_
        
        Rs_train.append(Rs_train_fold)
        Us_train.append(Us_train_fold)
        Vs_train.append(Vs_train_fold)
        FC_loadings_cv.append(FC_loadings_fold)
        Us_test.append(Us_test_fold)
        Vs_test.append(Vs_test_fold)
    
    reses.append((Rs_train, Us_train, Vs_train, Us_test, Vs_test, 
                  v_tops_cv, FC_loadings_cv))

# Save results
output_file = os.path.join(
    args.output_dir,
    f'alpha_{args.alpha}_penaltyu_{args.penaltyu}_'
    f'penaltyv_{args.penaltyv}_seed_{args.seed}.pkl'
)

with open(output_file, 'wb') as f:
    pickle.dump((reses, np.array(Rs_all), np.array(Us_all), Vs_all, 
                 v_top_all, selected_features), f)
