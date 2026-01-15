clear; clc;

% Configuration: Set input/output paths

% Clinical data files (Excel format)
baseline_path = 'path/to/xxx.xlsx';
month1_path   = 'path/to/xxx.xlsx';
month2_path   = 'path/to/xxx.xlsx';

% sCCA results
fc_score_path = 'path/to/xxx.xlsx';
loading_path  = 'path/to/xxx.xlsx';

% Output directory
output_folder = fullfile(pwd, 'LMM_Output');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Step 1: Read clinical data from all timepoints

data_baseline = readtable(baseline_path);
data_month1   = readtable(month1_path);
data_month2   = readtable(month2_path);

% Ensure subject IDs are strings for consistent merging
data_baseline.subject_id = string(data_baseline.subject_id);
data_month1.subject_id   = string(data_month1.subject_id);
data_month2.subject_id   = string(data_month2.subject_id);

% Convert categorical variables
data_baseline.medication = categorical(data_baseline.medication);
data_baseline.gender     = categorical(data_baseline.gender);

fprintf('Baseline: %d subjects\n', height(data_baseline));
fprintf('Month 1:  %d subjects\n', height(data_month1));
fprintf('Month 2:  %d subjects\n', height(data_month2));

% Step 2: Read and merge sCCA FC scores (component-locked)

% Get all component sheets from the FC score file
[~, sheet_names] = xlsfinfo(fc_score_path);
components = 0:(length(sheet_names)-1);

% Merge FC scores for each component into baseline data
for i = 1:length(sheet_names)
    
    c = components(i);
    
    % Read FC scores for this component
    tmp = readtable(fc_score_path, 'Sheet', sheet_names{i});
    tmp.Subject_ID = string(tmp.Subject_ID);
    
    % Rename columns for merging
    tmp.Properties.VariableNames = ...
        {'subject_id', ['FC_comp', num2str(c)]};
    
    % Left join to preserve all baseline subjects
    data_baseline = outerjoin( ...
        data_baseline, tmp, ...
        'Keys', 'subject_id', ...
        'MergeKeys', true, ...
        'Type', 'left');
end

% Step 3: Handle missing values (mean imputation for continuous variables)

cont_vars = {'age', 'education_years'};

% Add all FC component variables
for c = components
    cont_vars{end+1} = ['FC_comp', num2str(c)];
end

% Mean imputation for missing values
for i = 1:length(cont_vars)
    v = cont_vars{i};
    if any(ismissing(data_baseline.(v)))
        n_missing = sum(ismissing(data_baseline.(v)));
        data_baseline.(v)(ismissing(data_baseline.(v))) = nanmean(data_baseline.(v));
        fprintf('  %s: imputed %d missing values\n', v, n_missing);
    end
end

% Step 4: Calculate z-scores for FC variables (using baseline parameters)

FC_vars = cont_vars(3:end);  % Extract only FC variables
FC_mu = struct();
FC_sd = struct();

for i = 1:length(FC_vars)
    v = FC_vars{i};
    FC_mu.(v) = nanmean(data_baseline.(v));
    FC_sd.(v) = nanstd(data_baseline.(v));
    fprintf('  %s: mu=%.3f, sd=%.3f\n', v, FC_mu.(v), FC_sd.(v));
end

% Step 5: Reshape data from wide to long format

% Clinical outcome variables (measured at each timepoint)
clinical_vars = {   
    'S_AT',
    'S_rME',
    'S_pME',
    'S_EF',
    'O_EF',
    'O_AT',
    'O_PS',
    'O_ME',
    'HAMD'
    };  % Hamilton Depression Rating Scale

% Baseline covariates (time-invariant)
baseline_covars = [{'subject_id', 'medication', 'gender', 'age', 'education_years'}, FC_vars];
data_cov = data_baseline(:, baseline_covars);

% Initialize long-format table
data_long = table();

% Time 0 (baseline)
t0 = data_baseline(:, [{'subject_id'}, clinical_vars]);
t0.Time = zeros(height(t0), 1);
data_long = [data_long; t0];

% Time 1 (month 1)
t1 = data_month1;
t1.Properties.VariableNames(2:end) = clinical_vars;
t1.Time = ones(height(t1), 1);
data_long = [data_long; t1];

% Time 2 (month 2)
t2 = data_month2;
t2.Properties.VariableNames(2:end) = clinical_vars;
t2.Time = 2 * ones(height(t2), 1);
data_long = [data_long; t2];

% Merge with baseline covariates
data_final = outerjoin(data_long, data_cov, ...
    'Keys', 'subject_id', 'MergeKeys', true);

fprintf('Long format data: %d rows (observations)\n', height(data_final));

% Step 6: Apply FC z-score transformation

for i = 1:length(FC_vars)
    v = FC_vars{i};
    data_final.([v '_z']) = (data_final.(v) - FC_mu.(v)) ./ FC_sd.(v);
end

% Step 7: Set variable types for LMM analysis

% Time as categorical factor
data_final.Time = categorical(data_final.Time);
data_final.Time = reordercats(data_final.Time, {'0', '1', '2'});

% Ensure continuous variables are numeric
for i = 1:length(cont_vars)
    data_final.(cont_vars{i}) = double(data_final.(cont_vars{i}));
end

for i = 1:length(clinical_vars)
    data_final.(clinical_vars{i}) = double(data_final.(clinical_vars{i}));
end

% Step 8: Read loading stability table

clinical_tbl = readtable(loading_path, 'Sheet', 'Clinical_stable');
clinical_tbl.Feature = string(clinical_tbl.Feature);

% Step 9: Item-level LMM analysis (Clinical Measures ~ Time × FC)

CogItem_results = table();

for c = components
    
    fc_var = ['FC_comp', num2str(c), '_z'];
    
    % Check if FC variable exists in the data
    if ~ismember(fc_var, data_final.Properties.VariableNames)
        continue;
    end
    
    % Get stable clinical items for this component
    idx_item = clinical_tbl.Component == c & ...
               clinical_tbl.CI_excludes_0 == true;
    
    items = clinical_tbl.Feature(idx_item);
    if isempty(items)
        continue;
    end
    
    % Test each clinical item
    for j = 1:length(items)
        
        dv = items(j);
        fprintf('--- %s ~ Time × %s ---\n', dv, fc_var);
        
        % Fit LMM model
        [~, FE] = run_lmm_model_simple(data_final, dv, fc_var);
        
        if isempty(FE)
            fprintf('    Model failed to converge\n');
            continue;
        end
        
        % Extract Time2 × FC interaction effect
        idx = contains(FE.Name, 'Time_2') & contains(FE.Name, fc_var);
        
        if ~any(idx)
            fprintf('    Interaction term not found\n');
            continue;
        end
        
        % Store results
        CogItem_results = [CogItem_results; table( ...
            string(dv), ...
            c, ...
            string(fc_var), ...
            FE.Estimate(idx), ...
            FE.pValue(idx), ...
            'VariableNames', ...
            {'Item', 'Component', 'FC', 'Estimate_Time2_FC', 'p_Time2_FC'})];
        
        fprintf('    β=%.4f, p=%.4f\n', FE.Estimate(idx), FE.pValue(idx));
    end
end

% Step 10: FDR correction within each component

CogItem_results.p_FDR = nan(height(CogItem_results), 1);

for c = unique(CogItem_results.Component)
    
    idx = CogItem_results.Component == c;
    n_tests = sum(idx);
    
    % FDR correction using Benjamini-Hochberg procedure
    CogItem_results.p_FDR(idx) = ...
        mafdr(CogItem_results.p_Time2_FC(idx), 'BHFDR', true);
    
    n_sig = sum(CogItem_results.p_FDR(idx) < 0.05);
    fprintf('Component %d: %d/%d tests survived FDR correction (q<0.05)\n', ...
        c, n_sig, n_tests);
end

% Step 11: Save results to Excel

output_file = fullfile(output_folder, 'ItemLevel_LMM_ComponentLocked.xlsx');
writetable(CogItem_results, output_file);

fprintf('Results saved to: %s\n', output_file);
fprintf('Total tests: %d\n', height(CogItem_results));
fprintf('Significant after FDR: %d\n', sum(CogItem_results.p_FDR < 0.05));

% Function: Fit Linear Mixed-Effects Model

function [lme, FE] = run_lmm_model_simple(data, dv, fc)
    formula = sprintf( ...
        '%s ~ Time * %s + medication + age + gender + education_years + (1|subject_id)', ...
        dv, fc);
    
    try
        lme = fitlme(data, formula, 'FitMethod', 'REML');
        FE  = lme.Coefficients;
    catch ME
        warning('LMM failed for %s ~ %s: %s', dv, fc, ME.message);
        lme = [];
        FE  = [];
    end
end