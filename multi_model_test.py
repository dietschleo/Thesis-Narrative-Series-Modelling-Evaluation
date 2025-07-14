import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from single_model_test import EvaluationSingleModel
from pandas.tseries.offsets import DateOffset
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel, norm
from sklearn.metrics import jaccard_score
import warnings
warnings.filterwarnings("ignore", message="X has feature names")

print("imported EvaluationMultiModel vers 1.0")

class EvaluationMultiModel:
    def __init__(self, df: pd.DataFrame, folder, scaler: str = 'StandardScaler'):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'], dayfirst=True)
        self.df['date'] = self.df['date'].dt.to_period('M').dt.to_timestamp()
        self.collected_rmse_df = None #initiated with collect_model_rmses
        self.dm_pval_matrix = None #initiated with dm_tests
        self.dm_stat_matrix = None
        self.coef_df = None #initiated with get_non_zero_coef_matrix

        self.folder = folder
        self.scaler = scaler #not used so far (since each model is scaled individually)
        self.eval_models = []

    def list_pkl_files(self):
        """Returns a list of all .pkl files in the specified folder.""" 
        if not self.folder:
                raise ValueError("No folder specified.")
        return [f for f in os.listdir(self.folder) if f.endswith('.pkl')]
    
    def CreateEvaluations(self):
        """
        For each .pkl file in the folder, loads the model_dict, increments last_train_date by 1 month,
        and creates an EvaluationSingleModel instance.
        Returns a list of EvaluationSingleModel objects.
        """
        eval_models = []
        pkl_files = self.list_pkl_files()
        for pkl_file in pkl_files:
            model_dict = joblib.load(os.path.join(self.folder, pkl_file))
            last_train_date = pd.to_datetime(model_dict['last_train_date'])
            out_of_sample_start = last_train_date + DateOffset(months=1)
            eval_model = EvaluationSingleModel(
                df=self.df,
                model_dict=model_dict,
                out_of_sample_start=out_of_sample_start,
                #disable_scaling=True
                mute=True
            )
            eval_models.append(eval_model)
        
        self.eval_models = eval_models
        return eval_models
    
    def get_non_zero_coef_matrix(self) -> pd.DataFrame:
        """
        Extracts and returns a DataFrame of non-zero coefficients from each model
        in the EvaluationMultiModel instance.

        Rows = model dates, Columns = feature names.
        """
        if not self.eval_models:
            self.CreateEvaluations()

        coef_dict = {}
        for eval_model in self.eval_models:
            model_date = pd.to_datetime(eval_model.model_dict['last_train_date']).strftime('%Y-%m')
            coefs = eval_model.model_dict.get('non_zero_coefs', {})
            coef_dict[model_date] = pd.Series(coefs)

        coef_df = pd.DataFrame(coef_dict).T.sort_index()
        coef_df = coef_df.loc[:, (coef_df != 0).any(axis=0)]  # Drop all-zero columns

        return coef_df

    def plot_coef_heatmap(self, coef_df: pd.DataFrame = None, title: str = "Non-Zero Coefficient Heatmap (White=Zero)"):


        if coef_df is None:
            coef_df = self.get_non_zero_coef_matrix()

        # Compute color bounds
        zmax = coef_df.max().max()
        zmin = coef_df.min().min()

        # Handle degenerate case (no variation)
        if zmin == zmax:
            zmin, zmax = -1, 1

        midpoint = 0.0

        # Build the custom colorscale with actual coefficient values
        custom_colorscale = [
            [zmin, "rgb(0,0,255)"],          # strong blue = most negative
            [zmin * 0.5, "rgb(173,216,230)"],# pale blue = closer to zero
            [midpoint, "rgb(255,255,255)"],  # white at zero
            [zmax * 0.5, "rgb(255,182,193)"],# light pink = closer to zero
            [zmax, "rgb(255,0,0)"]           # strong red = most positive
        ]

        # px.imshow() expects normalized colorscale (0 to 1), so we use `color_continuous_midpoint`
        fig = px.imshow(
            coef_df,
            color_continuous_scale="RdBu",  # fallback — overridden next
            zmin=zmin,
            zmax=zmax,
            aspect='auto',
            labels=dict(x="Features", y="Model Date", color="Coefficient"),
            title=title
        )

        # Manually override the color axis with real-valued scale
        fig.update_traces(
            zmin=zmin,
            zmax=zmax,
            colorscale=[
                [0.0, "rgb(0,0,255)"],
                [0.25, "rgb(173,216,230)"],
                [0.5, "rgb(255,255,255)"],
                [0.75, "rgb(255,182,193)"],
                [1.0, "rgb(255,0,0)"]
            ]
        )

        fig.update_layout(height=600, width=900)
        fig.show()

    def compute_jaccard_similarity_over_time(self) -> pd.DataFrame:
        """
        Computes Jaccard similarity between the sets of non-zero coefficients
        in consecutive models (i.e., row-wise comparison in binary form).

        Returns:
            DataFrame with columns: ['row1', 'row2', 'jaccard_similarity']
        """

        # Ensure we have the coef matrix
        if not hasattr(self, 'coef_df') or self.coef_df is None:
            print("Retrieving coefiscients...")
            self.get_non_zero_coef_matrix()

        coef_df = self.coef_df

        # Convert to binary: True if coef ≠ 0
        binary = coef_df != 0

        jaccard_similarities = []
        rows = binary.index.tolist()

        for i in range(len(binary) - 1):
            sim = jaccard_score(binary.iloc[i], binary.iloc[i + 1])
            jaccard_similarities.append((rows[i], rows[i + 1], sim))

        jdf = pd.DataFrame(jaccard_similarities, columns=["row1", "row2", "jaccard_similarity"])
        return jdf


    def collect_model_rmses(self, eval_models=None, start_date=None, window_size=12):
        """
        Computes RMSE from `start_date` (or model-specific last_train_date+1mo)
        for each model in `eval_models`, using EvaluationSingleModel.predict(X).
        
        Returns a DataFrame with columns: date, model, error, is_test.
        """
        if eval_models is None:
            if not self.eval_models:
                eval_models = self.CreateEvaluations()
            else:
                eval_models = self.eval_models

        model_files = self.list_pkl_files()
        model_names = [os.path.splitext(f)[0] for f in model_files]

        results = []


        for eval_model, model_name in zip(eval_models, model_names):
            X_all = eval_model.scaled_df[eval_model.used_features]
            y_all = eval_model.scaled_df[eval_model.actual_cpi_col]
            valid_mask = ~X_all.isna().any(axis=1) & y_all.notna()
            X_all = X_all.loc[valid_mask]
            y_all = y_all.loc[valid_mask]

            # Fix the index to match 'date'
            dates = pd.to_datetime(eval_model.scaled_df.loc[valid_mask, 'date'], dayfirst=True)
            X_all.index = dates
            y_all.index = dates

            try:
                y_pred = eval_model.predict(X_all)
            except Exception as e:
                print(f"Prediction failed for model {model_name}: {e}")
                continue

            for date, y_true, y_hat in zip(y_all.index, y_all.values, y_pred):
                is_test = int(date >= eval_model.out_of_sample_start)
                results.append({
                    'date': date,
                    'model': model_name,
                    'actual': y_true,
                    'predicted': y_hat,
                    'error': y_hat - y_true,
                    'squared_error': (y_hat - y_true) ** 2,
                    'is_test': is_test
                })

        df_results = pd.DataFrame(results)
        df_results["rmse"] = np.sqrt(df_results["squared_error"])
        self.collected_rmse_df = df_results
        return df_results
    
    def plot_average_rmse_over_time(self):
        """
        Plots average RMSE over time for train and test sets.
        If `self.collected_rmse_df` is not populated, it triggers `collect_model_rmses()`.
        """
        if not hasattr(self, 'collected_rmse_df') or self.collected_rmse_df is None:
            print("No RMSE data found — computing it now.")
            self.collected_rmse_df = self.collect_model_rmses()

        df = self.collected_rmse_df.copy()

        # Compute average RMSE grouped by date and is_test
        avg_rmse_by_test = df.groupby(['date', 'is_test'])['rmse'].mean().reset_index()

        # Pivot to get separate columns for test==0 and test==1
        avg_rmse_pivot = avg_rmse_by_test.pivot(index='date', columns='is_test', values='rmse')
        avg_rmse_pivot = avg_rmse_pivot.rename(columns={0: 'train', 1: 'test'})

        # Plot
        fig = px.line(
            avg_rmse_pivot,
            x=avg_rmse_pivot.index,
            y=['train', 'test'],
            labels={'value': 'Average RMSE', 'date': 'Date', 'variable': 'Set'},
            title='Average RMSE Over Time (Train vs Test)'
        )
        fig.show()

    def compute_pairwise_dm_tests(self, holm_correction=False): #TODO see exactly what this does
        if self.collected_rmse_df is None:
            self.collect_model_rmses()

        df = self.collected_rmse_df.copy()
        pivot = df.pivot(index='date', columns='model', values='error').dropna()

        models = pivot.columns.tolist()
        n_models = len(models)
        pval_matrix = pd.DataFrame(np.ones((n_models, n_models)), index=models, columns=models)
        stat_matrix = pd.DataFrame(np.zeros((n_models, n_models)), index=models, columns=models)

        raw_pvals = []
        pairs = []

        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_i, model_j = models[i], models[j]
                d = pivot[model_i] - pivot[model_j]
                d = d.dropna()

                dm_stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                pval = 2 * (1 - norm.cdf(abs(dm_stat)))


                stat_matrix.loc[model_i, model_j] = dm_stat
                stat_matrix.loc[model_j, model_i] = -dm_stat
                pval_matrix.loc[model_i, model_j] = pval
                pval_matrix.loc[model_j, model_i] = pval

                raw_pvals.append(pval)
                pairs.append((model_i, model_j))

        if holm_correction:
            reject, pvals_corrected, _, _ = multipletests(raw_pvals, method='holm')
            for (i, (m1, m2)), pv_corr in zip(enumerate(pairs), pvals_corrected):
                pval_matrix.loc[m1, m2] = pv_corr
                pval_matrix.loc[m2, m1] = pv_corr

        self.dm_pval_matrix = pval_matrix
        self.dm_stat_matrix = stat_matrix
        return pval_matrix, stat_matrix

    def plot_dm_pval_matrix(self):
        if not hasattr(self, 'dm_pval_matrix') or self.dm_pval_matrix is None:
            print("Computing DM tests first...")
            self.compute_pairwise_dm_tests()

        df = self.dm_pval_matrix.copy() #I CHANGED THIS TO STAT_MATRIX (was dm_pval_matrix)

        # Custom color scale: white → light grey → dark grey → black
        color_scale = [
            [0.0, "white"],       # p = 0.0
            [0.9, "lightgrey"],   # p < 0.9
            [0.95, "grey"],       # 0.9 <= p < 0.95
            [0.99, "dimgray"],    # 0.95 <= p < 0.99
            [1.0, "black"]        # p >= 0.99
        ]

        fig = go.Figure(data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            zmin=0,
            zmax=1,
            colorscale=color_scale,
            colorbar=dict(title="p-value"),
            text=df.round(3).astype(str),
            hoverinfo="text"
        ))

        fig.update_layout(
            title="Pairwise Diebold-Mariano p-value Matrix",
            xaxis_title="Model",
            yaxis_title="Model",
            template="plotly_white"
        )
        fig.show()
