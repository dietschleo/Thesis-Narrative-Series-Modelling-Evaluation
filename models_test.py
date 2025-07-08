import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from datetime import datetime
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", message="X has feature names")
from sklearn.linear_model import LassoCV, ElasticNetCV, LassoLarsIC
import os
import joblib
from pandas.tseries.offsets import DateOffset

print("imported EvaluationSingleModel vers 1.1")

class EvaluationSingleModel:
    def __init__(self, df: pd.DataFrame, model_dict: dict,
                 out_of_sample_start: datetime = datetime(2021, 7, 1),
                 actual_cpi_col: str = 'delta_log_cpi_next_month_lag0',
                 disable_scaling: bool = False):
        self.df = df.copy()
        self.model_dict = model_dict
        self.out_of_sample_start = pd.to_datetime(out_of_sample_start)
        self.actual_cpi_col = actual_cpi_col
        self.scaled_df = None
        self.disable_scaling = disable_scaling

        if 'date' not in df.columns:
            raise ValueError("The DataFrame must contain a 'date' column.")
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Step 1: raw features from model_dict
        self.used_features = model_dict['non_zero_coefs'].index.tolist()
        # Exclude 'date' from feature_names if present
        self.feature_names = [f for f in model_dict.get('feature_names', []) if f != 'date']

        # Step 2: resolve lag0-suffixed column names
        self.used_features = self.resolve_lag0_features(df, self.used_features)
        self.feature_names = self.resolve_lag0_features(df, self.feature_names)
        # Align non-zero coefficients to resolved used_features
        resolved_feature_map = dict(zip(model_dict['non_zero_coefs'].index, self.used_features))

        # Rename the index of non_zero_coefs using the resolved names
        self.non_zero_coefs = model_dict['non_zero_coefs'].copy()
        self.non_zero_coefs.index = self.non_zero_coefs.index.map(lambda k: resolved_feature_map.get(k, k))

        # Optional: check that all used_features now exist in the df
        missing = [f for f in self.used_features if f not in df.columns]
        if missing:
            raise ValueError(f"The following features are still missing: {missing}")

        #print(df['date'].head())
        # Step X: filter numeric columns and apply variance thresholding like training
        numeric = self.df[self.feature_names].select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
        numeric = numeric.fillna(0)

        # Variance thresholding
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.0)
        X_reduced = selector.fit_transform(numeric)
        selected_columns = np.array(self.feature_names)[selector.get_support()]
        df_reduced = pd.DataFrame(X_reduced, columns=selected_columns)

        ## SCALING           
        if self.disable_scaling:
            self.scaled_df = df_reduced
        else:
            # Scale with ZCA (same shape preservation)
            if model_dict['scaler'].lower() == 'zca':
                whitened = self.zca_whiten(df_reduced)
                whitened_df = pd.DataFrame(whitened, columns=df_reduced.columns, index=df_reduced.index)
                # Explicitly add 'date' and other non-feature columns back
                self.scaled_df = pd.concat([whitened_df, self.df[['date', self.actual_cpi_col]]], axis=1)
                

            #print(model_dict['scaler'])

            else: #resorting to standard scaler
                if model_dict['scaler'] not in ['standard', 'StandardScaler']:
                    print("No adequate scaler found in model_dict, resorting to StandardScaler.")
                # 1. Fit on in-sample data only
                scaler = StandardScaler()
                scaler.fit(df_reduced.query('date < @self.out_of_sample_start')[self.feature_names])
                # 2. Transform full dataset on used features
                scaled_values = scaler.transform(df_reduced[self.feature_names])
                scaled_df = pd.DataFrame(scaled_values, columns=self.feature_names, index=df_reduced.index)
                # 3. Combine scaled features with original non-feature data (like 'date', target, etc.)
                self.scaled_df = pd.concat([scaled_df, df_reduced.drop(columns=self.feature_names)], axis=1)

        #print(self.scaled_df)
        print("Number of records in non NaN scaled_df:", len(self.scaled_df.dropna()))
        print("Resolved used_features:", self.used_features)


        self.model = model_dict.get('model')
        self.description = model_dict.get('description')
        self.mse = model_dict.get('mse')
        if self.description is not None:
            print('Now processing model:', self.description)
        
        print("Non-zero coefs in model_dict:", model_dict['non_zero_coefs'])

        
    def summary(self):
        """Prints a summary of the model's configuration and coefficients."""
        print("Model Summary")
        print("=" * 40)
        print(f"Description       : {self.description}")
        print(f"Model Type        : {type(self.model).__name__}")
        print(f"Features Used     : {', '.join(self.used_features)}")
        print(f"Mean Squared Error: {self.mse:.8f}")
        print("\nNon-zero Coefficients:")
        print(self.non_zero_coefs.round(6).to_string())
        print("=" * 40)

    def resolve_lag0_features(self, df: pd.DataFrame, feature_list: list, suffix="_lag0") -> list:
        resolved = []
        for feat in feature_list:
            if feat in df.columns:
                resolved.append(feat)
            elif f"{feat}{suffix}" in df.columns:
                resolved.append(f"{feat}{suffix}")
            else:
                raise ValueError(f"Feature '{feat}' not found in DataFrame (even with '{suffix}' fallback).")
        return resolved


    def predict(self,X):
        """Return model predictions on the current DataFrame."""
        
        if self.non_zero_coefs is None or len(self.non_zero_coefs) == 0:
            warnings.warn("Model has no non-zero coefficients. Returning zero predictions.", UserWarning)
            return np.zeros(len(X))
        # Ensure all required features are in the DataFrame
        missing = [f for f in self.used_features if f not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required features in input DataFrame: {missing}")
        
        # Extract feature matrix
        #X = self.scaled_df[self.used_features]

                #so that the model doesn't expect thousands of features for LassoCV or ElasticNetCV
        if isinstance(self.model, (LassoCV, ElasticNetCV, LassoLarsIC)):
            warnings.warn(
                "Model is of type LassoCV or ElasticNetCV. Falling back to manual dot product prediction.",
                UserWarning
            )
            return self.manual_predict_from_coefs(self.non_zero_coefs, X)
        

        # Drop rows with NaNs and inform user
        nan_rows = X[X.isna().any(axis=1)]
        dropped_dates = self.df.loc[nan_rows.index, 'date'].tolist() if 'date' in self.df.columns else []
        X = X.dropna()
        if len(nan_rows) > 0:
            print(f"Dropped {len(nan_rows)} rows due to NaN values. Associated dates: {dropped_dates}")
        
            # Use manual prediction if model is LassoCV or ElasticNetCV
        

        # Check feature alignment (optional but safe)
        model_coef_order = list(self.model.coef_)  # Ensure it's the same length
        if len(model_coef_order) != X.shape[1]:
            raise ValueError(
                f"Mismatch between model coefficient count ({len(model_coef_order)}) "
                f"and input feature count ({X.shape[1]})."
            )


        return self.model.predict(X)

    def manual_predict_from_coefs(self, non_zero_coefs, X):
        # Ensure the DataFrame includes all needed features
        missing = [feat for feat in non_zero_coefs.index if feat not in X.columns]
        if missing:
            raise ValueError(f"Missing features in X: {missing}")

        # Subset and compute predictions
        X_sub = X[non_zero_coefs.index]
        return X_sub.values @ non_zero_coefs.values

    def zca_whiten(self, X, epsilon=1e-5):
        X = np.asarray(X)
        X_centered = X - X.mean(axis=0)

        cov = np.cov(X_centered, rowvar=False)
        cov += np.eye(cov.shape[0]) * epsilon

        # Eigen decomposition
        S, U = np.linalg.eigh(cov)

        # Clip to avoid negative sqrt due to numerical noise
        S = np.clip(S, a_min=0, a_max=None)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(S + epsilon))

        ZCA_matrix = U @ D_inv_sqrt @ U.T
        X_zca = X_centered @ ZCA_matrix

        if np.any(np.isnan(X_zca)) or np.any(np.isinf(X_zca)):
            raise ValueError("❌ ZCA whitening produced invalid values (NaN or Inf).")

        return X_zca


    def plot_predictions(self, show_actual=True):
        """Interactive plot of model predictions (and optionally actuals), using dashed lines after out-of-sample start date."""
        
        if 'date' not in self.scaled_df.columns:
            raise ValueError("The scaled DataFrame must contain a 'date' column to plot predictions.")

        # 1. Prepare the feature matrix and drop NaN rows
        X = self.scaled_df[self.used_features]
        valid_mask = ~X.isna().any(axis=1)
        df_valid = self.scaled_df.loc[valid_mask].copy()

        # 2. Get predictions
        predictions = self.predict(X[valid_mask])

        # 3. Attach predictions to the valid DataFrame
        df_valid['prediction'] = predictions

        # 4. Split into in-sample and out-of-sample
        in_sample_mask = df_valid['date'] < self.out_of_sample_start
        out_sample_mask = ~in_sample_mask

        fig = go.Figure()

        # In-sample predictions
        fig.add_trace(go.Scatter(
            x=df_valid.loc[in_sample_mask, 'date'],
            y=df_valid.loc[in_sample_mask, 'prediction'],
            mode='lines+markers',
            name='Predicted CPI (in-sample)',
            line=dict(color = 'blue', dash='solid')
        ))

        # Out-of-sample predictions
        fig.add_trace(go.Scatter(
            x=df_valid.loc[out_sample_mask, 'date'],
            y=df_valid.loc[out_sample_mask, 'prediction'],
            mode='lines+markers',
            name='Predicted CPI (out-of-sample)',
            line=dict(color = 'blue', dash='dot')
        ))

        # Optional: plot actuals
        if show_actual:
            if self.actual_cpi_col not in df_valid.columns:
                raise ValueError(f"Column {self.actual_cpi_col} not found in DataFrame.")

            fig.add_trace(go.Scatter(
                x=df_valid['date'],
                y=df_valid[self.actual_cpi_col],
                mode='lines+markers',
                name='Actual CPI',
                line=dict(color='black', dash='solid')
            ))

        fig.update_layout(
            title='CPI Predictions vs. Actuals Over Time',
            xaxis_title='Date',
            yaxis_title='Delta log CPI',
            hovermode='x unified',
            template='plotly_white'
        )

        fig.show()

    def compute_rmse_from_date(self, start_date: datetime, window_size: int) -> float: #TODO 
        start_date = pd.to_datetime(start_date)

        # Prepare feature matrix and drop NaN rows
        X = self.scaled_df[self.used_features]
        
        df_valid = self.scaled_df.dropna(subset=self.used_features).copy()

        # Filter rows starting from the given date
        df_window = df_valid[df_valid['date'] >= start_date].head(window_size)

        if len(df_window) < window_size:
            print(f"⚠️ Only {len(df_window)} valid rows found starting from {start_date}, fewer than requested ({window_size}).")

        if self.actual_cpi_col not in df_window.columns:
            raise ValueError(f"Column '{self.actual_cpi_col}' not found in DataFrame.")

        # Predictions
        predictions = self.predict(df_window[self.used_features])

        # Actuals
        actuals = df_window[self.actual_cpi_col].values
        #print(actuals, predictions)
        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        return rmse

    def rolling_forward_rmse(self, window_size: int = 12) -> pd.DataFrame:

        X = self.scaled_df[self.used_features]
        #valid_mask = ~X.isna().any(axis=1)
        #df_valid = self.scaled_df.loc[valid_mask].copy()
        df_valid = self.scaled_df.copy()

        results = []
        num_rows = len(df_valid)

        for i in range(num_rows - window_size + 1):
            window = df_valid.iloc[i: i + window_size].copy()

            # Drop rows with NaNs in target column
            window = window.dropna(subset=self.used_features + [self.actual_cpi_col])
            if len(window) < window_size:
                continue  # skip incomplete windows

            y_true = window[self.actual_cpi_col].values
            y_pred = self.predict(window[self.used_features])
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            results.append({
                'start_date': window.iloc[0]['date'],
                'rmse': rmse
            })

        return pd.DataFrame(results)

class EvaluationMultiModel:
    def __init__(self, df: pd.DataFrame, folder, scaler: str = 'StandardScaler'):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'], dayfirst=True)
        self.df['date'] = self.df['date'].dt.to_period('M').dt.to_timestamp()

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
                out_of_sample_start=out_of_sample_start
                #disable_scaling=True
            )
            eval_models.append(eval_model)
        
        self.eval_models = eval_models
        return eval_models
    
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
            # Determine start index using model's training boundary if not set
            if start_date is None:
                start = pd.to_datetime(eval_model.model_dict['last_train_date']) + pd.DateOffset(months=1)
            else:
                start = pd.to_datetime(start_date)

            X_all = eval_model.scaled_df[eval_model.used_features]
            #print("X_all.shape",X_all.shape)
            y_all = eval_model.scaled_df[eval_model.actual_cpi_col]
            #print("y_all.shape",y_all.shape)
            valid_mask = ~X_all.isna().any(axis=1) & y_all.notna()
            X_all = X_all.loc[valid_mask]
            #print("X_all.shape after valid_mask",X_all.shape)
            y_all = y_all.loc[valid_mask]
            #print("y_all.shape after valid_mask",y_all.shape)

            # Ensure datetime index
            X_all.index = pd.to_datetime(eval_model.scaled_df.loc[valid_mask, 'date'], dayfirst=True)
            y_all.index = X_all.index  # Already aligned


            X_window = X_all.loc[X_all.index >= start].head(window_size)
            y_window = y_all.loc[X_window.index]
            print(eval_model.scaled_df['date'].head(20).tolist())

            if len(X_window) < window_size:
                print(f"⚠️ Only {len(X_window)} valid rows for model {model_name} from {start} — skipping.")
                continue

            y_pred = eval_model.predict(X_window)
            rmse = np.sqrt(mean_squared_error(y_window, y_pred))
            is_test = int(X_window.index[0] >= eval_model.out_of_sample_start)

            results.append({
                'date': X_window.index[0],
                'model': model_name,
                'error': rmse,
                'is_test': is_test
            })

        return pd.DataFrame(results)

