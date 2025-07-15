from sklearn.linear_model import LassoCV, ElasticNetCV, LinearRegression, LassoLarsIC
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from datetime import datetime
import joblib
import numpy as np
import os
import pandas as pd

# TODO is understand why lasso and encv yield 0non_zero_coefs
# -> likely due to processing data func. use that of sm

class Model:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df['date'] = pd.to_datetime(self.df['date'], dayfirst=True)
        self.lead = 1 #predict in next month inflation by default
        self.y = self.df.set_index('date')['delta_log_cpi_next_month_lag0'].shift(-(self.lead - 1))
        self.profile = None
        self.profilename = None
        self.maxlag = 24
        self.startdate = pd.to_datetime("2001-05-01")
        self.window = 60 # default window size for rolling regression
        self.auto_print = True
        self.modeltype = None
        self.testsize = 0.2
        self.scaler = "Standard"
        
###########################################################################################################
### Setters for Model parameters ##########################################################################
        
    def set_startdate(self, startdate: str):
        startdate = pd.to_datetime(startdate)
        df_index = pd.to_datetime(self.df.set_index('date').index)
        if startdate not in df_index:
            print(f"Error: startdate {startdate.date()} not found in DataFrame index.")
        self.startdate = startdate
        return self
    
    def set_modeltype(self, modeltype: str):
        if modeltype not in ['linear','lassocv', 'elasticnetcv', 'lassobic', 'elasticnetbic']:
            print("modeltype must be either 'linear', 'lassocv' or 'elasticnetcv'.")
        self.modeltype = modeltype
        return self

    def set_lead(self, lead: int):
        self.lead = lead
        self.y = self.df.set_index('date')['delta_log_cpi_next_month_lag0'].shift(-(self.lead - 1))
        return self
    
    def set_maxlag(self, maxlag: int):
        if maxlag < 0 or maxlag > 24:
            raise ValueError("maxlag must be between 0 and 24.")
        self.maxlag = maxlag - 1 #-1 as lag0 is a month ago
        return self
    
    def set_profile(self, profile):
        def create_inputlist():
            narratives_and_composite = ['std_SD_composite_AU', 'std_SI_composite_AU', 'std_DD_composite_AU', 'std_DI_composite_AU', 'std_netD_composite_AU', 'std_netS_composite_AU', 'std_SD_aluminium_AU', 'std_SI_aluminium_AU', 'std_DD_aluminium_AU', 'std_DI_aluminium_AU', 'std_netD_aluminium_AU', 'std_netS_aluminium_AU', 'std_SD_cattle_AU', 'std_SI_cattle_AU', 'std_DD_cattle_AU', 'std_DI_cattle_AU', 'std_netD_cattle_AU', 'std_netS_cattle_AU', 'std_SD_cocoa_AU', 'std_SI_cocoa_AU', 'std_DD_cocoa_AU', 'std_DI_cocoa_AU', 'std_netD_cocoa_AU', 'std_netS_cocoa_AU', 'std_SD_coffee_AU', 'std_SI_coffee_AU', 'std_DD_coffee_AU', 'std_DI_coffee_AU', 'std_netD_coffee_AU', 'std_netS_coffee_AU', 'std_SD_copper_AU', 'std_SI_copper_AU', 'std_DD_copper_AU', 'std_DI_copper_AU', 'std_netD_copper_AU', 'std_netS_copper_AU', 'std_SD_corn_AU', 'std_SI_corn_AU', 'std_DD_corn_AU', 'std_DI_corn_AU', 'std_netD_corn_AU', 'std_netS_corn_AU', 'std_SD_cotton_AU', 'std_SI_cotton_AU', 'std_DD_cotton_AU', 'std_DI_cotton_AU', 'std_netD_cotton_AU', 'std_netS_cotton_AU', 'std_SD_gasoil_AU', 'std_SI_gasoil_AU', 'std_DD_gasoil_AU', 'std_DI_gasoil_AU', 'std_netD_gasoil_AU', 'std_netS_gasoil_AU', 'std_SD_gasoline_AU', 'std_SI_gasoline_AU', 'std_DD_gasoline_AU', 'std_DI_gasoline_AU', 'std_netD_gasoline_AU', 'std_netS_gasoline_AU', 'std_SD_gold_AU', 'std_SI_gold_AU', 'std_DD_gold_AU', 'std_DI_gold_AU', 'std_netD_gold_AU', 'std_netS_gold_AU', 'std_SD_heatingoil_AU', 'std_SI_heatingoil_AU', 'std_DD_heatingoil_AU', 'std_DI_heatingoil_AU', 'std_netD_heatingoil_AU', 'std_netS_heatingoil_AU', 'std_SD_hog_AU', 'std_SI_hog_AU', 'std_DD_hog_AU', 'std_DI_hog_AU', 'std_netD_hog_AU', 'std_netS_hog_AU', 'std_SD_natgas_AU', 'std_SI_natgas_AU', 'std_DD_natgas_AU', 'std_DI_natgas_AU', 'std_netD_natgas_AU', 'std_netS_natgas_AU', 'std_SD_nickel_AU', 'std_SI_nickel_AU', 'std_DD_nickel_AU', 'std_DI_nickel_AU', 'std_netD_nickel_AU', 'std_netS_nickel_AU', 'std_SD_oil_AU', 'std_SI_oil_AU', 'std_DD_oil_AU', 'std_DI_oil_AU', 'std_netD_oil_AU', 'std_netS_oil_AU', 'std_SD_silver_AU', 'std_SI_silver_AU', 'std_DD_silver_AU', 'std_DI_silver_AU', 'std_netD_silver_AU', 'std_netS_silver_AU', 'std_SD_soybean_AU', 'std_SI_soybean_AU', 'std_DD_soybean_AU', 'std_DI_soybean_AU', 'std_netD_soybean_AU', 'std_netS_soybean_AU', 'std_SD_sugar_AU', 'std_SI_sugar_AU', 'std_DD_sugar_AU', 'std_DI_sugar_AU', 'std_netD_sugar_AU', 'std_netS_sugar_AU', 'std_SD_wheat_AU', 'std_SI_wheat_AU', 'std_DD_wheat_AU', 'std_DI_wheat_AU', 'std_netD_wheat_AU', 'std_netS_wheat_AU', 'std_SD_zinc_AU', 'std_SI_zinc_AU', 'std_DD_zinc_AU', 'std_DI_zinc_AU', 'std_netD_zinc_AU', 'std_netS_zinc_AU']
            controls = ['.DXY (TRDPRC_1)', 'Mich', 'Employment_cost', 'NFIB', 'Fed_total_asset', 'import_prices', 'MOODCAAA Index', 'MOODCBAA Index', 'US oil MCRSTUS1', 'OPEC oil production', 'World oil production', 'proxy OECD crude oil inventories', 'Output gap', 'Filly Fed', 'SCFI', 'Wu Xia Shadow Rate', 'Fed Effective funds rate', 'Wu Xia Spliced Policy Rate', 'SnP500', 'USCPIZ1Y', 'USCPIZ2Y', 'USCPIZ5Y', 'USCPIZ10Y', 'Employment absolute change', 'govt_expenditures', 'job_vacancies', 'fed M2', 'PMI Survey', 'GSCI_CrudeOil', 'GSCI_BrentCrude', 'GSCI_UnleadedGasoline', 'GSCI_HeatingOil', 'GSCI_Gasoil', 'GSCI_NaturalGas', 'GSCI_Aluminum', 'GSCI_Copper', 'GSCI_Lead', 'GSCI_Nickel', 'GSCI_Zinc', 'GSCI_Gold', 'GSCI_Silver', 'GSCI_WheatCBOT', 'GSCI_WheatKansas', 'GSCI_Corn', 'GSCI_Soybeans', 'GSCI_Cotton', 'GSCI_Sugar', 'GSCI_Coffee', 'GSCI_Cocoa', 'GSCI_LiveCattle', 'GSCI_FeederCattle', 'GSCI_LeanHogs', 'GSCI_Composite', 'Tbill10Y', 'Tbill1mo', 'VIX']
            narratives = [n for n in narratives_and_composite if "composite" not in n]
            narratives_net_only = [n for n in narratives if all(x not in n for x in ["SI", "SD", "DI", "DD"])]
            narratives_ID_only = [n for n in narratives if any(x in n for x in ["SI", "SD", "DI", "DD"])]
            BM_composite = [n for n in narratives_and_composite if "composite" in n]
            BM_composite_net_only = [n for n in BM_composite if all(x not in n for x in ["SI", "SD", "DI", "DD"])]
            all_features = narratives + controls

            inputlist = {
                "narratives_and_composite": narratives_and_composite,
                "controls": controls,
                "narratives": narratives,
                "narratives_net_only": narratives_net_only,
                "narratives_ID_only": narratives_ID_only,
                "BM_composite": BM_composite,
                "BM_composite_net_only": BM_composite_net_only,
                "all_features": all_features
            }
            return inputlist

        inputlist = create_inputlist()
        if profile not in inputlist:
            print(f"'{profile}' not found in inputlist.\nSelect one of the following:")
            print(", ".join(inputlist.keys()))
        else:
            self.profilename = profile
            self.profile = inputlist[profile]

###########################################################################################################
### Helper and data functions #############################################################################

    def compute_window_end(self):
        end_date = self.startdate + pd.DateOffset(months=self.window - 1)
        return end_date
    
    def df_to_use(self):
        """
        Filters X on:
        - Columns set by self.profile
        - Number of lags defined by self.maxlag
        - dates from self.startdate up to the last date of the window"""
        if self.profile is None:
            print("Profile not set. Use set_profile() to define the profile.")
        df = self.df.copy()
        df = df.set_index('date')
        
        # Keep only columns that contain any of the substrings in self.profile
        allowed_lags = [f"_lag{i}" for i in range(self.maxlag + 1)]
        
        # Filter columns that match both profile substring and allowed lag suffix
        filtered_columns = [
            col for col in df.columns
            if any(profile_key in col for profile_key in self.profile)
            and any(col.endswith(lag) for lag in allowed_lags)
        ]
        df = df[filtered_columns]
        
        df.index = pd.to_datetime(df.index)
        window_end = self.compute_window_end()
        df = df.loc[(df.index >= self.startdate) & (df.index <= window_end)]
        return df

    def y_to_use(self, df):
        # Ensure self.y has a proper datetime index
        if self.y.index.dtype == 'object':
            self.y.index = pd.to_datetime(self.y.index)

        #print(self.y.index.equals(df.index))  # should now be True if values match
        #print(self.y.index, df.index)
        #print(self.y.index.name, df.index.name)
        y_filtered = self.y.loc[df.index]
        return y_filtered

    def preprocess_data(self, X, y, random_state=42, return_test_dates=False, date_column=None):
        """
        Preprocesses data for regression models. Handles test_size=0 by skipping test split.
        Returns:
        - X_train_scaled, X_test_scaled: standardized numeric features (X_test may be None)
        - y_train, y_test: target variables (y_test may be None)
        - feature_names: names of features after variance thresholding
        - test_dates (optional)
        """
        scaler = self.scaler
        test_size = self.testsize
        if test_size > 0:
            tscv = TimeSeriesSplit(n_splits=int(1 / test_size))
            train_index, test_index = list(tscv.split(X))[-1]
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            #print(test_index)
            X_train = X_train.loc[y_train.notna()]
            y_train = y_train.dropna()
            X_test = X_test.loc[y_test.notna()]
            y_test = y_test.dropna()
        else:
            X_train, y_train = X.copy(), y.copy()
            X_test, y_test = None, None

        if return_test_dates and date_column and test_size > 0:
            test_dates = X_test[date_column].values
        else:
            test_dates = None

        numeric_columns = X_train.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        X_train_numeric = X_train[numeric_columns].fillna(0)
        X_test_numeric = X_test[numeric_columns].fillna(0) if test_size > 0 else None

        # Fit selector on training data
        selector = VarianceThreshold(threshold=0.0)
        X_train_reduced = selector.fit_transform(X_train_numeric)
        X_test_reduced = selector.transform(X_test_numeric) if X_test_numeric is not None else None
        selected_columns = numeric_columns[selector.get_support()]
        dropped_columns = numeric_columns[~selector.get_support()]

        # Print dropped columns
        if len(dropped_columns) > 0:
            print(f"🧹 Dropped {len(dropped_columns)} low-variance columns:")
            print(dropped_columns.tolist())
        else:
            print("✅ No columns dropped by variance thresholding.")

        if self.scaler == "Standard":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_reduced)
            X_test_scaled = scaler.transform(X_test_reduced) if X_test_reduced is not None else None
        elif self.scaler in ["zca", "ZCA"]:
            X_train_scaled = self.zca_whiten(X_train_reduced)
            X_test_scaled = self.zca_whiten(X_test_reduced)

        if return_test_dates:
            return X_train_scaled, X_test_scaled, y_train, y_test, selected_columns, test_dates
        #print("processed_data:\n\n\n", "\n\n\n" ,X_train_scaled, "\n\n\n", y_train)
        return X_train_scaled, X_test_scaled, y_train, y_test, selected_columns

    def zca_whiten(self, X):
        """
        Applies ZCA whitening (preserves original dimensions, unlike PCA).
        Assumes X is already standardized or will standardize internally.
        Returns: whitened data (same shape as X)
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # Eigen decomposition (or use SVD if preferred)
        U, S, _ = np.linalg.svd(cov)

        # Construct ZCA whitening matrix
        epsilon = 1e-5  # for numerical stability
        ZCA_matrix = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T

        # Whiten
        X_zca = X_centered @ ZCA_matrix
        return X_zca


    def save_model_package(self, package, folder, filename):
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        joblib.dump(package, save_path)
        #print(f"Model package saved to '{save_path}'.")
        return save_path

###########################################################################################################
### Launcher functions  ###################################################################################

    def run_model(self, random_state=42, cv=10, save_to_dir=None):
        
        X = self.df_to_use()
        y = self.y_to_use(X)
        #print("input:\n\n\n", X.columns, "\n\n\n" ,X.head(), "\n\n\n", y.head())
        
        if len(X) != len(y):
            print(f"Dimension mismatch: X has {len(X)} rows, y has {len(y)} rows.")

        model_output = {}

        if self.modeltype == 'lassocv':
            model, alpha, mse, feature_names, non_zero_coefs= self.lasso_cv(X, y, random_state, cv)
            model_output = {
                'model': model,
                'alpha': alpha,
                'mse': mse,
                'feature_names': feature_names,
                'non_zero_coefs':non_zero_coefs
            }
        
        elif self.modeltype == 'lassobic':
            model, alpha, mse, feature_names, non_zero_coefs= self.lasso_bic(X, y, random_state)
            model_output = {
                'model': model,
                'alpha': alpha,
                'mse': mse,
                'feature_names': feature_names, 
                'non_zero_coefs':non_zero_coefs
            }

        elif self.modeltype == 'elasticnetcv':
            model, alpha, l1_ratio, mse, feature_names, non_zero_coefs= self.elastic_net_cv(X, y, random_state, cv)
            model_output = {
                'model': model,
                'alpha': alpha,
                'l1_ratio': l1_ratio,
                'mse': mse,
                'feature_names': feature_names,
                'non_zero_coefs':non_zero_coefs
            }

        elif self.modeltype == 'elasticnetbic':
            print("error: elasticnetbic not implemented yet")
            #model, alpha, l1_ratio, mse, non_zero_coefs= self.elastic_net_bic(X, y, random_state, cv)
            model_output = {
                'model': model,
                'alpha': alpha,
                'l1_ratio': l1_ratio,
                'mse': mse,
                'non_zero_coefs':non_zero_coefs
            }


        elif self.modeltype == 'linear':
            model, mse, feature_names, non_zero_coefs= self.autoregression_model(X, y, random_state)
            model_output = {
                'model': model,
                'mse': mse,
                'feature_names': feature_names,
                'non_zero_coefs':non_zero_coefs
            }

        else:
            print("Model type not set or invalid. Use set_modeltype() to define the model type.")
            return None

        if model_output["non_zero_coefs"].empty:
            print("Warning: Model has no non-zero coefficients. Check your data and model settings.")

        # Calculate end of window and test end date
        window_end = self.compute_window_end()
        last_train_date = None
        if self.testsize > 0:
            tscv = TimeSeriesSplit(n_splits=int(1 / self.testsize))
            train_index, test_index = list(tscv.split(X))[-1]
            last_train_date = X.iloc[train_index].index[-1] if len(train_index) > 0 else None
            print(f"Last train date: {last_train_date}", f"self.testsize: {self.testsize}")

        # Add additional info to model_output
        model_output['startdate'] = self.startdate
        model_output['window_end'] = window_end
        model_output['last_train_date'] = last_train_date
        model_output['scaler'] = self.scaler

        print(f"{self.profilename} {self.modeltype} model fitted from {self.startdate.date()} to {window_end.date()} with lag = {self.maxlag} months and lead = {self.lead} months.")
        if save_to_dir is not None:
            if not isinstance(save_to_dir, list) or len(save_to_dir) != 2:
                raise ValueError('"save_to_dir" must be a list of [folder, filename]')
            folder, filename = save_to_dir
            self.save_model_package(model_output, folder, filename)
            print(f"Model saved to {folder}/{filename}.")
        return model_output

    def backtest_fixed_window_with_refit(self):

        # Prepare output folder
        runtime = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.modeltype}_{self.profilename}_lag{self.maxlag}_lead{self.lead}_{runtime}"
        os.makedirs(folder_name, exist_ok=True)

        # Prepare for iteration
        coef_matrix = []
        date_matrix = []

        # Full feature set for column alignment
        feature_columns = self.df_to_use().columns
        print("Starting compilation. Number of features to select from", len(feature_columns))

        print("self.startdate:", self.startdate)
        print("Available dates in DataFrame:", self.df['date'].unique())
        
        # Get all available dates sorted
        all_dates = pd.to_datetime(pd.Series(self.df['date'].unique())).sort_values().reset_index(drop=True)
        all_dates = pd.DatetimeIndex(all_dates)


        start_idx = all_dates.searchsorted(self.startdate)
        window_months = self.window

        # Iterate over rolling windows
        while True:
            window_start = all_dates[start_idx]
            window_end = window_start + pd.DateOffset(months=window_months)

            # Stop if window_end exceeds available data
            if window_end > all_dates[-1]:
                break

            # Set model window and run model
            self.set_startdate(str(window_start.date()))
            model_result = self.run_model()

            # Save model object to file
            model_filename = f"{window_start.date()}_{window_end.date()}.pkl"
            self.save_model_package(model_result, folder_name, model_filename)


            # Extract model_output and get coefficients
            #model_output = model_result['model_output']
            non_zero_coefs= model_result["non_zero_coefs"]  # <- access by key, not index

            # Ensure Series and align with full feature set
            if isinstance(non_zero_coefs, np.ndarray):
                non_zero_coefs= pd.Series(non_zero_coefs, index=feature_columns[:len(non_zero_coefs)])
            non_zero_coefs=non_zero_coefs.reindex(feature_columns, fill_value=0)

            coef_matrix.append(non_zero_coefs)
            date_matrix.append(window_start.date())

            # Advance the rolling window
            start_idx += 1
            if start_idx + window_months > len(all_dates):
                break

        # Final DataFrame with consistent columns
        coef_df = pd.DataFrame(coef_matrix, index=date_matrix)
        coef_df.index.name = "window_start"
        coef_df.to_csv(os.path.join(folder_name, "coefficient_matrix.csv"))

        print(f"Backtest complete. Results saved in {folder_name}")

###########################################################################################################
### Model implementations  ################################################################################

    def autoregression_model(self, X, y, random_state=42):
        X_train, X_test, y_train, y_test, feature_names = self.preprocess_data(X, y, random_state)
        
        test_size = self.testsize
        # Train OLS model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Coefficients
        coef = pd.Series(model.coef_, index=feature_names)
        non_zero_coefs= coef[coef != 0]

        if self.auto_print:
            print(f"Number of Non-zero coefficients: {non_zero_coefs.shape[0]}")
            print(f"Test MSE: {mse:.6f}")
            print(f"Non-zero coefficients:\n{non_zero_coefs.reindex(non_zero_coefs.abs().sort_values(ascending=False).index)}\n")

        return model, mse, feature_names, non_zero_coefs
        
    def lasso_cv(self, X, y, random_state=42, cv=5):

        X_train, X_test, y_train, y_test, feature_names = self.preprocess_data(X, y, random_state)
        test_size = self.testsize
        y_train.plot()
        model = LassoCV(cv=TimeSeriesSplit(n_splits=cv), random_state=random_state, max_iter=100000)
        model.fit(X_train, y_train)

        best_alpha = model.alpha_
        coef = pd.Series(model.coef_, index=feature_names)
        non_zero_coefs= coef[coef != 0]

        mse = None
        if X_test is not None and len(X_test) > 0:
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

        if self.auto_print:
            print(f"Number of Non-zero coefficients: {non_zero_coefs.shape[0]}")
            print(f"Optimal alpha: {best_alpha}")
            print(f"Test MSE: {mse:.6f}" if mse is not None else "Test MSE: Not computed (no test set)")
            print(f"Non-zero coefficients:\n{non_zero_coefs.reindex(non_zero_coefs.abs().sort_values(ascending=False).index)}\n")

        return model, best_alpha, mse, feature_names, non_zero_coefs

    def lasso_bic(self, X, y, random_state=42):
        """
        Fits a Lasso model using LARS path and selects alpha via BIC.
        Returns the model, selected alpha, test MSE (if applicable), and non-zero coefficients.
        """

        # Preprocess data
        X_train, X_test, y_train, y_test, feature_names = self.preprocess_data(X, y, random_state)
        test_size = self.testsize

        # Fit Lasso using BIC to choose alpha
        model = LassoLarsIC(criterion='bic')
        model.fit(X_train, y_train)

        best_alpha = model.alpha_
        coef = pd.Series(model.coef_, index=feature_names)
        non_zero_coefs= coef[coef != 0]

        # Evaluate performance if test data exists
        mse = None
        if X_test is not None and len(X_test) > 0:
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

        # Optional printing
        if self.auto_print:
            print(f"Number of Non-zero coefficients: {non_zero_coefs.shape[0]}")
            print(f"Optimal alpha (BIC): {best_alpha}")
            print(f"Test MSE: {mse:.6f}" if mse is not None else "Test MSE: Not computed (no test set)")
            print(f"Non-zero coefficients:\n{non_zero_coefs.reindex(non_zero_coefs.abs().sort_values(ascending=False).index)}\n")

        return model, best_alpha, mse, feature_names, non_zero_coefs
        
    def elastic_net_cv(self, X, y, random_state=42, cv=5, l1_ratio=None):
        if l1_ratio is None:
            l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        test_size = self.testsize
        X_train, X_test, y_train, y_test, feature_names = self.preprocess_data(X, y, random_state)

        model = ElasticNetCV(
            l1_ratio=l1_ratio,
            alphas=None,
            cv=TimeSeriesSplit(n_splits=cv),
            random_state=random_state,
            max_iter=100000,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        best_alpha = model.alpha_
        best_l1 = model.l1_ratio_
        coef = pd.Series(model.coef_, index=feature_names)
        non_zero_coefs= coef[coef != 0]

        mse = None
        if X_test is not None and len(X_test) > 0:
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

        if self.auto_print:
            print(f"Optimal alpha: {best_alpha}")
            print(f"Optimal l1_ratio: {best_l1}")
            print(f"Number of Non-zero coefficients: {non_zero_coefs.shape[0]}")
            print(f"Test MSE: {mse:.6f}" if mse is not None else "Test MSE: Not computed (no test set)")
            print(f"Non-zero coefficients:\n{non_zero_coefs.reindex(non_zero_coefs.abs().sort_values(ascending=False).index)}\n")

        return model, best_alpha, best_l1, mse, feature_names, non_zero_coefs

