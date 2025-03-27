import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
import time
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

# Import XGBoost if available
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available. Installing recommended for best performance.")

# Try to import seaborn for better visualizations
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class EnhancedPowerPredictor:
    def __init__(self, input_path, output_path, output_dir=None, 
                 use_xgboost=True, low_memory_mode=True):
        """Initialize with file paths and parameters"""
        self.input_path = input_path
        self.output_path = output_path
        self.low_memory_mode = low_memory_mode
        self.use_xgboost = use_xgboost and HAS_XGB
            
        # Create output directory with timestamp
        if output_dir is None:
            self.output_dir = f"power_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.output_dir = output_dir
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Initialize components
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.models = {}            # All trained models
        self.station_groups = []    # Groups of similar stations
        self.group_models = {}      # Models by station group
        self.feature_selection = False
        
        # Configuration settings - improved defaults
        self.config = {
            'lag_features': True,
            'use_station_clustering': True,
            'n_station_clusters': 4,  # Number of station groups to create
            'create_ensembles': True,
            'optimize_hyperparams': True,
            'feature_selection': True,
            'n_components': 300,      # SVD components
            'n_lags': 8,              # Increased lag features
            'rolling_windows': [3, 7, 14, 30],  # Multiple windows for better pattern capture
            'feature_selection_threshold': 0.80,  # Keep features that contribute to 80% importance
            'min_features': 20,        # Minimum number of features to keep
            'forecast_horizon': 1,     # Steps ahead to forecast
            'use_fourier_features': True,  # Add Fourier features for time series
            'fourier_periods': [7, 14, 30],  # Common periods in days
            'xgb_params': {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'hist'
            }
        }
        
        # Set up logging
        self.log_file = f"{self.output_dir}/log.txt"
        with open(self.log_file, "w") as f:
            f.write(f"Enhanced Power Predictor - Run started at {datetime.now()}\n\n")
        
    def log(self, message):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
    
    def load_data(self):
        """Load and explore the datasets"""
        self.log("Loading input and output data...")
        
        # Load data efficiently
        if self.low_memory_mode:
            try:
                chunks = []
                for chunk in pd.read_csv(self.input_path, chunksize=1000):
                    chunks.append(chunk)
                self.input_data = pd.concat(chunks)
                del chunks
                gc.collect()
            except:
                self.input_data = pd.read_csv(self.input_path)
        else:
            self.input_data = pd.read_csv(self.input_path)
            
        self.output_data = pd.read_csv(self.output_path)
        
        # Log basic info
        self.log(f"Input data shape: {self.input_data.shape}")
        self.log(f"Output data shape: {self.output_data.shape}")
        
        # Visualize and analyze data
        self.analyze_data()
    
    def analyze_data(self):
        """Analyze data characteristics and create visualizations"""
        self.log("Analyzing data patterns...")
        
        # Create a directory for analysis plots
        analysis_dir = f"{self.output_dir}/data_analysis"
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
            
        # Plot raw output patterns for sample stations
        self.plot_raw_outputs(analysis_dir)
            
        # Check for time-based patterns
        self.analyze_temporal_patterns(analysis_dir)
        
        # Analyze correlation between power stations
        self.analyze_station_correlations(analysis_dir)
        
        # Track output statistics
        self.calculate_output_statistics(analysis_dir)
        
        # Cluster stations into groups
        if self.config['use_station_clustering']:
            self.cluster_stations()
    
    def plot_raw_outputs(self, output_dir):
        """Plot raw output patterns"""
        self.log("Plotting power output patterns...")
        
        # Plot first few outputs
        plt.figure(figsize=(15, 10))
        for i in range(min(5, self.output_data.shape[1])):
            plt.subplot(5, 1, i+1)
            plt.plot(self.output_data.iloc[:, i])
            plt.title(f'Power Station {i+1} Raw Values')
            plt.xlabel('Time')
            plt.ylabel('Power Output')
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(f'{output_dir}/raw_outputs.png')
        plt.close()
    
    def analyze_temporal_patterns(self, output_dir):
        """Analyze temporal patterns in the data"""
        self.log("Analyzing temporal patterns...")
        
        # Calculate rolling statistics
        rolling_mean = self.output_data.rolling(window=14).mean()
        rolling_std = self.output_data.rolling(window=14).std()
        
        # Plot for a few stations
        for i in range(min(3, self.output_data.shape[1])):
            plt.figure(figsize=(12, 6))
            plt.plot(self.output_data.iloc[:, i], label='Original', alpha=0.7)
            plt.plot(rolling_mean.iloc[:, i], label='14-day Rolling Mean', linewidth=2)
            plt.plot(rolling_std.iloc[:, i], label='14-day Rolling Std', linewidth=2)
            plt.legend()
            plt.title(f'Temporal Patterns for Power Station {i+1}')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{output_dir}/temporal_station_{i+1}.png')
            plt.close()
        
        # Check for autocorrelation
        try:
            from pandas.plotting import autocorrelation_plot
            plt.figure(figsize=(12, 6))
            for i in range(min(3, self.output_data.shape[1])):
                autocorrelation_plot(self.output_data.iloc[:, i])
            plt.title('Autocorrelation Plot for First 3 Stations')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{output_dir}/autocorrelation.png')
            plt.close()
        except:
            self.log("Skipping autocorrelation plot - pandas plotting not available")
    
    def analyze_station_correlations(self, output_dir):
        """Analyze correlations between power stations"""
        self.log("Analyzing station correlations...")
        
        # Calculate correlation matrix
        corr_matrix = self.output_data.corr()
        
        # Plot heatmap (using seaborn if available)
        try:
            plt.figure(figsize=(12, 10))
            if HAS_SEABORN:
                sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                            annot=False, square=True)
            else:
                plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
                plt.colorbar()
            plt.title('Correlation Between Power Stations')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/station_correlation.png')
            plt.close()
        except Exception as e:
            self.log(f"Error creating correlation heatmap: {str(e)}")
    
    def calculate_output_statistics(self, output_dir):
        """Calculate and save statistics about outputs"""
        self.log("Calculating output statistics...")
        
        # Calculate basic statistics for each station
        stats = {
            'mean': self.output_data.mean(),
            'std': self.output_data.std(),
            'min': self.output_data.min(),
            'max': self.output_data.max(),
            'range': self.output_data.max() - self.output_data.min(),
            'zeros': (self.output_data == 0).sum(),
            'missing': self.output_data.isna().sum()
        }
        
        # Create a DataFrame with the stats
        stats_df = pd.DataFrame(stats)
        
        # Save to CSV
        stats_df.to_csv(f"{output_dir}/output_statistics.csv")
        
        # Log summary information
        self.log(f"Output range: [{stats_df['min'].min():.2f} to {stats_df['max'].max():.2f}]")
        self.log(f"Average output: {stats_df['mean'].mean():.2f}")
        self.log(f"Average standard deviation: {stats_df['std'].mean():.2f}")
        
        # Return important statistics for later use
        return stats_df
    
    def cluster_stations(self):
        """Group similar stations using clustering"""
        self.log("Clustering stations into groups...")
        
        # Get station statistics as features
        station_features = []
        for i in range(self.output_data.shape[1]):
            station_data = self.output_data.iloc[:, i]
            features = [
                station_data.mean(),
                station_data.std(),
                station_data.min(),
                station_data.max(),
                station_data.skew(),
                station_data.kurtosis()
            ]
            # Add autocorrelation features
            for lag in [1, 7, 14]:
                features.append(station_data.autocorr(lag=lag))
            station_features.append(features)
            
        # Convert to array
        station_features = np.array(station_features)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(station_features)
        
        # Determine optimal number of clusters using silhouette score
        n_clusters = min(self.config['n_station_clusters'], self.output_data.shape[1] // 3)
        n_clusters = max(3, n_clusters)  # At least 3 clusters
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Group stations by cluster
        self.station_groups = [[] for _ in range(n_clusters)]
        for i, cluster_id in enumerate(clusters):
            self.station_groups[cluster_id].append(i)
            
        # Log cluster sizes
        self.log(f"Created {n_clusters} station groups:")
        for i, group in enumerate(self.station_groups):
            self.log(f"  Group {i+1}: {len(group)} stations")
            
        # Save station groups
        with open(f"{self.output_dir}/station_groups.txt", "w") as f:
            for i, group in enumerate(self.station_groups):
                f.write(f"Group {i+1}: Stations {[j+1 for j in group]}\n")
                
        # Store cluster assignments for each station
        self.station_clusters = clusters
        
    def preprocess_data(self):
        """Preprocess data with efficient dimensionality reduction"""
        self.log("Preprocessing data...")
        
        # Scale inputs and outputs
        self.log("Scaling input and output data...")
        self.input_scaled = self.input_scaler.fit_transform(self.input_data)
        self.output_scaled = self.output_scaler.fit_transform(self.output_data)
        
        # Free up memory
        del self.input_data
        gc.collect()
        
        # Dimension reduction via Singular Value Decomposition
        from sklearn.decomposition import TruncatedSVD
        
        n_components = min(self.config['n_components'], self.input_scaled.shape[0]-1, self.input_scaled.shape[1])
        self.log(f"Performing SVD with {n_components} components...")
        
        self.svd = TruncatedSVD(n_components=n_components)
        self.input_reduced = self.svd.fit_transform(self.input_scaled)
        
        explained_var = self.svd.explained_variance_ratio_.sum()
        self.log(f"Explained variance: {explained_var:.4f}")
        
        # Free up more memory
        del self.input_scaled
        gc.collect()
        
        # Create a DataFrame for easier manipulation
        self.output_df = pd.DataFrame(self.output_scaled)
        
        # Create comprehensive forecast features
        self.create_forecast_features()
        
    def create_forecast_features(self):
        """Create comprehensive features for forecasting"""
        self.log("Creating advanced forecast features...")
        
        # Base features from SVD
        feature_sets = [self.input_reduced]
        feature_names = [f'svd_{i}' for i in range(self.input_reduced.shape[1])]
        
        # Create time index
        time_index = np.arange(len(self.output_scaled))
        
        if self.config['lag_features']:
            # Add lagged outputs (critical for forecasting)
            for lag in range(1, self.config['n_lags'] + 1):
                output_lagged = np.roll(self.output_scaled, lag, axis=0)
                output_lagged[:lag, :] = 0  # Clear invalid values
                
                feature_sets.append(output_lagged)
                feature_names.extend([f'lag{lag}_output{i}' for i in range(output_lagged.shape[1])])
            
            # Create rolling statistics from output with multiple windows
            for window in self.config['rolling_windows']:
                # Rolling mean captures trend
                rolling_mean = self.output_df.rolling(window=window, min_periods=1).mean().values
                feature_sets.append(rolling_mean)
                feature_names.extend([f'rollmean{window}_output{i}' for i in range(rolling_mean.shape[1])])
                
                # Rolling standard deviation captures volatility
                rolling_std = self.output_df.rolling(window=window, min_periods=1).std().fillna(0).values
                feature_sets.append(rolling_std)
                feature_names.extend([f'rollstd{window}_output{i}' for i in range(rolling_std.shape[1])])
                
                # Rolling min/max to capture range
                rolling_min = self.output_df.rolling(window=window, min_periods=1).min().values
                feature_sets.append(rolling_min)
                feature_names.extend([f'rollmin{window}_output{i}' for i in range(rolling_min.shape[1])])
                
                rolling_max = self.output_df.rolling(window=window, min_periods=1).max().values
                feature_sets.append(rolling_max)
                feature_names.extend([f'rollmax{window}_output{i}' for i in range(rolling_max.shape[1])])
            
            # Add first, second and third-order differences 
            for order in range(1, 4):
                diff = self.output_df.diff(order).fillna(0).values
                feature_sets.append(diff)
                feature_names.extend([f'diff{order}_output{i}' for i in range(diff.shape[1])])
            
            # Exponential weighted mean with multiple spans
            for span in [5, 10, 20]:
                ewm_mean = self.output_df.ewm(span=span).mean().values
                feature_sets.append(ewm_mean)
                feature_names.extend([f'ewm{span}_output{i}' for i in range(ewm_mean.shape[1])])
            
            # Add interaction features between nearby time periods
            interaction = self.output_df.shift(1).values * self.output_df.shift(2).values
            interaction[np.isnan(interaction)] = 0
            feature_sets.append(interaction)
            feature_names.extend([f'interaction_output{i}' for i in range(interaction.shape[1])])
        
        # Add Fourier features for cyclical patterns if enabled
        if self.config['use_fourier_features']:
            for period in self.config['fourier_periods']:
                # Convert to actual timesteps
                p = period * 24  # Assuming hourly data
                
                # Add sine and cosine features
                sin_feat = np.sin(2 * np.pi * time_index / p).reshape(-1, 1)
                cos_feat = np.cos(2 * np.pi * time_index / p).reshape(-1, 1)
                
                feature_sets.append(sin_feat)
                feature_sets.append(cos_feat)
                
                feature_names.append(f'sin_period{period}')
                feature_names.append(f'cos_period{period}')
        
        # Combine all features
        self.forecast_features = np.hstack(feature_sets)
        self.feature_names = feature_names
        
        self.log(f"Final feature set: {self.forecast_features.shape[1]} features")
        
        # Create time shift for target (forecast horizon)
        if self.config['forecast_horizon'] > 1:
            # Shift targets by forecast horizon
            self.output_shifted = np.roll(self.output_scaled, 
                                          -self.config['forecast_horizon'], 
                                          axis=0)
            # Clear invalid values at end
            self.output_shifted[-self.config['forecast_horizon']:, :] = 0
        else:
            self.output_shifted = self.output_scaled
    
    def split_data(self, test_size=0.15, val_size=0.15):
        """Split data into train, validation and test sets"""
        self.log("Splitting data with time-aware approach...")
        
        # Use order-preserving split for time series
        total_samples = len(self.forecast_features)
        
        # Remove the first few rows that have zero-padding from lag creation
        if self.config['lag_features']:
            start_idx = self.config['n_lags']
        else:
            start_idx = 0
            
        # Remove samples at the end if using forecast_horizon > 1
        if self.config['forecast_horizon'] > 1:
            end_idx = total_samples - self.config['forecast_horizon']
        else:
            end_idx = total_samples
            
        # Adjust indices to account for removed samples
        adjusted_samples = end_idx - start_idx
        
        # Calculate split points
        train_end = start_idx + int(adjusted_samples * (1 - test_size - val_size))
        val_end = start_idx + int(adjusted_samples * (1 - test_size))
        
        # Training data
        self.X_train = self.forecast_features[start_idx:train_end]
        self.y_train = self.output_shifted[start_idx:train_end]
        
        # Validation data
        self.X_val = self.forecast_features[train_end:val_end]
        self.y_val = self.output_shifted[train_end:val_end]
        
        # Test data
        self.X_test = self.forecast_features[val_end:end_idx]
        self.y_test = self.output_shifted[val_end:end_idx]
        
        self.log(f"Data split: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
        
    def train_baseline_models(self):
        """Train multiple baseline models for comparison"""
        self.log("Training baseline models...")
        
        # Dictionary to store each model type's performance
        results = {}
        
        # Models to try
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=None, 
                min_samples_leaf=1,
                random_state=42, 
                n_jobs=1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=3,
                random_state=42
            )
        }
        
        # Add XGBoost if available
        if self.use_xgboost:
            params = self.config['xgb_params']
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                tree_method=params.get('tree_method', 'hist'),
                random_state=42
            )
        
        # We'll train separate models for each output column
        for name, model_class in models.items():
            self.log(f"Training {name} baseline models...")
            start_time = time.time()
            
            # Create containers for models and scores
            output_models = {}
            val_r2_scores = []
            
            # Train one model for each output column
            for i in range(self.y_train.shape[1]):
                # Create a fresh model instance
                if name == 'LinearRegression':
                    model = LinearRegression()
                elif name == 'Ridge':
                    model = Ridge(alpha=1.0)
                elif name == 'Lasso':
                    model = Lasso(alpha=0.1)
                elif name == 'RandomForest':
                    model = RandomForestRegressor(
                        n_estimators=100, 
                        max_depth=None, 
                        min_samples_leaf=1,
                        random_state=42,
                        n_jobs=1
                    )
                elif name == 'GradientBoosting':
                    model = GradientBoostingRegressor(
                        n_estimators=100, 
                        learning_rate=0.1, 
                        max_depth=3,
                        random_state=42
                    )
                elif name == 'XGBoost':
                    params = self.config['xgb_params']
                    model = xgb.XGBRegressor(
                        n_estimators=params.get('n_estimators', 100),
                        learning_rate=params.get('learning_rate', 0.1),
                        max_depth=params.get('max_depth', 3),
                        subsample=params.get('subsample', 0.8),
                        colsample_bytree=params.get('colsample_bytree', 0.8),
                        tree_method=params.get('tree_method', 'hist'),
                        random_state=42
                    )
                
                # Train model for this output column
                model.fit(self.X_train, self.y_train[:, i])
                
                # Evaluate on validation set
                val_preds = model.predict(self.X_val)
                val_r2 = r2_score(self.y_val[:, i], val_preds)
                val_r2_scores.append(val_r2)
                
                # Store model
                output_models[i] = model
                
                # Log progress for large datasets
                if (i+1) % 10 == 0:
                    self.log(f"  {name}: Trained {i+1}/{self.y_train.shape[1]} models")
                
            # Calculate average performance
            avg_val_r2 = np.mean(val_r2_scores)
            results[name] = {
                'avg_val_r2': avg_val_r2,
                'models': output_models,
                'r2_scores': val_r2_scores
            }
            
            duration = time.time() - start_time
            self.log(f"  {name}: Average validation R²: {avg_val_r2:.4f}, Time: {duration:.1f}s")
            
        # Find best model type based on average R²
        best_model = max(results.items(), key=lambda x: x[1]['avg_val_r2'])
        self.best_model_type = best_model[0]
        self.models = results
        
        self.log(f"Best baseline model: {self.best_model_type} with R²: {best_model[1]['avg_val_r2']:.4f}")
        
        # Analyze per-station performance
        self.analyze_baseline_performance()
    
    def analyze_baseline_performance(self):
        """Analyze performance of baseline models across stations"""
        self.log("Analyzing baseline model performance...")
        
        # Create directory for analysis
        analysis_dir = f"{self.output_dir}/model_analysis"
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
            
        # Extract per-station performance for each model
        model_names = list(self.models.keys())
        station_r2 = np.zeros((len(model_names), self.y_val.shape[1]))
        
        for i, name in enumerate(model_names):
            station_r2[i, :] = self.models[name]['r2_scores']
            
        # Find best model for each station
        best_model_per_station = np.argmax(station_r2, axis=0)
        station_best_models = {}
        
        for i, model_idx in enumerate(best_model_per_station):
            model_name = model_names[model_idx]
            if model_name not in station_best_models:
                station_best_models[model_name] = []
            station_best_models[model_name].append(i)
            
        # Log best model distribution
        self.log("Best model distribution across stations:")
        for name, stations in station_best_models.items():
            self.log(f"  {name}: {len(stations)} stations ({len(stations)/self.y_val.shape[1]*100:.1f}%)")
            
        # Plot comparison of model performance by station
        plt.figure(figsize=(12, 8))
        for i, name in enumerate(model_names):
            plt.plot(station_r2[i], 'o-', label=name, alpha=0.7)
            
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Power Station')
        plt.ylabel('Validation R²')
        plt.title('Model Performance by Power Station')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{analysis_dir}/model_comparison_by_station.png')
        plt.close()
        
        # Save best model assignments
        self.station_best_models = station_best_models
        
        # Create heatmap of model performance
        if HAS_SEABORN:
            plt.figure(figsize=(12, 8))
            sns.heatmap(station_r2, cmap='RdYlGn', yticklabels=model_names)
            plt.title('R² Values by Model and Station')
            plt.xlabel('Power Station')
            plt.ylabel('Model')
            plt.tight_layout()
            plt.savefig(f'{analysis_dir}/performance_heatmap.png')
            plt.close()
    
    def select_important_features(self):
        """Select important features based on the best model"""
        if not self.config['feature_selection']:
            self.log("Feature selection disabled, skipping...")
            return
        
        self.log("Selecting important features...")
        
        # Only supported for tree-based models
        if self.best_model_type not in ['RandomForest', 'GradientBoosting', 'XGBoost']:
            self.log("Feature selection requires tree-based models, using RandomForest instead...")
            # Use RandomForest for feature selection if best model isn't suitable
            if 'RandomForest' in self.models:
                feature_model = list(self.models['RandomForest']['models'].values())[0]
            else:
                # Create a RandomForest model specifically for feature selection
                feature_model = RandomForestRegressor(n_estimators=100, random_state=42)
                feature_model.fit(self.X_train, self.y_train[:, 0])
        else:
            # Use the best model for feature selection
            feature_model = list(self.models[self.best_model_type]['models'].values())[0]
        
        # Get feature importances
        if self.best_model_type == 'XGBoost':
            importances = feature_model.feature_importances_
        else:
            importances = feature_model.feature_importances_
            
        # Get indices sorted by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(14, 8))
        plt.title(f'Feature Importances ({self.best_model_type})')
        plt.bar(range(min(30, len(importances))), 
                importances[indices[:30]], align='center')
        plt.xticks(range(min(30, len(importances))), 
                  [self.feature_names[i] for i in indices[:30]], rotation=90)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png')
        plt.close()
        
        # Calculate cumulative importance
        sorted_importances = np.sort(importances)[::-1]
        cumulative_importance = np.cumsum(sorted_importances)
        
        # Find how many features needed for threshold
        threshold = self.config['feature_selection_threshold']
        n_features = np.argmax(cumulative_importance >= threshold) + 1
        
        # Use at least the minimum number of features
        n_features = max(self.config['min_features'], n_features)
        
        self.log(f"Selected {n_features} features (top {n_features/len(importances):.1%}) that account for {threshold:.1%} of importance")
        
        # Select features
        selected_indices = indices[:n_features]
        self.X_train_selected = self.X_train[:, selected_indices]
        self.X_val_selected = self.X_val[:, selected_indices]
        self.X_test_selected = self.X_test[:, selected_indices]
        self.selected_feature_names = [self.feature_names[i] for i in selected_indices]
        self.selected_indices = selected_indices  # Store for later use
        
        # Save feature importance analysis
        with open(f"{self.output_dir}/feature_importance.txt", "w") as f:
            f.write(f"Feature importance for {self.best_model_type} model:\n\n")
            for i in indices[:n_features]:
                f.write(f"{self.feature_names[i]}: {importances[i]:.6f}\n")
                
        # Group feature importance
        feature_groups = {
            'SVD': sum(importances[i] for i, name in enumerate(self.feature_names) if name.startswith('svd')),
            'Lagged': sum(importances[i] for i, name in enumerate(self.feature_names) if 'lag' in name and 'output' in name),
            'Rolling Mean': sum(importances[i] for i, name in enumerate(self.feature_names) if 'rollmean' in name),
            'Rolling Std': sum(importances[i] for i, name in enumerate(self.feature_names) if 'rollstd' in name),
            'Rolling Min/Max': sum(importances[i] for i, name in enumerate(self.feature_names) if 'rollmin' in name or 'rollmax' in name),
            'Diff': sum(importances[i] for i, name in enumerate(self.feature_names) if 'diff' in name),
            'EWM': sum(importances[i] for i, name in enumerate(self.feature_names) if 'ewm' in name),
            'Fourier': sum(importances[i] for i, name in enumerate(self.feature_names) if 'sin_period' in name or 'cos_period' in name),
            'Interaction': sum(importances[i] for i, name in enumerate(self.feature_names) if 'interaction' in name)
        }
        
        # Log group importance
        self.log("Feature importance by group:")
        for group, importance in feature_groups.items():
            self.log(f"  {group}: {importance:.4f}")
            
        # Set flag for later use
        self.feature_selection = True
    
    def optimize_hyperparameters(self):
        """Optimize hyperparameters for the best model type"""
        if not self.config['optimize_hyperparams']:
            self.log("Hyperparameter optimization disabled, skipping...")
            return
        
        self.log(f"Optimizing hyperparameters for {self.best_model_type}...")
        
        # Choose correct data to use based on whether feature selection was done
        if self.feature_selection:
            X_train = self.X_train_selected
        else:
            X_train = self.X_train
        
        # Define parameter grid based on model type
        if self.best_model_type == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 15, 25, 35],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
        elif self.best_model_type == 'GradientBoosting':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9, 1.0]
            }
            
        elif self.best_model_type == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            
        elif self.best_model_type == 'Lasso':
            param_grid = {
                'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
            }
            
        elif self.best_model_type == 'Ridge':
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            }
            
        else:
            self.log(f"Hyperparameter optimization not supported for {self.best_model_type}")
            return
        
        # Optimize for each station group
        station_params = {}
        group_params = {}
        
        # If we're using station groups, optimize for each group separately
        if self.config['use_station_clustering'] and hasattr(self, 'station_groups'):
            for group_id in range(len(self.station_groups)):
                # Get stations in this group
                stations = self.station_groups[group_id]
                
                if len(stations) == 0:
                    continue
                    
                # Select a representative station from the group
                rep_station = stations[0]
                
                self.log(f"Optimizing for group {group_id+1} (representative station {rep_station+1})...")
                
                # Optimize parameters for this station
                best_params = self._optimize_for_station(rep_station, X_train, param_grid)
                
                # Store parameters for this group
                group_params[group_id] = best_params
                
                # Apply to all stations in the group
                for station in stations:
                    station_params[station] = best_params
        else:
            # Sample a few stations to optimize
            station_subset = min(5, self.y_train.shape[1])
            sample_stations = np.random.choice(self.y_train.shape[1], station_subset, replace=False)
            
            for station in sample_stations:
                self.log(f"Optimizing for station {station+1}...")
                
                # Optimize parameters for this station
                best_params = self._optimize_for_station(station, X_train, param_grid)
                
                # Store parameters
                station_params[station] = best_params
                
            # For stations not in the sample, use the parameters of the nearest station in sample
            for station in range(self.y_train.shape[1]):
                if station not in station_params:
                    # Find most similar station in sample
                    best_sample = sample_stations[0]
                    best_corr = -np.inf
                    
                    for sample_station in sample_stations:
                        corr = np.corrcoef(self.output_data.iloc[:, station], 
                                          self.output_data.iloc[:, sample_station])[0, 1]
                        if corr > best_corr:
                            best_corr = corr
                            best_sample = sample_station
                            
                    # Use parameters from the most similar station
                    station_params[station] = station_params[best_sample]
                    
        # Store optimized parameters
        self.station_params = station_params
        if group_params:
            self.group_params = group_params
            
        # Log the most common parameter settings
        param_counts = {}
        for param_name in param_grid.keys():
            values = [params.get(param_name) for params in station_params.values()]
            param_counts[param_name] = {}
            
            for value in values:
                if value not in param_counts[param_name]:
                    param_counts[param_name][value] = 0
                param_counts[param_name][value] += 1
                
        self.log("Most common parameter values:")
        for param, counts in param_counts.items():
            most_common = max(counts.items(), key=lambda x: x[1])
            self.log(f"  {param}: {most_common[0]} (used for {most_common[1]} stations)")
    
    def _optimize_for_station(self, station, X_train, param_grid):
        """Optimize hyperparameters for a specific station"""
        # Get station data
        y_station = self.y_train[:, station]
        
        # Create base model for optimization
        if self.best_model_type == 'RandomForest':
            base_model = RandomForestRegressor(random_state=42)
        elif self.best_model_type == 'GradientBoosting':
            base_model = GradientBoostingRegressor(random_state=42)
        elif self.best_model_type == 'XGBoost':
            base_model = xgb.XGBRegressor(random_state=42)
        elif self.best_model_type == 'Lasso':
            base_model = Lasso(random_state=42, max_iter=10000)  # Increased max_iter
        else:  # Ridge
            base_model = Ridge(random_state=42)
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Find best parameters
        from sklearn.model_selection import GridSearchCV
        
        search = GridSearchCV(
            base_model, param_grid,
            scoring='neg_mean_squared_error',
            cv=tscv,
            verbose=0,
            n_jobs=1
        )
        
        # Fit on the station data
        search.fit(X_train, y_station)
        
        # Get best parameters
        best_params = search.best_params_
        best_score = -search.best_score_
        
        self.log(f"  Best parameters: {best_params}")
        self.log(f"  Best score: {best_score:.4f} MSE")
        
        return best_params
            
    def train_optimized_models(self):
        """Train models with optimized hyperparameters"""
        self.log("Training optimized models...")
        
        # Choose data to use - always use the feature-selected data if available
        if self.feature_selection:
            X_train = self.X_train_selected
            X_val = self.X_val_selected
        else:
            X_train = self.X_train
            X_val = self.X_val
            
        # Create optimized models for each station
        optimized_models = {}
        val_r2_scores = []
        
        for i in range(self.y_train.shape[1]):
            # Get optimized parameters for this station
            if hasattr(self, 'station_params') and i in self.station_params:
                params = self.station_params[i]
            else:
                # Use default parameters if optimization wasn't performed
                params = {}
                
            # Create model with optimized parameters
            model = self._get_model_instance(self.best_model_type, params)
            
            # Train on this station's data
            model.fit(X_train, self.y_train[:, i])
            
            # Evaluate on validation set
            val_preds = model.predict(X_val)
            val_r2 = r2_score(self.y_val[:, i], val_preds)
            val_r2_scores.append(val_r2)
            
            # Store model
            optimized_models[i] = model
            
            # Log progress
            if (i+1) % 10 == 0:
                self.log(f"  Trained {i+1}/{self.y_train.shape[1]} optimized models")
                
        # Calculate average validation performance
        avg_val_r2 = np.mean(val_r2_scores)
        
        # Store in models dict
        self.models['Optimized'] = {
            'avg_val_r2': avg_val_r2,
            'models': optimized_models,
            'r2_scores': val_r2_scores
        }
        
        self.log(f"Optimized models average validation R²: {avg_val_r2:.4f}")
        
        # Update best model type if optimized is better
        if avg_val_r2 > self.models[self.best_model_type]['avg_val_r2']:
            self.log(f"Optimized model outperforms {self.best_model_type}: {avg_val_r2:.4f} > {self.models[self.best_model_type]['avg_val_r2']:.4f}")
            self.best_model_type = 'Optimized'

    def _get_model_instance(self, model_type, params=None):
        """Create a model instance of the specified type with given parameters"""
        if params is None:
            params = {}
            
        # Convert numpy types to Python native types
        clean_params = {}
        for key, value in params.items():
            if hasattr(value, 'item'):
                value = value.item()
            if key == 'max_depth' and value is not None:
                value = int(value)
            elif key in ['n_estimators', 'min_samples_split', 'min_samples_leaf']:
                if value is not None and value > 1:
                    value = int(value)
            clean_params[key] = value
            
        # Create model with the appropriate parameters
        if model_type == 'LinearRegression':
            return LinearRegression()
        elif model_type == 'Ridge':
            return Ridge(**clean_params)
        elif model_type == 'Lasso':
            return Lasso(**clean_params)
        elif model_type == 'RandomForest':
            return RandomForestRegressor(random_state=42, **clean_params)
        elif model_type == 'GradientBoosting':
            return GradientBoostingRegressor(random_state=42, **clean_params)
        elif model_type == 'XGBoost':
            return xgb.XGBRegressor(random_state=42, **clean_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def create_stacking_ensemble(self):
        """Create advanced stacking ensemble from multiple base models"""
        if not self.config['create_ensembles']:
            self.log("Ensemble creation disabled, skipping...")
            return
        
        self.log("Creating stacking ensemble model...")
        
        # Identify top performing base models for ensemble
        model_performance = [(name, info['avg_val_r2']) 
                          for name, info in self.models.items() if name != 'Optimized']
        
        # Sort by performance (descending)
        model_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Select top models (maximum 3)
        top_models = [model_performance[i][0] for i in range(min(3, len(model_performance)))]
        
        # Only include models with positive R²
        base_models = [name for name in top_models if self.models[name]['avg_val_r2'] > 0]
        
        # Make sure we have at least 2 models
        if len(base_models) < 2:
            self.log("Not enough good models for ensemble, skipping...")
            return
            
        self.log(f"Building stacking ensemble using: {base_models}")
        
        # Set up datasets
        X_val_full = self.X_val
        X_test_full = self.X_test
        
        # Generate predictions one station at a time to avoid dimension issues
        meta_models = {}
        test_ensemble_preds = np.zeros_like(self.y_test)
        test_r2_scores = []
        
        # Process one station at a time
        for station_idx in range(self.y_train.shape[1]):
            # Get validation predictions for this station from each base model
            val_station_preds = []
            
            for name in base_models:
                model = self.models[name]['models'][station_idx]
                val_pred = model.predict(X_val_full)
                val_station_preds.append(val_pred)
            
            # Stack predictions horizontally to create meta-features
            stacking_X_val = np.column_stack(val_station_preds)
            
            # Train meta-model for this station
            meta_model = Ridge(alpha=0.1)
            meta_model.fit(stacking_X_val, self.y_val[:, station_idx])
            
            # Get test predictions for this station
            test_station_preds = []
            
            for name in base_models:
                model = self.models[name]['models'][station_idx]
                test_pred = model.predict(X_test_full)
                test_station_preds.append(test_pred)
            
            # Stack in the same way as for validation
            stacking_X_test = np.column_stack(test_station_preds)
            
            # Make prediction with meta-model
            test_pred = meta_model.predict(stacking_X_test)
            test_ensemble_preds[:, station_idx] = test_pred
            
            # Calculate R² score
            r2 = r2_score(self.y_test[:, station_idx], test_pred)
            test_r2_scores.append(r2)
            
            # Store meta-model
            meta_models[station_idx] = meta_model
        
        # Calculate overall ensemble performance
        avg_r2 = np.mean(test_r2_scores)
        
        # Store ensemble model
        self.models['Ensemble'] = {
            'avg_val_r2': avg_r2,
            'models': meta_models,
            'r2_scores': test_r2_scores,
            'base_models': base_models,
            'predictions': test_ensemble_preds
        }
        
        self.log(f"Stacking ensemble test R²: {avg_r2:.4f}")
        
        # Update best model if ensemble is better
        current_best_r2 = self.models[self.best_model_type]['avg_val_r2']
        if avg_r2 > current_best_r2:
            self.log(f"Ensemble outperforms {self.best_model_type} ({avg_r2:.4f} > {current_best_r2:.4f})")
            self.best_model_type = 'Ensemble'
    
    def evaluate_models(self):
        """Comprehensive evaluation of all models"""
        self.log("Evaluating all models on test data...")
        
        # Dictionary to store performance metrics
        model_performance = {}
        
        # Evaluate regular models first (non-ensemble)
        for model_type, model_info in self.models.items():
            if model_type == 'Ensemble':
                continue  # Handle ensemble separately
                
            # Choose appropriate test data
            if model_type == 'Optimized' and self.feature_selection:
                X_test = self.X_test_selected
            else:
                X_test = self.X_test
                
            # Make predictions for each output column
            test_preds = np.zeros_like(self.y_test)
            
            for i, model in enumerate(model_info['models'].values()):
                test_preds[:, i] = model.predict(X_test)
                
            # Calculate metrics
            mse = mean_squared_error(self.y_test, test_preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, test_preds)
            
            # Calculate R² for each output
            r2_scores = []
            for i in range(self.y_test.shape[1]):
                r2 = r2_score(self.y_test[:, i], test_preds[:, i])
                r2_scores.append(r2)
                
            avg_r2 = np.mean(r2_scores)
            
            # Store metrics
            model_performance[model_type] = {
                'mse': mse,
                'rmse': rmse, 
                'mae': mae,
                'avg_r2': avg_r2,
                'r2_scores': r2_scores,
                'predictions': test_preds
            }
            
            # Log results
            self.log(f"{model_type} - Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, Avg R²: {avg_r2:.4f}")
        
        # Evaluate ensemble if available
        if 'Ensemble' in self.models and 'predictions' in self.models['Ensemble']:
            # Ensemble predictions are already calculated
            test_preds = self.models['Ensemble']['predictions']
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, test_preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, test_preds)
            
            # R² scores already calculated during ensemble creation
            r2_scores = self.models['Ensemble']['r2_scores']
            avg_r2 = np.mean(r2_scores)
            
            # Store metrics
            model_performance['Ensemble'] = {
                'mse': mse,
                'rmse': rmse, 
                'mae': mae,
                'avg_r2': avg_r2,
                'r2_scores': r2_scores,
                'predictions': test_preds
            }
            
            # Log results
            self.log(f"Ensemble - Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, Avg R²: {avg_r2:.4f}")
        
        # Calculate metrics in original scale for best model
        best_preds = model_performance[self.best_model_type]['predictions']
        best_preds_original = self.output_scaler.inverse_transform(best_preds)
        y_test_original = self.output_scaler.inverse_transform(self.y_test)
        
        # Calculate metrics in original scale
        mse_orig = mean_squared_error(y_test_original, best_preds_original)
        rmse_orig = np.sqrt(mse_orig)
        mae_orig = mean_absolute_error(y_test_original, best_preds_original)
        
        self.log(f"Best model ({self.best_model_type}) in original scale:")
        self.log(f"  MSE: {mse_orig:.4f}")
        self.log(f"  RMSE: {rmse_orig:.4f}")
        self.log(f"  MAE: {mae_orig:.4f}")
        
        # Calculate R² in original scale
        r2_orig = r2_score(y_test_original.flatten(), best_preds_original.flatten())
        self.log(f"  Overall R²: {r2_orig:.4f}")
        
        # Create visualizations
        self.create_evaluation_plots(model_performance, y_test_original, best_preds_original)
        
        # Store performance data
        self.model_performance = model_performance
        
        # Display final accuracy and other key metrics
        self.log("\n" + "="*50)
        self.log("FINAL MODEL PERFORMANCE SUMMARY")
        self.log("="*50)
        self.log(f"Best Model: {self.best_model_type}")
        self.log(f"Test R² Score: {model_performance[self.best_model_type]['avg_r2']:.4f}")
        self.log(f"Test RMSE (normalized): {model_performance[self.best_model_type]['rmse']:.4f}")
        self.log(f"Test RMSE (original scale): {rmse_orig:.4f}")
        self.log(f"Test MAE (original scale): {mae_orig:.4f}")
        
        # Calculate accuracy as 1 - normalized MAE (a common metric)
        mae_norm = model_performance[self.best_model_type]['mae']
        accuracy = max(0, 1 - mae_norm)  # Ensure non-negative
        self.log(f"Model Accuracy: {accuracy:.4f} (1 - normalized MAE)")
        self.log("="*50 + "\n")
        
        return model_performance
    
    def create_evaluation_plots(self, model_performance, actuals, predictions):
        """Create detailed evaluation plots"""
        self.log("Creating evaluation visualizations...")
        
        # Create directory for plots
        eval_dir = f"{self.output_dir}/evaluation"
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        
        # 1. Best and worst predictions
        self.plot_best_worst_predictions(actuals, predictions, eval_dir)
        
        # 2. R² score distribution
        self.plot_r2_distribution(model_performance, eval_dir)
        
        # 3. Model comparison
        self.plot_model_comparison(model_performance, eval_dir)
        
        # 4. Actual vs Predicted plot
        self.plot_actual_vs_predicted(actuals, predictions, eval_dir)
        
    def plot_best_worst_predictions(self, actuals, predictions, output_dir):
        """Plot best and worst station predictions"""
        # Calculate R² for each output
        r2_values = []
        for i in range(predictions.shape[1]):
            r2 = r2_score(actuals[:, i], predictions[:, i])
            r2_values.append((i, r2))
        
        # Sort by R² value
        r2_values.sort(key=lambda x: x[1], reverse=True)
        
        # Plot top 3 best and top 3 worst predictions
        plt.figure(figsize=(15, 15))
        
        # Plot best predictions
        for idx, (col_idx, r2) in enumerate(r2_values[:3]):
            plt.subplot(6, 1, idx+1)
            plt.plot(actuals[:, col_idx], 'b-', label='Actual')
            plt.plot(predictions[:, col_idx], 'r-', label='Predicted')
            plt.title(f'Best Prediction #{idx+1} - Station {col_idx+1} (R² = {r2:.4f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        # Plot worst predictions
        for idx, (col_idx, r2) in enumerate(r2_values[-3:]):
            plt.subplot(6, 1, idx+4)
            plt.plot(actuals[:, col_idx], 'b-', label='Actual')
            plt.plot(predictions[:, col_idx], 'r-', label='Predicted')
            plt.title(f'Worst Prediction #{idx+1} - Station {col_idx+1} (R² = {r2:.4f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(f'{output_dir}/best_worst_predictions.png')
        plt.close()
        
    def plot_r2_distribution(self, model_performance, output_dir):
        """Plot distribution of R² scores"""
        # Get R² scores for best model
        r2_scores = model_performance[self.best_model_type]['r2_scores']
        
        plt.figure(figsize=(10, 6))
        plt.hist(r2_scores, bins=20)
        plt.axvline(0, color='r', linestyle='--')
        plt.title(f'Distribution of R² Scores - {self.best_model_type} Model')
        plt.xlabel('R²')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/r2_distribution.png')
        plt.close()
        
        # Count stations with positive and negative R²
        positive_r2 = sum(1 for r2 in r2_scores if r2 > 0)
        negative_r2 = sum(1 for r2 in r2_scores if r2 <= 0)
        
        self.log(f"R² distribution: {positive_r2} stations with positive R², {negative_r2} with negative R²")
        
    def plot_model_comparison(self, model_performance, output_dir):
        """Plot comparison of different models"""
        # Extract metrics for comparison
        model_names = list(model_performance.keys())
        r2_values = [model_performance[name]['avg_r2'] for name in model_names]
        rmse_values = [model_performance[name]['rmse'] for name in model_names]
        mae_values = [model_performance[name]['mae'] for name in model_names]
        
        # Create bar chart comparison
        plt.figure(figsize=(12, 8))
        
        # R² comparison
        plt.subplot(3, 1, 1)
        plt.bar(model_names, r2_values, color='skyblue')
        plt.title('Average R² by Model')
        plt.grid(True, alpha=0.3)
        
        # RMSE comparison
        plt.subplot(3, 1, 2)
        plt.bar(model_names, rmse_values, color='salmon')
        plt.title('RMSE by Model')
        plt.grid(True, alpha=0.3)
        
        # MAE comparison
        plt.subplot(3, 1, 3)
        plt.bar(model_names, mae_values, color='lightgreen')
        plt.title('MAE by Model')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png')
        plt.close()
        
    def plot_actual_vs_predicted(self, actuals, predictions, output_dir):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 10))
        
        # Flatten the arrays for overall comparison
        actual_flat = actuals.flatten()
        pred_flat = predictions.flatten()
        
        # Create the scatter plot
        plt.scatter(actual_flat, pred_flat, alpha=0.3)
        
        # Add perfect prediction line
        min_val = min(actual_flat.min(), pred_flat.min())
        max_val = max(actual_flat.max(), pred_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
        
        # Calculate overall R²
        r2 = r2_score(actual_flat, pred_flat)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/actual_vs_predicted.png')
        plt.close()
    
    def save_best_model(self):
        """Save the best model and preprocessing components"""
        self.log(f"Saving best model ({self.best_model_type})...")
        
        # Create model directory
        model_dir = f"{self.output_dir}/best_model"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Handle different model types
        if self.best_model_type == 'Ensemble':
            # Save component models
            component_dir = f"{model_dir}/components"
            if not os.path.exists(component_dir):
                os.makedirs(component_dir)
                
            # Save base models information
            base_models = self.models['Ensemble']['base_models']
            joblib.dump(base_models, f"{model_dir}/base_models.pkl")
            
            # Save meta-models (one per station)
            for i, model in enumerate(self.models['Ensemble']['models'].values()):
                joblib.dump(model, f"{model_dir}/meta_model_{i+1}.pkl")
                
            # Also save component models
            for name in base_models:
                model_subdir = f"{component_dir}/{name}"
                if not os.path.exists(model_subdir):
                    os.makedirs(model_subdir)
                    
                for i, model in enumerate(self.models[name]['models'].values()):
                    joblib.dump(model, f"{model_subdir}/station_{i+1}_model.pkl")
                    
        else:
            # Save individual models for each station
            for i, model in enumerate(self.models[self.best_model_type]['models'].values()):
                joblib.dump(model, f"{model_dir}/station_{i+1}_model.pkl")
                
            # If per-station parameters were used, save them
            if hasattr(self, 'station_params'):
                joblib.dump(self.station_params, f"{model_dir}/station_params.pkl")
            
        # Save preprocessing components
        joblib.dump(self.input_scaler, f"{model_dir}/input_scaler.pkl")
        joblib.dump(self.output_scaler, f"{model_dir}/output_scaler.pkl")
        joblib.dump(self.svd, f"{model_dir}/svd.pkl")
        
        # Save feature selection info if used
        if self.feature_selection:
            joblib.dump(self.selected_feature_names, f"{model_dir}/selected_features.pkl")
            joblib.dump(self.selected_indices, f"{model_dir}/selected_indices.pkl")
            
        # Save station grouping if available
        if hasattr(self, 'station_groups'):
            joblib.dump(self.station_groups, f"{model_dir}/station_groups.pkl")
            
        # Save configuration
        config = {
            'model_type': self.best_model_type,
            'feature_selection': self.feature_selection,
            'n_stations': self.output_scaled.shape[1],
            'svd_components': self.config['n_components'],
            'n_lags': self.config['n_lags'],
            'forecast_horizon': self.config['forecast_horizon'],
            'use_xgboost': self.use_xgboost,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(config, f"{model_dir}/config.pkl")
        
        # Save performance summary
        if hasattr(self, 'model_performance'):
            # Extract key metrics to avoid large file sizes
            performance_summary = {}
            for model_type, perf in self.model_performance.items():
                performance_summary[model_type] = {
                    'mse': perf['mse'],
                    'rmse': perf['rmse'],
                    'mae': perf['mae'],
                    'avg_r2': perf['avg_r2'],
                    'r2_scores': perf['r2_scores']
                }
                
            joblib.dump(performance_summary, f"{model_dir}/performance_metrics.pkl")
            
        self.log(f"Model and components saved to {model_dir}")
        
    def run_pipeline(self):
        """Run the complete modeling pipeline with improved ordering"""
        start_time = time.time()
        self.log("Starting enhanced power prediction pipeline...")
        
        try:
            # Data processing
            self.load_data()
            self.preprocess_data()
            self.split_data()
            
            # Model training - train baseline models first
            self.train_baseline_models()
            
            # Feature selection - do this before optimized models
            if self.config['feature_selection']:
                self.select_important_features()
                
            # Hyperparameter optimization after feature selection
            # This ensures we optimize on the reduced feature set
            if self.config['optimize_hyperparams']:
                self.optimize_hyperparameters()
                
            # Train optimized models on the reduced feature set
            self.train_optimized_models()
            
            # Create ensemble using only baseline models to avoid feature mismatch
            if self.config['create_ensembles']:
                self.create_stacking_ensemble()
                
            # Final evaluation and saving
            results = self.evaluate_models()
            self.save_best_model()
            
            # Calculate total runtime
            runtime = time.time() - start_time
            self.log(f"Pipeline completed successfully in {runtime:.1f} seconds ({runtime/60:.1f} minutes)!")
            
            # Get best model's performance
            best_result = results[self.best_model_type]
            
            # Return key metrics
            return {
                'model_type': self.best_model_type,
                'mse': best_result['mse'],
                'rmse': best_result['rmse'], 
                'mae': best_result['mae'],
                'avg_r2': best_result['avg_r2'],
                'accuracy': max(0, 1 - best_result['mae'])  # 1 - normalized MAE
            }
            
        except Exception as e:
            self.log(f"ERROR: Pipeline failed: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None


# Run the pipeline
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Enhanced Power Prediction Model")
    print("="*70 + "\n")
    
    # Paths to data files
    input_path = "input_dataset (1).csv"
    output_path = "output_dataset (1).csv"
    
    # Create and run the model with optimized settings
    model = EnhancedPowerPredictor(
        input_path=input_path,
        output_path=output_path,
        use_xgboost=True  # XGBoost is now available!
    )
    
    # Key improvements: 
    # 1. Increase minimum features to avoid overfitting
    model.config['min_features'] = 20
    
    # 2. Lower feature selection threshold for broader feature inclusion
    model.config['feature_selection_threshold'] = 0.80
    
    # 3. Adjust station clustering
    model.config['n_station_clusters'] = 5
    
    # 4. Improved XGBoost parameters
    model.config['xgb_params'] = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist'  # Much faster training
    }
    
    # Run the complete pipeline
    results = model.run_pipeline()
    
    # Print final results
    print("\n" + "="*70)
    print(f"Power Prediction Model Complete! Results saved to {model.output_dir}")
    
    if results is not None:
        print(f"\nBest Model Type: {results['model_type']}")
        print(f"Test R² Score: {results['avg_r2']:.4f}")
        print(f"Test RMSE: {results['rmse']:.4f}")
        print(f"Test MAE: {results['mae']:.4f}")
        print(f"Model Accuracy: {results['accuracy']:.4f}")
        
        # Compare to previous model
        previous_r2 = 0.947  # R² from previous run
        improvement = (results['avg_r2'] - previous_r2) / previous_r2 * 100
        print(f"\nImprovement over previous model: {improvement:.2f}%")
    
    print("="*70 + "\n")