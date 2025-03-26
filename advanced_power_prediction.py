import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
import joblib
import os
import time
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

class AdvancedPowerPredictor:
    def __init__(self, input_path, output_path, output_dir=None, 
                 use_parallel=False, low_memory_mode=True):
        """Initialize with file paths and parameters"""
        self.input_path = input_path
        self.output_path = output_path
        self.low_memory_mode = low_memory_mode
        
        # Parallel processing - disabled by default
        self.use_parallel = use_parallel
            
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
        self.models = {}            # Models before feature selection
        self.optimized_models = {}  # Models after feature selection
        self.best_model_type = None
        self.feature_selection = False
        
        # Configuration settings
        self.config = {
            'lag_features': True,
            'use_station_clustering': True,
            'create_ensembles': True,
            'optimize_hyperparams': True,
            'feature_selection': True,
            'n_components': 300,  # SVD components
            'n_lags': 5,          # How many past values to use
            'rolling_windows': [3, 5, 10],  # Window sizes
            'feature_selection_threshold': 0.90,  # Keep features that contribute to 90% importance
            'forecast_horizon': 1  # Steps ahead to forecast
        }
        
        # Set up logging
        self.log_file = f"{self.output_dir}/log.txt"
        with open(self.log_file, "w") as f:
            f.write(f"Advanced Power Predictor - Run started at {datetime.now()}\n\n")
        
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
        rolling_mean = self.output_data.rolling(window=10).mean()
        rolling_std = self.output_data.rolling(window=10).std()
        
        # Plot for a few stations
        for i in range(min(3, self.output_data.shape[1])):
            plt.figure(figsize=(12, 6))
            plt.plot(self.output_data.iloc[:, i], label='Original', alpha=0.7)
            plt.plot(rolling_mean.iloc[:, i], label='10-period Rolling Mean', linewidth=2)
            plt.plot(rolling_std.iloc[:, i], label='10-period Rolling Std', linewidth=2)
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
        
        # Identify highly correlated station groups
        high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix.abs() < 1.0)
        
        # Create station groups with similar behavior
        self.station_groups = []
        visited = set()
        
        for i in range(corr_matrix.shape[0]):
            if i in visited:
                continue
                
            # Find stations correlated with this one
            correlated = high_corr.iloc[i].index[high_corr.iloc[i]].tolist()
            
            if len(correlated) > 1:  # At least one other station highly correlated
                group = [i] + [idx for idx in correlated if idx != i]
                self.station_groups.append(group)
                visited.update(group)
            else:
                self.station_groups.append([i])
                visited.add(i)
                
        self.log(f"Identified {len(self.station_groups)} station groups")
        
        if len(self.station_groups) < self.output_data.shape[1]:
            with open(f"{output_dir}/station_groups.txt", "w") as f:
                for i, group in enumerate(self.station_groups):
                    f.write(f"Group {i+1}: Stations {[j+1 for j in group]}\n")
            
            self.log(f"Station grouping saved to {output_dir}/station_groups.txt")
    
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
            
            # First-order difference (recent changes)
            diff1 = self.output_df.diff().fillna(0).values
            feature_sets.append(diff1)
            feature_names.extend([f'diff1_output{i}' for i in range(diff1.shape[1])])
            
            # Exponential weighted mean (gives more weight to recent observations)
            ewm_mean = self.output_df.ewm(span=10).mean().values
            feature_sets.append(ewm_mean)
            feature_names.extend([f'ewm_output{i}' for i in range(ewm_mean.shape[1])])
        
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
                n_jobs=1  # Single core to avoid multiprocessing issues
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=3,
                random_state=42
            )
        }
        
        # We'll train separate models for each output column
        for name, model_class in models.items():
            self.log(f"Training {name} baseline models...")
            start_time = time.time()
            
            # Create containers for models and scores
            output_models = {}
            val_r2_scores = []
            
            # Train one model for each output column
            for i in range(self.y_train.shape[1]):
                # Clone the model
                if name == 'LinearRegression':
                    # LinearRegression doesn't have get_params
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
                else:  # GradientBoosting
                    model = GradientBoostingRegressor(
                        n_estimators=100, 
                        learning_rate=0.1, 
                        max_depth=3,
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
        self.analyze_station_performance()
    
    def analyze_station_performance(self):
        """Analyze per-station performance of baseline models"""
        self.log("Analyzing per-station performance...")
        
        # Create directory for station analysis
        analysis_dir = f"{self.output_dir}/station_analysis"
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
            
        # Compare model performance across stations
        model_names = list(self.models.keys())
        station_count = self.y_train.shape[1]
        
        # Extract R² scores per station for each model
        station_scores = np.zeros((len(model_names), station_count))
        
        for i, name in enumerate(model_names):
            station_scores[i, :] = self.models[name]['r2_scores']
            
        # Find best model for each station
        best_model_idx = np.argmax(station_scores, axis=0)
        best_model_stations = {}
        
        for name in model_names:
            best_model_stations[name] = []
            
        for station_idx in range(station_count):
            model_name = model_names[best_model_idx[station_idx]]
            best_model_stations[model_name].append(station_idx)
            
        # Plot model comparison across stations
        plt.figure(figsize=(12, 6))
        
        for i, name in enumerate(model_names):
            plt.plot(station_scores[i], 'o-', label=name, alpha=0.7)
            
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Power Station')
        plt.ylabel('Validation R²')
        plt.title('Model Performance by Power Station')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{analysis_dir}/model_comparison_by_station.png')
        plt.close()
        
        # Log best model distribution
        self.log("Best model distribution across stations:")
        for name, stations in best_model_stations.items():
            self.log(f"  {name}: {len(stations)} stations")
            
        # Note difficult stations (negative R²)
        difficult_stations = []
        
        for station_idx in range(station_count):
            if max(station_scores[:, station_idx]) < 0:
                difficult_stations.append(station_idx)
                
        if difficult_stations:
            self.log(f"Found {len(difficult_stations)} difficult stations with negative R² across all models")
            with open(f"{analysis_dir}/difficult_stations.txt", "w") as f:
                f.write(f"Stations with negative R² across all models:\n")
                f.write(str([idx+1 for idx in difficult_stations]))
    
    def select_important_features(self):
        """Select important features based on the best model"""
        if not self.config['feature_selection']:
            self.log("Feature selection disabled, skipping...")
            return
        
        self.log("Selecting important features...")
        
        # Only supported for tree-based models
        if self.best_model_type not in ['RandomForest', 'GradientBoosting']:
            self.log("Feature selection requires tree-based models, skipping...")
            return
        
        # Sample a model to determine feature importance
        sample_model = list(self.models[self.best_model_type]['models'].values())[0]
        
        # Get feature importances
        importances = sample_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
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
        
        # Use at least 5 features regardless of importance threshold
        n_features = max(5, n_features) 
        
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
            'Lagged': sum(importances[i] for i, name in enumerate(self.feature_names) if 'lag' in name),
            'Rolling Mean': sum(importances[i] for i, name in enumerate(self.feature_names) if 'rollmean' in name),
            'Rolling Std': sum(importances[i] for i, name in enumerate(self.feature_names) if 'rollstd' in name),
            'Diff': sum(importances[i] for i, name in enumerate(self.feature_names) if 'diff' in name),
            'EWM': sum(importances[i] for i, name in enumerate(self.feature_names) if 'ewm' in name)
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
        
        # Choose data to use
        if self.feature_selection:
            X_train = self.X_train_selected
            X_val = self.X_val_selected
        else:
            X_train = self.X_train
            X_val = self.X_val
        
        # Define parameter grid based on model type
        if self.best_model_type == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
        elif self.best_model_type in ['GradientBoosting']:
            param_grid = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9]
            }
            
        elif self.best_model_type == 'Lasso':
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 0.5, 1.0]
            }
            
        elif self.best_model_type == 'Ridge':
            param_grid = {
                'alpha': [0.1, 1.0, 10.0]
            }
            
        else:
            self.log(f"Hyperparameter optimization not supported for {self.best_model_type}")
            return
        
        # Use a subset of stations for optimization to save time
        station_subset = min(3, self.y_train.shape[1])
        best_params = {}
        
        # For each selected station, find best parameters
        for i in range(station_subset):
            self.log(f"Optimizing for station {i+1}...")
            
            # Use TimeSeriesSplit for more reliable validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Create base model
            if self.best_model_type == 'RandomForest':
                base_model = RandomForestRegressor(random_state=42)
            elif self.best_model_type == 'GradientBoosting':
                base_model = GradientBoostingRegressor(random_state=42)
            elif self.best_model_type == 'Lasso':
                base_model = Lasso(random_state=42)
            else:  # Ridge
                base_model = Ridge(random_state=42)
                
            # Use grid search to find best parameters
            search = GridSearchCV(
                base_model, param_grid,
                scoring='neg_mean_squared_error',
                cv=tscv,
                verbose=0,
                n_jobs=1  # Single job to avoid multiprocessing issues
            )
            
            # Fit on this station's data
            search.fit(X_train, self.y_train[:, i])
            
            # Store best parameters
            best_params[i] = search.best_params_
            self.log(f"  Best parameters: {search.best_params_}")
            self.log(f"  Best score: {-search.best_score_:.4f} MSE")
            
        # Analyze parameter consistency
        param_consistency = {}
        
        # For each parameter
        for param in param_grid.keys():
            # Get all values across stations
            values = [best_params[i][param] for i in range(station_subset)]
            
            # If all values are the same, use that value
            if len(set(values)) == 1:
                param_consistency[param] = values[0]
            else:
                # Otherwise, find most common value or median
                if isinstance(values[0], (int, float)) or values[0] is None:
                    # For numeric values or None, use the first value (simplest approach)
                    param_consistency[param] = values[0]
                else:
                    from collections import Counter
                    param_consistency[param] = Counter(values).most_common(1)[0][0]
        
        self.log(f"Consistent parameters across stations: {param_consistency}")
        self.best_hyperparams = param_consistency
    
    def train_optimized_models(self):
        """Train models with optimized hyperparameters"""
        self.log("Training optimized models...")
        
        # Choose data to use
        if self.feature_selection:
            X_train = self.X_train_selected
            X_val = self.X_val_selected
        else:
            X_train = self.X_train
            X_val = self.X_val
            
        # Create optimized model
        if hasattr(self, 'best_hyperparams'):
            # Using optimized hyperparameters - Create a clean copy
            params = {}
            
            # Convert any numpy types to Python native types and ensure correct types
            for key, value in self.best_hyperparams.items():
                if hasattr(value, 'item'):  # If numpy scalar
                    value = value.item()  # Convert to Python scalar
                
                # Handle specific parameter types
                if key == 'max_depth' and value is not None:
                    value = int(value)  # Ensure integer
                elif key == 'n_estimators':
                    value = int(value)  # Ensure integer
                elif key in ['min_samples_split', 'min_samples_leaf']:
                    value = int(value) if value > 1 else value  # Convert to int if > 1
                
                params[key] = value
                
            self.log(f"Using optimized parameters: {params}")
        else:
            # Default improved parameters
            if self.best_model_type == 'RandomForest':
                params = {
                    'n_estimators': 200,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            elif self.best_model_type == 'GradientBoosting':
                params = {
                    'n_estimators': 150,
                    'learning_rate': 0.05,
                    'max_depth': 5,
                    'subsample': 0.8
                }
            elif self.best_model_type == 'Lasso':
                params = {
                    'alpha': 0.1
                }
            else:
                # No optimization for other models
                self.log("No optimization available for this model type")
                return
                
        # Train optimized models for each output
        self.log(f"Training {self.y_train.shape[1]} optimized models...")
        
        optimized_models = {}
        val_r2_scores = []
        
        for i in range(self.y_train.shape[1]):
            # Create model with optimized parameters
            if self.best_model_type == 'RandomForest':
                model = RandomForestRegressor(random_state=42, **params)
            elif self.best_model_type == 'GradientBoosting':
                model = GradientBoostingRegressor(random_state=42, **params)
            elif self.best_model_type == 'Lasso':
                model = Lasso(random_state=42, **params)
            else:  # Ridge
                model = Ridge(random_state=42, **params)
                
            # Train on this output
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
        
        # Store in optimized_models dict separately from the original models
        self.optimized_models = {
            'avg_val_r2': avg_val_r2,
            'models': optimized_models,
            'r2_scores': val_r2_scores
        }
        
        # And also in the main models dict
        self.models['Optimized'] = self.optimized_models
        
        self.log(f"Optimized models average validation R²: {avg_val_r2:.4f}")
        
        # Update best model type if better
        if avg_val_r2 > self.models[self.best_model_type]['avg_val_r2']:
            self.log(f"Optimized model outperforms {self.best_model_type}: {avg_val_r2:.4f} > {self.models[self.best_model_type]['avg_val_r2']:.4f}")
            self.best_model_type = 'Optimized'
    
    def create_ensemble_model(self):
        """Create ensemble model by combining multiple approaches"""
        if not self.config['create_ensembles']:
            self.log("Ensemble creation disabled, skipping...")
            return
        
        self.log("Creating ensemble model...")
        
        # We'll create an ensemble of the best performing models
        # Find model types with best average performance
        model_performance = [(name, info['avg_val_r2']) 
                           for name, info in self.models.items()]
        
        # Sort by performance (descending)
        model_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 2-3 models for ensemble
        top_models = min(3, len(model_performance))
        ensemble_models = [model_performance[i][0] for i in range(top_models)]
        
        # Only include models with positive R²
        ensemble_models = [name for name in ensemble_models 
                         if self.models[name]['avg_val_r2'] > 0]
        
        if len(ensemble_models) == 0:
            self.log("No suitable models for ensemble (all have negative R²)")
            return
            
        self.log(f"Creating ensemble from: {ensemble_models}")
        
        # For each station, evaluate models on validation set
        # and find optimal weights
        ensemble_weights = {}
        
        # Use correct datasets for validation based on whether the model was
        # trained before feature selection or after
        X_val_full = self.X_val
        X_test_full = self.X_test
        
        if self.feature_selection:
            X_val_selected = self.X_val_selected
            X_test_selected = self.X_test_selected
        else:
            X_val_selected = self.X_val
            X_test_selected = self.X_test
        
        # For each station, find optimal weights
        for i in range(self.y_train.shape[1]):
            # Get validation predictions from each model
            val_preds = {}
            
            for name in ensemble_models:
                model = self.models[name]['models'][i]
                
                # Use appropriate dataset based on whether this is a post-feature-selection model
                if name == 'Optimized' and self.feature_selection:
                    val_preds[name] = model.predict(X_val_selected)
                else:
                    val_preds[name] = model.predict(X_val_full)
            
            # Calculate weights based on validation R²
            weights = {}
            total_r2 = 0
            
            for name in ensemble_models:
                r2 = r2_score(self.y_val[:, i], val_preds[name])
                # Only include positive contributions
                if r2 > 0:
                    weights[name] = r2
                    total_r2 += r2
            
            # Normalize weights
            if total_r2 > 0:
                for name in weights:
                    weights[name] /= total_r2
            else:
                # If all models have negative R², use equal weights
                for name in ensemble_models:
                    weights[name] = 1.0 / len(ensemble_models)
            
            # Store weights for this station
            ensemble_weights[i] = weights
        
        # Create ensemble test predictions
        test_ensemble_preds = np.zeros((len(self.X_test), self.y_test.shape[1]))
        
        for i in range(self.y_test.shape[1]):
            # Get weights for this station
            weights = ensemble_weights[i]
            
            # Weighted sum of predictions
            for name, weight in weights.items():
                model = self.models[name]['models'][i]
                
                # Use appropriate dataset based on whether this is a post-feature-selection model
                if name == 'Optimized' and self.feature_selection:
                    test_preds = model.predict(X_test_selected)
                else:
                    test_preds = model.predict(X_test_full)
                    
                test_ensemble_preds[:, i] += weight * test_preds
        
        # Calculate ensemble performance
        ensemble_r2 = []
        for i in range(self.y_test.shape[1]):
            r2 = r2_score(self.y_test[:, i], test_ensemble_preds[:, i])
            ensemble_r2.append(r2)
        
        avg_ensemble_r2 = np.mean(ensemble_r2)
        
        self.models['Ensemble'] = {
            'avg_val_r2': avg_ensemble_r2,
            'r2_scores': ensemble_r2,
            'weights': ensemble_weights,
            'component_models': ensemble_models,
            'predictions': test_ensemble_preds  # Store predictions for evaluation
        }
        
        self.log(f"Ensemble test R²: {avg_ensemble_r2:.4f}")
        
        # Update best model if ensemble is better
        if avg_ensemble_r2 > self.models[self.best_model_type]['avg_val_r2']:
            self.log(f"Ensemble outperforms {self.best_model_type}")
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
        if 'Ensemble' in self.models:
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
        
        # Create visualizations
        self.create_evaluation_plots(model_performance, y_test_original, best_preds_original)
        
        # Store performance data
        self.model_performance = model_performance
        
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
                
            # Save ensemble weights
            joblib.dump(self.models['Ensemble']['weights'], f"{model_dir}/ensemble_weights.pkl")
            
            # Save component models
            component_models = self.models['Ensemble']['component_models']
            joblib.dump(component_models, f"{model_dir}/component_models.pkl")
            
            # Save each component model
            for name in component_models:
                model_subdir = f"{component_dir}/{name}"
                if not os.path.exists(model_subdir):
                    os.makedirs(model_subdir)
                    
                for i, model in enumerate(self.models[name]['models'].values()):
                    joblib.dump(model, f"{model_subdir}/station_{i+1}_model.pkl")
        else:
            # Save individual models for each station
            for i, model in enumerate(self.models[self.best_model_type]['models'].values()):
                joblib.dump(model, f"{model_dir}/station_{i+1}_model.pkl")
            
        # Save preprocessing components
        joblib.dump(self.input_scaler, f"{model_dir}/input_scaler.pkl")
        joblib.dump(self.output_scaler, f"{model_dir}/output_scaler.pkl")
        joblib.dump(self.svd, f"{model_dir}/svd.pkl")
        
        # Save feature selection info if used
        if self.feature_selection:
            joblib.dump(self.selected_feature_names, f"{model_dir}/selected_features.pkl")
            joblib.dump(self.selected_indices, f"{model_dir}/selected_indices.pkl")
            
        # Save configuration
        config = {
            'model_type': self.best_model_type,
            'feature_selection': self.feature_selection,
            'n_stations': self.output_scaled.shape[1],
            'svd_components': self.config['n_components'],
            'n_lags': self.config['n_lags'],
            'forecast_horizon': self.config['forecast_horizon'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(config, f"{model_dir}/config.pkl")
        
        # Save full performance metrics
        if hasattr(self, 'model_performance'):
            # Extract only what's needed to avoid large file sizes
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
        """Run the complete modeling pipeline"""
        start_time = time.time()
        self.log("Starting advanced power prediction pipeline...")
        
        try:
            # Data processing
            self.load_data()
            self.preprocess_data()
            self.split_data()
            
            # Model training and optimization
            self.train_baseline_models()
            
            # Feature selection if enabled
            if self.config['feature_selection']:
                self.select_important_features()
                
            # Hyperparameter optimization if enabled
            if self.config['optimize_hyperparams']:
                self.optimize_hyperparameters()
                
            # Train optimized models
            self.train_optimized_models()
            
            # Create ensemble if enabled
            if self.config['create_ensembles']:
                self.create_ensemble_model()
                
            # Evaluation and saving
            results = self.evaluate_models()
            self.save_best_model()
            
            # Calculate total runtime
            runtime = time.time() - start_time
            self.log(f"Pipeline completed successfully in {runtime:.1f} seconds ({runtime/60:.1f} minutes)!")
            
            # Return best model's performance
            best_result = results[self.best_model_type]
            return best_result['mse'], best_result['rmse'], best_result['mae'], best_result['avg_r2'], self.best_model_type
            
        except Exception as e:
            self.log(f"ERROR: Pipeline failed: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None, None, None, None, None


# Run the pipeline
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Advanced Power Prediction Model With Optimizations")
    print("="*70 + "\n")
    
    # Paths to data files
    input_path = "input_dataset (1).csv"
    output_path = "output_dataset (1).csv"
    
    # Create and run the model with optimized settings
    model = AdvancedPowerPredictor(
        input_path=input_path,
        output_path=output_path,
        use_parallel=False  # Disable parallel processing to avoid pickling errors
    )
    
    # Run pipeline
    mse, rmse, mae, r2, best_model = model.run_pipeline()
    
    print("\n" + "="*70)
    print(f"Model training complete! Results saved to {model.output_dir}")
    
    if mse is not None:
        print(f"Best Model Type: {best_model}")
        print(f"Final Metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    print("="*70 + "\n")