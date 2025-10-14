#!/usr/bin/env python
# SCADA Transformer Monitoring Pipeline
# End-to-end pipeline for processing and predicting transformer faults from SCADA data

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
from .features import SCADAFeatureExtractor

# Import FocalLoss to ensure custom model loading works
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "class_weights": {str(k): float(v) for k, v in self.class_weights.items()} if self.class_weights else None
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        if config.get("class_weights"):
            config["class_weights"] = {int(k): float(v) for k, v in config["class_weights"].items()}
        return cls(**config)

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        ce = -y_true * tf.math.log(y_pred)
        
        if self.class_weights is not None:
            n_classes = tf.shape(y_true)[1]
            class_weights_tensor = tf.ones(n_classes)
            
            for cls_idx, weight in self.class_weights.items():
                class_weights_tensor = tf.tensor_scatter_nd_update(
                    class_weights_tensor,
                    [[cls_idx]],
                    [tf.cast(weight, tf.float32)]
                )
            
            weighted_ce = ce * tf.expand_dims(class_weights_tensor, 0)
        else:
            weighted_ce = ce
        
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.ones_like(y_true) * self.alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_t, 1 - alpha_t)
        focal_weight = alpha_t * tf.pow((1 - p_t), self.gamma)
        focal_loss = focal_weight * weighted_ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))

class SCADATransformerPipeline:
    """End-to-end pipeline for transformer fault prediction using SCADA data"""
    
    def __init__(self, model_path='best_scada_model.h5', scaler_path='scada_scaler.pkl', 
                 encoder_path='scada_fault_encoder.pkl'):
        """
        Initialize the SCADA transformer monitoring pipeline
        
        Args:
            model_path: Path to the trained model file (.h5)
            scaler_path: Path to the feature scaler file (.pkl)
            encoder_path: Path to the fault encoder file (.pkl)
        """
        self.feature_extractor = SCADAFeatureExtractor()
        self.model = None
        self.scaler = None
        self.encoder = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        
        # Load model and preprocessing components
        self._load_components()
        
        # Define fault severity and criticality thresholds
        self.anomaly_threshold = 0.5
        self.severity_levels = {
            'low': (0.0, 0.3),
            'medium': (0.3, 0.7),
            'high': (0.7, 1.0)
        }
        self.criticality_levels = {
            'normal': (0.0, 0.3),
            'warning': (0.3, 0.7),
            'critical': (0.7, 1.0)
        }
    
    def _load_components(self):
        """Load model, scaler, and encoder"""
        try:
            print(f"Loading model from {self.model_path}...")
            self.model = tf.keras.models.load_model(
                self.model_path, 
                custom_objects={'FocalLoss': FocalLoss}
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
        
        try:
            print(f"Loading scaler from {self.scaler_path}...")
            self.scaler = joblib.load(self.scaler_path)
            print("✅ Scaler loaded successfully")
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            sys.exit(1)
            
        try:
            print(f"Loading encoder from {self.encoder_path}...")
            self.encoder = joblib.load(self.encoder_path)
            # Ensure class labels are standard strings
            self.encoder.classes_ = [str(cls) for cls in self.encoder.classes_]
            print("✅ Encoder loaded successfully")
            print(f"   Fault classes: {self.encoder.classes_}")
        except Exception as e:
            print(f"❌ Error loading encoder: {e}")
            sys.exit(1)
    
    def process_data(self, data, format_type='csv'):
        """
        Process input data and extract features
        
        Args:
            data: Input data (path to CSV file, DataFrame, or dictionary)
            format_type: Type of input ('csv', 'dataframe', 'dict')
            
        Returns:
            Processed features as numpy array
        """
        if format_type == 'csv':
            if isinstance(data, str) and os.path.exists(data):
                print(f"Loading data from {data}...")
                df = pd.read_csv(data)
            else:
                raise ValueError(f"CSV file not found: {data}")
                
        elif format_type == 'dataframe':
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                raise ValueError("Expected pandas DataFrame but got different type")
                
        elif format_type == 'dict':
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("Expected dictionary but got different type")
                
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        # Check if required columns are present
        print("Extracting features from data...")
        X_features = []
        
        # Process each row
        for _, row in df.iterrows():
            features = self.feature_extractor.extract(row.to_dict())
            X_features.append(list(features.values()))
        
        # Convert to numpy array
        X = np.array(X_features, dtype=np.float32)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        print(f"✅ Processed {len(X_scaled)} data points with {X_scaled.shape[1]} features")
        return X_scaled
    
    def predict(self, X_scaled, use_tta=True, n_tta=5, tta_noise=0.01):
        """
        Make predictions using the loaded model with Test Time Augmentation
        
        Args:
            X_scaled: Scaled features
            use_tta: Whether to use Test Time Augmentation for robust predictions
            n_tta: Number of TTA iterations
            tta_noise: Magnitude of noise to add during TTA
            
        Returns:
            Dictionary with prediction results
        """
        if use_tta:
            print(f"Making predictions with Test Time Augmentation (iterations={n_tta})...")
            all_preds = []
            
            for i in range(n_tta):
                # Add small noise for TTA
                X_aug = X_scaled + np.random.normal(0, tta_noise, X_scaled.shape)
                preds = self.model.predict(X_aug, verbose=0)
                all_preds.append(preds)
            
            # Average predictions
            ensemble_preds = [np.mean([pred[i] for pred in all_preds], axis=0) for i in range(4)]
        else:
            print("Making predictions...")
            preds = self.model.predict(X_scaled, verbose=0)
            ensemble_preds = preds
        
        return ensemble_preds
    
    def interpret_results(self, raw_predictions, confidence_threshold=0.6):
        """
        Interpret raw model predictions into meaningful results
        
        Args:
            raw_predictions: Raw model output
            confidence_threshold: Threshold for reliable predictions
            
        Returns:
            List of dictionaries with interpreted results
        """
        # Unpack predictions
        anomaly_preds = raw_predictions[0].flatten()
        fault_preds = raw_predictions[1]
        severity_preds = raw_predictions[2].flatten()
        criticality_preds = raw_predictions[3].flatten()
        
        results = []
        
        for i in range(len(anomaly_preds)):
            # Get predicted fault type and confidence
            fault_idx = np.argmax(fault_preds[i])
            fault_confidence = fault_preds[i][fault_idx]
            
            # Get top 3 fault types with probabilities
            top3_indices = np.argsort(fault_preds[i])[-3:][::-1]
            top3_faults = [(self.encoder.classes_[idx], float(fault_preds[i][idx])) for idx in top3_indices]
            
            # Determine fault state
            is_anomaly = bool(anomaly_preds[i] > self.anomaly_threshold)
            fault_type = self.encoder.classes_[fault_idx]
            
            # Get severity level
            severity_value = float(severity_preds[i])
            severity_level = None
            for level, (low, high) in self.severity_levels.items():
                if low <= severity_value < high:
                    severity_level = level
            
            # Get criticality level
            criticality_value = float(criticality_preds[i])
            criticality_level = None
            for level, (low, high) in self.criticality_levels.items():
                if low <= criticality_value < high:
                    criticality_level = level
            
            # Reliability assessment
            if fault_confidence < confidence_threshold:
                reliability = "low"
            else:
                reliability = "high"
            
            # Create result dictionary
            result = {
                'anomaly_detected': is_anomaly,
                'anomaly_probability': float(anomaly_preds[i]),
                'fault_type': fault_type if is_anomaly else 'healthy',
                'fault_confidence': float(fault_confidence),
                'fault_reliability': reliability,
                'top_fault_candidates': top3_faults,
                'severity': float(severity_value),
                'severity_level': severity_level,
                'criticality': float(criticality_value),
                'criticality_level': criticality_level,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'prediction_id': f"pred_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            }
            
            results.append(result)
        
        return results
    
    def generate_recommendations(self, results):
        """
        Generate recommendations based on prediction results
        
        Args:
            results: List of interpreted prediction results
            
        Returns:
            List of dictionaries with recommendations
        """
        recommendations = []
        
        for result in results:
            rec = {
                'prediction_id': result['prediction_id'],
                'actions': [],
                'timeframe': None,
                'priority': None
            }
            
            # Base recommendations on fault type and severity
            if not result['anomaly_detected'] or result['fault_type'] == 'healthy':
                rec['actions'].append("No action required. Continue normal monitoring.")
                rec['timeframe'] = "N/A"
                rec['priority'] = "low"
            else:
                fault_type = result['fault_type']
                severity = result['severity_level']
                criticality = result['criticality_level']
                
                # Set priority based on criticality
                if criticality == 'critical':
                    rec['priority'] = "high"
                    rec['timeframe'] = "Immediate (24-48 hours)"
                elif criticality == 'warning':
                    rec['priority'] = "medium"
                    rec['timeframe'] = "Soon (1-2 weeks)"
                else:
                    rec['priority'] = "low"
                    rec['timeframe'] = "Scheduled maintenance"
                
                # Fault-specific recommendations
                if 'thermal' in fault_type.lower():
                    rec['actions'].append("Check cooling system functionality")
                    rec['actions'].append("Verify oil circulation and levels")
                    rec['actions'].append("Inspect cooling fans and radiators")
                    
                elif 'electrical' in fault_type.lower() or 'winding' in fault_type.lower():
                    rec['actions'].append("Perform insulation resistance test")
                    rec['actions'].append("Check for voltage imbalances")
                    rec['actions'].append("Consider power factor testing")
                    
                elif 'mechanical' in fault_type.lower():
                    rec['actions'].append("Check for unusual vibrations")
                    rec['actions'].append("Inspect physical mounting and connections")
                    rec['actions'].append("Perform acoustic testing for discharge activity")
                    
                elif 'oil' in fault_type.lower() or 'moisture' in fault_type.lower():
                    rec['actions'].append("Take oil sample for detailed DGA analysis")
                    rec['actions'].append("Check for oil leaks and moisture ingress")
                    rec['actions'].append("Verify oil quality and contamination levels")
                    
                # Add severity-specific actions
                if severity == 'high':
                    rec['actions'].append("Consider temporary load reduction")
                    rec['actions'].append("Schedule emergency maintenance")
                    
                elif severity == 'medium':
                    rec['actions'].append("Increase monitoring frequency")
                    rec['actions'].append("Prepare maintenance plan")
            
            recommendations.append(rec)
            
        return recommendations
    
    def run_pipeline(self, data, format_type='csv', output_path=None, use_tta=True):
        """
        Run the complete pipeline from data to recommendations
        
        Args:
            data: Input data (path, DataFrame, or dictionary)
            format_type: Type of input data
            output_path: Path to save results (optional)
            use_tta: Whether to use Test Time Augmentation
            
        Returns:
            Dictionary with complete results
        """
        print("=" * 60)
        print("🔄 RUNNING SCADA TRANSFORMER MONITORING PIPELINE")
        print("=" * 60)
        
        # 1. Process data
        X_scaled = self.process_data(data, format_type)
        
        # 2. Make predictions
        raw_predictions = self.predict(X_scaled, use_tta=use_tta)
        
        # 3. Interpret results
        results = self.interpret_results(raw_predictions)
        
        # 4. Generate recommendations
        recommendations = self.generate_recommendations(results)
        
        # 5. Combine results
        pipeline_results = {
            'processed_data_shape': X_scaled.shape,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'predictions': results,
            'recommendations': recommendations,
            'summary': {
                'n_samples': len(results),
                'anomalies_detected': sum(1 for r in results if r['anomaly_detected']),
                'critical_faults': sum(1 for r in results if r['criticality_level'] == 'critical'),
                'low_reliability_predictions': sum(1 for r in results if r['fault_reliability'] == 'low')
            }
        }
        
        # 6. Save results if path provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2)
            print(f"✅ Results saved to {output_path}")
        
        print("=" * 60)
        print("✅ PIPELINE EXECUTION COMPLETE")
        print(f"   Processed {pipeline_results['summary']['n_samples']} samples")
        print(f"   Detected {pipeline_results['summary']['anomalies_detected']} anomalies")
        print(f"   Found {pipeline_results['summary']['critical_faults']} critical faults")
        print("=" * 60)
        
        return pipeline_results

    def run_from_json(self, data_dict, output_path=None, use_tta=True):
        """Convenience method: run the pipeline using a JSON/dict payload.

        Args:
            data_dict: Dictionary with SCADA measurements; keys should match the
                expected features from SCADAFeatureExtractor (missing values will be
                filled as 0.0 by the extractor).
            output_path: Optional path to save results JSON.
            use_tta: Whether to enable Test Time Augmentation during prediction.

        Returns:
            Dictionary with complete results (same schema as run_pipeline).
        """
        return self.run_pipeline(data=data_dict, format_type='dict', output_path=output_path, use_tta=use_tta)

def main():
    """Run the pipeline from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SCADA Transformer Monitoring Pipeline")
    parser.add_argument('--input', '-i', required=True, help="Input data file path (CSV)")
    parser.add_argument('--output', '-o', help="Output JSON file path")
    parser.add_argument('--model', '-m', default='best_scada_model.h5', help="Model file path")
    parser.add_argument('--scaler', '-s', default='scada_scaler.pkl', help="Scaler file path")
    parser.add_argument('--encoder', '-e', default='scada_fault_encoder.pkl', help="Encoder file path")
    parser.add_argument('--no-tta', action='store_true', help="Disable Test Time Augmentation")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SCADATransformerPipeline(
        model_path=args.model,
        scaler_path=args.scaler,
        encoder_path=args.encoder
    )
    
    # Set default output path if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        args.output = f"scada_results_{timestamp}.json"
    
    # Run pipeline
    pipeline.run_pipeline(
        data=args.input,
        format_type='csv',
        output_path=args.output,
        use_tta=not args.no_tta
    )

if __name__ == "__main__":
    main()