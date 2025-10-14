# Complete FRA Pipeline for Industrial Fault Detection
# End-to-end pipeline from FRA CSV upload to industrial fault diagnosis

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

# Custom FocalLoss for model loading
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
                    class_weights_tensor, [[cls_idx]], [tf.cast(weight, tf.float32)]
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

class FRAFeatureExtractor:
    """Extract FRA band features for industrial fault detection"""
    
    def __init__(self):
        # Industrial frequency bands for transformer FRA
        self.frequency_bands = {
            'VLF': (20, 1e3),      # Very Low Frequency - winding resistance
            'LF': (1e3, 1e5),      # Low Frequency - core/winding interaction  
            'MF': (1e5, 2e6),      # Medium Frequency - winding capacitance
            'HF': (2e6, 25e6)      # High Frequency - bushing/lead effects
        }
    
    def extract_features(self, frequencies, magnitude_db, phase_deg, transformer_meta=None):
        """Extract statistical features from FRA frequency bands"""
        features = {}
        
        for band_name, (f_min, f_max) in self.frequency_bands.items():
            # Find indices in frequency band
            mask = (frequencies >= f_min) & (frequencies <= f_max)
            
            if np.sum(mask) > 0:
                band_mag = magnitude_db[mask]
                band_phase = phase_deg[mask]
                
                # Statistical features for each band
                features.update({
                    f'mag_mean_{band_name}': np.mean(band_mag),
                    f'mag_std_{band_name}': np.std(band_mag),
                    f'mag_min_{band_name}': np.min(band_mag),
                    f'mag_max_{band_name}': np.max(band_mag),
                    f'mag_range_{band_name}': np.max(band_mag) - np.min(band_mag),
                    f'ph_mean_{band_name}': np.mean(band_phase),
                    f'ph_std_{band_name}': np.std(band_phase),
                    f'ph_min_{band_name}': np.min(band_phase),
                    f'ph_max_{band_name}': np.max(band_phase),
                    f'ph_range_{band_name}': np.max(band_phase) - np.min(band_phase)
                })
            else:
                # Fill with zeros if no data in band
                for metric in ['mag_mean', 'mag_std', 'mag_min', 'mag_max', 'mag_range',
                             'ph_mean', 'ph_std', 'ph_min', 'ph_max', 'ph_range']:
                    features[f'{metric}_{band_name}'] = 0.0
        
        # Add transformer metadata if provided
        if transformer_meta:
            features.update({
                'voltage_kv': transformer_meta.get('voltage_kv', 132),
                'power_mva': transformer_meta.get('power_mva', 50),
                'age_years': transformer_meta.get('age_years', 15)
            })
        
        return features

class IndustrialFRAPipeline:
    """Complete pipeline for industrial FRA fault detection"""
    
    def __init__(self, model_path="best_industrial_fra_model.h5", 
                 scaler_path="fra_scaler.pkl", 
                 encoder_path="fra_fault_encoder.pkl",
                 metadata_path="fra_model_metadata.json"):
        
        self.feature_extractor = FRAFeatureExtractor()
        self.model = None
        self.scaler = None
        self.encoder = None
        self.metadata = None
        
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        self.metadata_path = metadata_path
        
        # Load components
        self.load_components()
        
        # Industrial fault categories
        self.industrial_faults = {
            'healthy': {
                'description': 'No fault detected',
                'severity_threshold': 0.05,
                'action_required': 'Continue normal monitoring'
            },
            'thermal_fault': {
                'description': 'Overheating/cooling issues',
                'severity_threshold': 0.3,
                'action_required': 'Check cooling system and oil circulation'
            },
            'electrical_fault': {
                'description': 'Electrical system problems', 
                'severity_threshold': 0.4,
                'action_required': 'Perform insulation and winding tests'
            },
            'mechanical_fault': {
                'description': 'Physical/structural damage',
                'severity_threshold': 0.3,
                'action_required': 'Check for vibrations and loose connections'
            },
            'oil_degradation': {
                'description': 'Oil quality issues',
                'severity_threshold': 0.2,
                'action_required': 'Oil analysis and potential replacement'
            },
            'partial_discharge': {
                'description': 'Insulation breakdown',
                'severity_threshold': 0.5,
                'action_required': 'Urgent insulation inspection required'
            }
        }
        
        # Severity and criticality levels
        self.severity_levels = {
            'normal': (0.0, 0.2),
            'minor': (0.2, 0.4), 
            'moderate': (0.4, 0.6),
            'major': (0.6, 0.8),
            'critical': (0.8, 1.0)
        }
        
        self.criticality_levels = {
            'low': (0.0, 0.3),
            'medium': (0.3, 0.6),
            'high': (0.6, 0.8),
            'critical': (0.8, 1.0)
        }
    
    def load_components(self):
        """Load trained model and preprocessing components"""
        
        # Load model
        try:
            print(f"Loading FRA model from {self.model_path}...")
            self.model = tf.keras.models.load_model(
                self.model_path, 
                custom_objects={'FocalLoss': FocalLoss}
            )
            print("✅ FRA model loaded successfully")
        except Exception as e:
            print(f"⚠️  Error loading model via load_model: {e}")
            # Fallback: handle directory with config.json + model.weights.h5
            try:
                if os.path.isdir(self.model_path):
                    config_path = os.path.join(self.model_path, 'config.json')
                    weights_path = os.path.join(self.model_path, 'model.weights.h5')
                    if os.path.exists(config_path) and os.path.exists(weights_path):
                        print("Attempting fallback: reconstruct model from config.json and weights...")
                        with open(config_path, 'r', encoding='utf-8') as cf:
                            config_json = cf.read()
                        model = tf.keras.models.model_from_json(
                            config_json,
                            custom_objects={'FocalLoss': FocalLoss}
                        )
                        model.load_weights(weights_path)
                        self.model = model
                        print("✅ FRA model reconstructed from config + weights")
                    else:
                        raise FileNotFoundError("config.json or model.weights.h5 not found for fallback")
                else:
                    raise
            except Exception as e2:
                print(f"❌ Model load failed after fallback: {e2}")
                sys.exit(1)
        
        # Load scaler
        try:
            print(f"Loading scaler from {self.scaler_path}...")
            self.scaler = joblib.load(self.scaler_path)
            print("✅ Scaler loaded successfully")
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            sys.exit(1)
        
        # Load encoder
        try:
            print(f"Loading encoder from {self.encoder_path}...")
            self.encoder = joblib.load(self.encoder_path)
            print("✅ Encoder loaded successfully")
            print(f"   Industrial fault classes: {self.encoder.classes_}")
        except Exception as e:
            print(f"❌ Error loading encoder: {e}")
            sys.exit(1)
        
        # Load metadata
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print("✅ Model metadata loaded")
            else:
                print("⚠️  Model metadata not found")
        except Exception as e:
            print(f"⚠️  Error loading metadata: {e}")
    
    def parse_fra_csv(self, csv_path):
        """Parse FRA CSV file and validate format"""
        try:
            print(f"📄 Loading FRA data from {csv_path}...")
            df = pd.read_csv(csv_path)
            
            # Check required columns
            required_cols = ['frequency', 'magnitude_db', 'phase_deg']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                # Try alternative column names
                alt_names = {
                    'frequency': ['freq', 'f', 'frequency_hz'],
                    'magnitude_db': ['magnitude', 'mag', 'mag_db', 'amplitude_db'],
                    'phase_deg': ['phase', 'ph', 'phase_degree', 'phase_rad']
                }
                
                for req_col in missing_cols:
                    found = False
                    for alt in alt_names.get(req_col, []):
                        if alt in df.columns:
                            df = df.rename(columns={alt: req_col})
                            found = True
                            break
                    
                    if not found:
                        raise ValueError(f"Required column '{req_col}' not found in CSV. "
                                       f"Available columns: {list(df.columns)}")
            
            # Validate data
            frequencies = df['frequency'].values
            magnitude_db = df['magnitude_db'].values
            phase_deg = df['phase_deg'].values
            
            # Check for reasonable ranges
            if np.min(frequencies) < 1 or np.max(frequencies) > 100e6:
                print("⚠️  Warning: Frequency range seems unusual")
            
            if len(frequencies) < 50:
                print("⚠️  Warning: Very few frequency points, results may be unreliable")
            
            print(f"✅ Parsed FRA data:")
            print(f"   Frequency points: {len(frequencies)}")
            print(f"   Frequency range: {np.min(frequencies):.1f} - {np.max(frequencies):.1e} Hz")
            print(f"   Magnitude range: {np.min(magnitude_db):.1f} - {np.max(magnitude_db):.1f} dB")
            print(f"   Phase range: {np.min(phase_deg):.1f} - {np.max(phase_deg):.1f}°")
            
            return frequencies, magnitude_db, phase_deg
            
        except Exception as e:
            raise Exception(f"Error parsing FRA CSV: {e}")
    
    def extract_features(self, frequencies, magnitude_db, phase_deg, transformer_meta=None):
        """Extract FRA features for model prediction"""
        print("🔄 Extracting FRA band features...")
        
        features = self.feature_extractor.extract_features(
            frequencies, magnitude_db, phase_deg, transformer_meta
        )
        
        # Convert to model input format
        if self.metadata and 'feature_columns' in self.metadata['model_info']:
            feature_cols = self.metadata['model_info']['feature_columns']
            X = np.array([features.get(col, 0.0) for col in feature_cols]).reshape(1, -1)
        else:
            # Fallback to sorted feature names
            feature_names = sorted(features.keys())
            X = np.array([features[name] for name in feature_names]).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        print(f"✅ Extracted {X_scaled.shape[1]} FRA features")
        return X_scaled, features
    
    def predict(self, X_scaled, use_tta=True, n_tta=5):
        """Make industrial fault predictions with TTA"""
        
        if use_tta:
            print(f"🔮 Making predictions with Test Time Augmentation ({n_tta} iterations)...")
            all_preds = []
            
            for i in range(n_tta):
                # Add small noise for TTA
                X_aug = X_scaled + np.random.normal(0, 0.005, X_scaled.shape)
                preds = self.model.predict(X_aug, verbose=0)
                all_preds.append(preds)
            
            # Ensemble predictions
            ensemble_preds = [np.mean([pred[i] for pred in all_preds], axis=0) for i in range(4)]
        else:
            print("🔮 Making predictions...")
            ensemble_preds = self.model.predict(X_scaled, verbose=0)
        
        return ensemble_preds
    
    def interpret_results(self, raw_predictions, confidence_threshold=0.7):
        """Interpret model predictions into industrial fault diagnosis"""
        
        anomaly_prob = float(raw_predictions[0][0][0])
        fault_probs = raw_predictions[1][0]
        severity_score = float(raw_predictions[2][0][0])
        criticality_score = float(raw_predictions[3][0][0])
        
        # Get fault predictions
        fault_idx = np.argmax(fault_probs)
        fault_confidence = float(fault_probs[fault_idx])
        fault_type = self.encoder.classes_[fault_idx]
        
        # Get top 3 fault candidates
        top3_indices = np.argsort(fault_probs)[-3:][::-1]
        top3_faults = [(self.encoder.classes_[idx], float(fault_probs[idx])) 
                      for idx in top3_indices]
        
        # Determine fault state
        is_anomaly = anomaly_prob > 0.5
        
        # Determine severity level
        severity_level = 'normal'
        for level, (low, high) in self.severity_levels.items():
            if low <= severity_score < high:
                severity_level = level
                break
        
        # Determine criticality level
        criticality_level = 'low'
        for level, (low, high) in self.criticality_levels.items():
            if low <= criticality_score < high:
                criticality_level = level
                break
        
        # Reliability assessment
        reliability = 'high' if fault_confidence >= confidence_threshold else 'medium'
        if fault_confidence < 0.5:
            reliability = 'low'
        
        # Generate diagnosis
        diagnosis = {
            'anomaly_detected': is_anomaly,
            'anomaly_probability': round(anomaly_prob, 3),
            'primary_fault': {
                'type': fault_type if is_anomaly else 'healthy',
                'confidence': round(fault_confidence, 3),
                'description': self.industrial_faults.get(fault_type, {}).get('description', 'Unknown fault'),
                'reliability': reliability
            },
            'fault_candidates': [
                {'type': fault, 'probability': round(prob, 3)} 
                for fault, prob in top3_faults
            ],
            'severity': {
                'score': round(severity_score, 3),
                'level': severity_level
            },
            'criticality': {
                'score': round(criticality_score, 3),
                'level': criticality_level
            },
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return diagnosis
    
    def generate_recommendations(self, diagnosis):
        """Generate industrial maintenance recommendations"""
        
        fault_type = diagnosis['primary_fault']['type']
        severity_level = diagnosis['severity']['level']
        criticality_level = diagnosis['criticality']['level']
        
        recommendations = {
            'immediate_actions': [],
            'maintenance_actions': [],
            'monitoring_recommendations': [],
            'priority': 'low',
            'timeframe': 'next_maintenance'
        }
        
        if fault_type == 'healthy':
            recommendations['immediate_actions'].append("✅ No immediate action required")
            recommendations['monitoring_recommendations'].append("Continue routine monitoring")
            
        else:
            fault_config = self.industrial_faults.get(fault_type, {})
            base_action = fault_config.get('action_required', 'Investigate fault')
            
            # Set priority based on criticality
            if criticality_level == 'critical':
                recommendations['priority'] = 'critical'
                recommendations['timeframe'] = 'immediate'
                recommendations['immediate_actions'].append(f"🚨 URGENT: {base_action}")
                
            elif criticality_level == 'high':
                recommendations['priority'] = 'high'
                recommendations['timeframe'] = 'within_24_hours'
                recommendations['immediate_actions'].append(f"⚠️ HIGH PRIORITY: {base_action}")
                
            elif criticality_level == 'medium':
                recommendations['priority'] = 'medium'
                recommendations['timeframe'] = 'within_week'
                recommendations['maintenance_actions'].append(f"📋 SCHEDULED: {base_action}")
                
            else:
                recommendations['maintenance_actions'].append(f"📝 ROUTINE: {base_action}")
            
            # Fault-specific recommendations
            if fault_type == 'thermal_fault':
                recommendations['maintenance_actions'].extend([
                    "Check cooling system functionality and oil circulation",
                    "Verify cooling fans and radiator cleanliness",
                    "Monitor oil temperature trends",
                    "Consider thermal imaging inspection"
                ])
                
            elif fault_type == 'electrical_fault':
                recommendations['maintenance_actions'].extend([
                    "Perform insulation resistance testing",
                    "Check for voltage imbalances and harmonics",
                    "Verify electrical connections and contacts",
                    "Consider power factor and tan delta testing"
                ])
                
            elif fault_type == 'mechanical_fault':
                recommendations['maintenance_actions'].extend([
                    "Check for unusual vibrations or noise",
                    "Inspect physical mounting and connections",
                    "Perform acoustic monitoring for discharge",
                    "Verify core grounding integrity"
                ])
                
            elif fault_type == 'oil_degradation':
                recommendations['maintenance_actions'].extend([
                    "Perform comprehensive dissolved gas analysis (DGA)",
                    "Check oil quality parameters (moisture, acidity, etc.)",
                    "Inspect for oil leaks and moisture ingress",
                    "Consider oil filtration or replacement"
                ])
                
            elif fault_type == 'partial_discharge':
                recommendations['immediate_actions'].extend([
                    "🔍 INVESTIGATE: Potential insulation breakdown detected",
                    "Perform detailed partial discharge testing"
                ])
                recommendations['maintenance_actions'].extend([
                    "Comprehensive insulation assessment required",
                    "Check for moisture ingress and contamination",
                    "Monitor gas generation rates",
                    "Consider offline diagnostic testing"
                ])
            
            # Monitoring recommendations
            if severity_level in ['major', 'critical']:
                recommendations['monitoring_recommendations'].append("Increase monitoring frequency")
                
            if diagnosis['primary_fault']['reliability'] == 'low':
                recommendations['monitoring_recommendations'].append("Consider additional diagnostic tests for confirmation")
        
        return recommendations
    
    def run_pipeline(self, csv_path, transformer_meta=None, use_tta=True, output_path=None):
        """Run complete FRA analysis pipeline"""
        
        print("=" * 70)
        print("🔬 RUNNING INDUSTRIAL FRA FAULT DETECTION PIPELINE")
        print("=" * 70)
        
        try:
            # 1. Parse FRA CSV
            frequencies, magnitude_db, phase_deg = self.parse_fra_csv(csv_path)
            
            # 2. Extract features
            X_scaled, raw_features = self.extract_features(frequencies, magnitude_db, phase_deg, transformer_meta)
            
            # 3. Make predictions
            raw_predictions = self.predict(X_scaled, use_tta=use_tta)
            
            # 4. Interpret results
            diagnosis = self.interpret_results(raw_predictions)
            
            # 5. Generate recommendations  
            recommendations = self.generate_recommendations(diagnosis)
            
            # 6. Compile results
            pipeline_results = {
                'input_file': os.path.basename(csv_path),
                'transformer_metadata': transformer_meta,
                'fra_data_summary': {
                    'frequency_points': len(frequencies),
                    'frequency_range_hz': [float(np.min(frequencies)), float(np.max(frequencies))],
                    'magnitude_range_db': [float(np.min(magnitude_db)), float(np.max(magnitude_db))],
                    'phase_range_deg': [float(np.min(phase_deg)), float(np.max(phase_deg))]
                },
                'diagnosis': diagnosis,
                'recommendations': recommendations,
                'analysis_metadata': {
                    'model_used': 'industrial_fra_model',
                    'tta_enabled': use_tta,
                    'feature_count': X_scaled.shape[1],
                    'industrial_fault_categories': self.encoder.classes_.tolist()
                }
            }
            
            # 7. Save results
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(pipeline_results, f, indent=2)
                print(f"📄 Results saved to {output_path}")
            
            # 8. Display results
            self.display_results(pipeline_results)
            
            return pipeline_results
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            return None
    
    def display_results(self, results):
        """Display analysis results in formatted output"""
        
        print("\n" + "=" * 70)
        print("📊 INDUSTRIAL FRA ANALYSIS RESULTS")
        print("=" * 70)
        
        diagnosis = results['diagnosis']
        recommendations = results['recommendations']
        
        # Main diagnosis
        print(f"\n🔍 PRIMARY DIAGNOSIS:")
        if diagnosis['anomaly_detected']:
            print(f"   ⚠️  FAULT DETECTED: {diagnosis['primary_fault']['type'].upper()}")
            print(f"   📝 Description: {diagnosis['primary_fault']['description']}")
            print(f"   🎯 Confidence: {diagnosis['primary_fault']['confidence']} ({diagnosis['primary_fault']['reliability']} reliability)")
        else:
            print(f"   ✅ HEALTHY: No significant fault detected")
        
        print(f"   🚨 Anomaly Probability: {diagnosis['anomaly_probability']}")
        
        # Severity and criticality
        print(f"\n📈 SEVERITY & CRITICALITY:")
        print(f"   📊 Severity: {diagnosis['severity']['score']} ({diagnosis['severity']['level']})")
        print(f"   🎯 Criticality: {diagnosis['criticality']['score']} ({diagnosis['criticality']['level']})")
        
        # Top fault candidates
        print(f"\n🏆 TOP FAULT CANDIDATES:")
        for i, candidate in enumerate(diagnosis['fault_candidates'][:3], 1):
            print(f"   {i}. {candidate['type']}: {candidate['probability']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   🔥 Priority: {recommendations['priority'].upper()}")
        print(f"   ⏰ Timeframe: {recommendations['timeframe'].replace('_', ' ').title()}")
        
        if recommendations['immediate_actions']:
            print(f"\n   🚨 IMMEDIATE ACTIONS:")
            for action in recommendations['immediate_actions']:
                print(f"      • {action}")
        
        if recommendations['maintenance_actions']:
            print(f"\n   🔧 MAINTENANCE ACTIONS:")
            for action in recommendations['maintenance_actions']:
                print(f"      • {action}")
        
        if recommendations['monitoring_recommendations']:
            print(f"\n   📊 MONITORING:")
            for rec in recommendations['monitoring_recommendations']:
                print(f"      • {rec}")
        
        print("\n" + "=" * 70)

def main():
    """Run FRA pipeline from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Industrial FRA Fault Detection Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input FRA CSV file path")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--voltage", type=float, default=132, help="Transformer voltage (kV)")
    parser.add_argument("--power", type=float, default=50, help="Transformer power (MVA)")
    parser.add_argument("--age", type=int, default=15, help="Transformer age (years)")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test Time Augmentation")
    parser.add_argument("--model", default="best_industrial_fra_model.h5", help="Model file path")
    parser.add_argument("--scaler", default="fra_scaler.pkl", help="Scaler file path")
    parser.add_argument("--encoder", default="fra_fault_encoder.pkl", help="Encoder file path")
    
    args = parser.parse_args()
    
    # Transformer metadata
    transformer_meta = {
        'voltage_kv': args.voltage,
        'power_mva': args.power,
        'age_years': args.age
    }
    
    # Set output path
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"fra_analysis_{timestamp}.json"
    
    # Create pipeline
    pipeline = IndustrialFRAPipeline(
        model_path=args.model,
        scaler_path=args.scaler,
        encoder_path=args.encoder
    )
    
    # Run analysis
    results = pipeline.run_pipeline(
        csv_path=args.input,
        transformer_meta=transformer_meta,
        use_tta=not args.no_tta,
        output_path=args.output
    )
    
    if results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📄 Results saved to: {args.output}")

if __name__ == "__main__":
    main()