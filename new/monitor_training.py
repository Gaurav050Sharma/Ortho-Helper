#!/usr/bin/env python3
"""
Training Progress Monitor
Monitor the DenseNet121 training progress
"""

import os
import time
import json
from datetime import datetime

def monitor_training_progress():
    """Monitor training progress by checking output files"""
    output_dir = "new"
    
    print("🔍 DenseNet121 Training Progress Monitor")
    print("=" * 50)
    
    while True:
        try:
            # Check if training log exists
            log_file = os.path.join(output_dir, 'training_log.csv')
            if os.path.exists(log_file):
                print(f"📊 Training log found: {log_file}")
                # Read last few lines of training log
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            print(f"   Latest entry: {lines[-1].strip()}")
                except:
                    pass
            
            # Check for saved models
            model_files = [f for f in os.listdir(output_dir) if f.endswith(('.h5', '.keras'))]
            if model_files:
                print(f"💾 Model files found: {len(model_files)}")
                for model_file in model_files[-3:]:  # Show last 3
                    print(f"   • {model_file}")
            
            # Check for evaluation results
            eval_file = os.path.join(output_dir, 'evaluation_results.json')
            if os.path.exists(eval_file):
                try:
                    with open(eval_file, 'r') as f:
                        results = json.load(f)
                        print(f"🎯 Training completed!")
                        print(f"   Test Accuracy: {results['test_accuracy']:.4f}")
                        print(f"   Test Precision: {results['test_precision']:.4f}")
                        print(f"   Test Recall: {results['test_recall']:.4f}")
                        print(f"   ROC AUC: {results['roc_auc']:.4f}")
                        break
                except:
                    pass
            
            print(f"⏰ Checking again in 30 seconds... ({datetime.now().strftime('%H:%M:%S')})")
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Error monitoring: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_training_progress()