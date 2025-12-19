#!/usr/bin/env python3
"""
Training Progress Monitor
========================
Monitor the comprehensive training pipeline progress in real-time.
"""

import os
import json
import time
from datetime import datetime

def monitor_training():
    """Monitor training progress continuously"""
    progress_file = "new/training_progress.json"
    
    print("📊 Training Progress Monitor")
    print("🔄 Monitoring comprehensive training pipeline...")
    print("⌨️ Press Ctrl+C to stop monitoring")
    print("=" * 60)
    
    last_completed = 0
    
    try:
        while True:
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                    
                    total_combinations = 75  # 5 datasets × 5 architectures × 3 configs
                    completed = len([k for k, v in progress.items() if v['status'] == 'completed'])
                    failed = len([k for k, v in progress.items() if v['status'] == 'failed'])
                    in_progress = len([k for k, v in progress.items() if v['status'] == 'started'])
                    
                    # Clear screen (works on most terminals)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    print("📊 COMPREHENSIVE TRAINING PROGRESS MONITOR")
                    print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("=" * 60)
                    
                    print(f"📈 OVERALL PROGRESS:")
                    print(f"   ✅ Completed: {completed}/{total_combinations} ({completed/total_combinations*100:.1f}%)")
                    print(f"   ❌ Failed: {failed}")
                    print(f"   🔄 In Progress: {in_progress}")
                    print(f"   ⏳ Remaining: {total_combinations - completed - failed}")
                    
                    # Show progress bar
                    progress_percentage = completed / total_combinations
                    bar_length = 50
                    filled_length = int(bar_length * progress_percentage)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"   📊 [{bar}] {progress_percentage*100:.1f}%")
                    
                    if completed > last_completed:
                        print(f"\n🎉 NEW COMPLETIONS (+{completed - last_completed}):")
                        last_completed = completed
                    
                    # Show recent activity
                    if progress:
                        print(f"\n📋 RECENT ACTIVITY:")
                        recent_items = sorted(progress.items(), 
                                            key=lambda x: x[1].get('timestamp', ''), 
                                            reverse=True)[:5]
                        
                        for combo_key, combo_data in recent_items:
                            status = combo_data.get('status', 'unknown')
                            dataset = combo_data.get('dataset', 'unknown')
                            arch = combo_data.get('architecture', 'unknown')
                            config = combo_data.get('configuration', 'unknown')
                            
                            status_emoji = {"completed": "✅", "failed": "❌", "started": "🔄"}.get(status, "❓")
                            
                            if status == 'completed' and 'results' in combo_data:
                                accuracy = combo_data['results'].get('test_accuracy', 0)
                                print(f"   {status_emoji} {dataset}_{arch}_{config}: {accuracy*100:.1f}%")
                            else:
                                print(f"   {status_emoji} {dataset}_{arch}_{config}: {status}")
                    
                    # Show best performers
                    completed_results = [(k, v) for k, v in progress.items() 
                                       if v['status'] == 'completed' and 'results' in v]
                    
                    if completed_results:
                        print(f"\n🏆 TOP PERFORMERS:")
                        top_performers = sorted(completed_results, 
                                              key=lambda x: x[1]['results'].get('test_accuracy', 0), 
                                              reverse=True)[:3]
                        
                        for i, (combo_key, combo_data) in enumerate(top_performers, 1):
                            dataset = combo_data.get('dataset', 'unknown')
                            arch = combo_data.get('architecture', 'unknown') 
                            config = combo_data.get('configuration', 'unknown')
                            accuracy = combo_data['results'].get('test_accuracy', 0)
                            medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else "🏆"
                            print(f"   {medal} {dataset}_{arch}_{config}: {accuracy*100:.1f}%")
                    
                    print(f"\n📁 Results Directory: new/")
                    print(f"🔄 Next update in 30 seconds...")
                    
                except Exception as e:
                    print(f"⚠️ Error reading progress: {e}")
            else:
                print(f"📊 Waiting for training to start...")
                print(f"📁 Looking for: {progress_file}")
            
            time.sleep(30)  # Update every 30 seconds
    
    except KeyboardInterrupt:
        print(f"\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    monitor_training()