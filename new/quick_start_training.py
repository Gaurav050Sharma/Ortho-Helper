#!/usr/bin/env python3
"""
Quick Training Starter
=====================
Immediately starts comprehensive training with minimal prompts.
"""

import subprocess
import sys
import os

def main():
    print("🚀 Quick Training Starter")
    print("🏥 Medical X-Ray AI Comprehensive Training")
    print("=" * 50)
    
    print("⚡ Starting comprehensive training immediately...")
    print("📊 Training all 5 datasets × 5 architectures × 3 configurations = 75 combinations")
    print("⏱️ Estimated time: 8-12 hours (depends on hardware)")
    print("💾 Continuous saving enabled - training can be safely interrupted")
    print("📁 Results will be saved in 'new' directory")
    
    try:
        # Run the training pipeline directly
        exec(open('comprehensive_training_pipeline.py').read())
    except KeyboardInterrupt:
        print("\n⛔ Training interrupted by user")
    except FileNotFoundError:
        print("❌ comprehensive_training_pipeline.py not found!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()