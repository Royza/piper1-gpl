#!/usr/bin/env python3
"""
Simple test script to verify the Flask web server is working
"""

from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "Enhanced PiperTTS Training System is running!",
        "version": "1.1",
        "features": [
            "Web Interface",
            "Audio Processing",
            "AI Chat Integration",
            "TensorBoard",
            "Real-time Monitoring"
        ]
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": "2024-01-17"
    })

if __name__ == '__main__':
    print("🧪 Starting test server...")
    print("📊 Access at: http://localhost:5000")
    print("🏥 Health check at: http://localhost:5000/health")
    app.run(host='0.0.0.0', port=5000, debug=True)
