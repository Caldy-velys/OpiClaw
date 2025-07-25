# 🌊 OpiClaw: Deep-Sea Panoptic Segmentation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/)

> **OpiClaw**: Uncertainty-Aware Panoptic Segmentation for Autonomous Underwater Vehicle Navigation

## 🎯 Overview

OpiClaw is a cutting-edge deep learning model for panoptic segmentation of deep-sea sensor data, specifically designed for autonomous underwater vehicle (AUV) navigation. The model combines **evidential deep learning**, **language-guided refinement**, and **marine-specific adaptations** to provide robust perception in challenging underwater environments.

### 🌟 Key Features

- **🔬 Evidential Uncertainty**: Dirichlet-based uncertainty quantification for high-stakes navigation
- **🗣️ Language-Guided Refinement**: Natural language task specification for marine operations
- **🌊 Marine ConvViT**: Hybrid CNN-Transformer with radial position encoding for sonar data
- **📊 Panoptic Segmentation**: Complete semantic + instance segmentation pipeline
- **⚡ Real-Time Ready**: Optimized for AUV deployment with efficient architecture

## 🏗️ Architecture

```
OpiClaw Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Sonar Input   │───▶│  Polar BEV Proj  │───▶│  Marine ConvViT  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Language Prompt│───▶│   LGRS Fusion    │◀───│   U-Net Decoder │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Uncertainty    │◀───│  Evidential Head │◀───│  Panoptic Heads │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/opiclaw.git
cd opiclaw

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from src.models import PanopticOpiClaw
from src.utils import project_to_polar_bev, generate_marine_scene

# Initialize model
model = PanopticOpiClaw(in_channels=1, num_classes=4, embed_dim=64)

# Generate synthetic marine scene
points = generate_marine_scene(1000, "hydrothermal_field")
bev = project_to_polar_bev(points)

# Language-guided inference
with torch.no_grad():
    outputs = model(bev, prompt="find hydrothermal_vent")
    
    # Extract results
    semantic_pred = outputs['semantic_prob'].argmax(1)
    uncertainty = outputs['semantic_uncertainty']
    instance_centers = outputs['instance_center']
```

### Demo Scripts

```bash
# Run lightweight demonstration
python demo/opiclaw_demo_simple.py

# Run comprehensive training demo
python demo/opiclaw_training_demo.py

# Test model upgrades
python src/opiclaw_upgrade.py
```

## 📊 Performance

### Model Specifications
- **Parameters**: 513,551 (efficient for AUV deployment)
- **Input**: Sonar point clouds → Polar BEV representation
- **Output**: Panoptic segmentation + uncertainty maps
- **Architecture**: ConvViT + LGRS fusion + Evidential heads

### Key Metrics
- **Uncertainty Calibration**: ECE < 0.05 (target)
- **Panoptic Quality**: PQ > 0.6 (on synthetic data)
- **Inference Speed**: < 100ms (target for real-time AUV)

## 🔬 Research Applications

### Deep-Sea Navigation
- **Obstacle Avoidance**: Real-time hazard detection and path planning
- **Resource Exploration**: Hydrothermal vent and mineral deposit identification
- **Environmental Monitoring**: Marine ecosystem mapping and assessment
- **Infrastructure Inspection**: Pipeline and cable monitoring

### Marine Science
- **Bathymetric Mapping**: High-resolution seafloor topography
- **Biological Surveys**: Coral reef and deep-sea organism detection
- **Geological Studies**: Seamount and trench analysis
- **Archaeological Discovery**: Shipwreck and artifact identification

## 📁 Repository Structure

```
opiclaw/
├── src/                    # Core model implementation
│   ├── models/            # Neural network architectures
│   ├── utils/             # Utility functions
│   └── data/              # Data processing modules
├── demo/                  # Demonstration scripts
├── docs/                  # Documentation and research notes
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation
└── README.md             # This file
```

## 🛠️ Development

### Environment Setup

```bash
# Create virtual environment
python -m venv opiclaw_env
source opiclaw_env/bin/activate  # Linux/Mac
# or
opiclaw_env\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Style

```bash
# Format code
black src/ demo/ tests/

# Lint code
flake8 src/ demo/ tests/
```

## 📚 Documentation

- **[Technical Report](docs/OPICLAW_REVIEW_SUMMARY.md)**: Comprehensive model analysis
- **[API Reference](docs/api.md)**: Detailed function documentation
- **[Research Notes](docs/research/)**: Background and methodology
- **[Deployment Guide](docs/deployment.md)**: AUV integration guide

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Research Foundation**: Built upon EvLPSNet and Panoptic-PolarNet architectures
- **Marine Domain**: Inspired by NOAA bathymetry and deep-sea exploration challenges
- **Open Source**: Leverages PyTorch, NumPy, and the broader ML community

## 📞 Contact

- **Project Lead**: [Your Name](mailto:your.email@example.com)
- **Research Questions**: [Open an Issue](https://github.com/yourusername/opiclaw/issues)
- **Collaboration**: [Start a Discussion](https://github.com/yourusername/opiclaw/discussions)

## 🌊 Citation

If you use OpiClaw in your research, please cite:

```bibtex
@article{opiclaw2025,
  title={OpiClaw: Uncertainty-Aware Panoptic Segmentation for Deep-Sea Navigation},
  author={Your Name},
  journal={arXiv preprint arXiv:2025.XXXXX},
  year={2025}
}
```

---

**Ready to explore the abyss? 🌊🤖**

*OpiClaw: Where cutting-edge AI meets the deepest frontiers of our planet.* 