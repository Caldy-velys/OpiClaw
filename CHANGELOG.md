# Changelog

All notable changes to OpiClaw will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX

### Added
- **Initial Release**: Complete OpiClaw panoptic segmentation model
- **Marine ConvViT Blocks**: Hybrid CNN-Transformer with radial position encoding
- **Language-Guided Refinement System (LGRS)**: Natural language task specification
- **Evidential Uncertainty**: Dirichlet-based uncertainty quantification
- **Polar BEV Projection**: Efficient sonar point cloud processing
- **Panoptic Heads**: Complete semantic + instance segmentation pipeline
- **Marine Scene Generation**: Synthetic bathymetry for testing
- **Comprehensive Documentation**: README, API docs, and research summary
- **Testing Framework**: Unit tests for all core components
- **Demo Scripts**: Simple and comprehensive demonstration examples

### Architecture Features
- 513,551 parameters optimized for AUV deployment
- Real-time capable architecture (target <100ms inference)
- Marine-specific vocabulary (20 specialized terms)
- Uncertainty-aware navigation support
- Multi-scene handling (hydrothermal, debris, seafloor)

### Technical Improvements
- Fixed tensor dimension mismatches in BEV projection
- Proper evidential loss implementation
- Numerically stable uncertainty calculations
- Efficient batch processing for training

### Research Contributions
- First application of evidential deep learning to sonar segmentation
- Novel marine-specific ConvViT architecture
- Language-guided underwater navigation system
- Uncertainty-aware panoptic fusion for maritime operations

---

## [Unreleased]

### Planned
- Real NOAA bathymetry dataset integration
- Model quantization for edge deployment
- Multi-modal sensor fusion (side-scan + SAS)
- Temporal consistency for navigation sequences
- Open-set unknown object detection
- Performance benchmarking on Marine Debris FLS dataset

---

*For detailed technical analysis, see [docs/OPICLAW_REVIEW_SUMMARY.md](docs/OPICLAW_REVIEW_SUMMARY.md)* 