# Contributing to OpiClaw

Thank you for your interest in contributing to OpiClaw! This document provides guidelines for contributing to the project.

##  Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Git

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/opiclaw.git
cd opiclaw

# Create virtual environment
python -m venv opiclaw_env
source opiclaw_env/bin/activate  # Linux/Mac
# or
opiclaw_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

##  Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py
```

### 4. Code Quality Checks
```bash
# Format code
black src/ demo/ tests/

# Lint code
flake8 src/ demo/ tests/

# Type checking
mypy src/
```

### 5. Commit Your Changes
```bash
git add .
git commit -m "feat: add new feature description"
```

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

##  Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and concise

### Commit Messages
Use conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

### Example
```
feat: add uncertainty calibration metrics

- Add ECE calculation function
- Implement calibration plotting utilities
- Update evaluation pipeline
```

##  Testing Guidelines

### Writing Tests
- Write tests for all new functionality
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies

### Test Structure
```python
def test_feature_name():
    """Test description"""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result is not None
    assert result.shape == expected_shape
```

##  Documentation

### Code Documentation
- Use Google-style docstrings
- Include type hints
- Provide usage examples

### Example
```python
def process_sonar_data(points: torch.Tensor, 
                      config: Dict[str, Any]) -> torch.Tensor:
    """Process sonar point cloud data.
    
    Args:
        points: Input point cloud of shape (N, 3)
        config: Processing configuration dictionary
        
    Returns:
        Processed tensor of shape (1, 1, H, W)
        
    Raises:
        ValueError: If points tensor is empty
    """
```

##  Areas for Contribution

### High Priority
- **Real Data Integration**: NOAA bathymetry dataset loading
- **Performance Optimization**: Model quantization for AUV deployment
- **Evaluation Metrics**: Panoptic Quality (PQ) implementation
- **Documentation**: API reference and tutorials

### Medium Priority
- **Multi-Modal Fusion**: Side-scan + synthetic aperture sonar
- **Temporal Consistency**: LSTM/Transformer for navigation
- **Open-Set Detection**: Unknown object identification
- **Visualization Tools**: Interactive plotting utilities

### Low Priority
- **Model Variants**: Different backbone architectures
- **Data Augmentation**: Physics-based sonar simulations
- **Deployment Tools**: Docker containers and deployment scripts

##  Reporting Issues

### Bug Reports
When reporting bugs, please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces

### Feature Requests
For feature requests, please include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Expected benefits

##  Pull Request Process

### Before Submitting
1. Ensure all tests pass
2. Update documentation
3. Add/update tests for new functionality
4. Follow code style guidelines
5. Update CHANGELOG.md if applicable

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] CHANGELOG updated
```

##  Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Email**: Contact maintainers directly for urgent matters

##  Recognition

Contributors will be recognized in:
- Project README
- Release notes
- Academic publications (if applicable)

Thank you for contributing to OpiClaw!  