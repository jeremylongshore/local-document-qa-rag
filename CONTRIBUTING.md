# 🤝 Contributing to NEXUS

First off, thank you for considering contributing to NEXUS! 🎉

NEXUS is an open-source autonomous document intelligence system, and we welcome contributions from developers, researchers, and AI enthusiasts worldwide.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in NEXUS a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Examples of behavior that contributes to creating a positive environment:**
- 🤝 Using welcoming and inclusive language
- 🎯 Being respectful of differing viewpoints and experiences
- ✨ Gracefully accepting constructive criticism
- 🌟 Focusing on what is best for the community
- 💫 Showing empathy towards other community members

**Examples of unacceptable behavior:**
- ❌ Trolling, insulting/derogatory comments, and personal or political attacks
- ❌ Public or private harassment
- ❌ Publishing others' private information without permission
- ❌ Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by opening an issue or contacting the project team. All complaints will be reviewed and investigated promptly and fairly.

## 💡 How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

**Bug Report Template:**
```markdown
**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.9.5]
- NEXUS version: [e.g., 1.0.0]
- Ollama version: [e.g., 0.1.0]

**Description:**
A clear and concise description of the bug.

**Steps to Reproduce:**
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior:**
What you expected to happen.

**Actual Behavior:**
What actually happened.

**Screenshots:**
If applicable, add screenshots.

**Logs:**
```
Paste relevant logs here
```
```

### ✨ Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title** describing the suggestion
- **Step-by-step description** of the suggested enhancement
- **Specific examples** to demonstrate the steps
- **Explanation** of why this enhancement would be useful

### 🔧 Your First Code Contribution

Unsure where to begin? Look for these labels:

- `good first issue` - Simple issues perfect for beginners
- `help wanted` - Issues where we need community help
- `documentation` - Documentation improvements
- `performance` - Performance optimization opportunities

## 🚀 Development Setup

### Prerequisites

```bash
# System requirements
Python 3.9+
Git
Ollama

# Development tools
black (code formatter)
pytest (testing)
mypy (type checking)
```

### Local Development

1. **Fork the repository**
   ```bash
   # Click 'Fork' on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/nexus-rag.git
   cd nexus-rag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make your changes**
   ```bash
   # Edit files
   # Add tests
   # Update documentation
   ```

6. **Run tests**
   ```bash
   pytest tests/
   mypy app.py
   black .
   ```

7. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

8. **Push to GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Branch is up-to-date with main

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. **Automated checks** run on all PRs
2. **Code review** by maintainers
3. **Testing** in CI/CD pipeline
4. **Merge** when approved

## 🎨 Style Guidelines

### Python Code Style

```python
# Good example
class DocumentProcessor:
    """Process documents for NEXUS pipeline."""

    def __init__(self, config: dict):
        """Initialize processor with configuration."""
        self.config = config

    def process(self, document: str) -> List[str]:
        """Process document into chunks."""
        # Clear, commented logic
        chunks = self._split_document(document)
        return self._validate_chunks(chunks)
```

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions
- `chore`: Maintenance tasks

**Examples:**
```
feat(embeddings): add support for custom embedding models

fix(cache): resolve memory leak in LRU cache

docs(readme): update installation instructions

perf(search): optimize vector similarity search by 50%
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Keep README updated
- Document all public APIs

## 🧪 Testing Guidelines

### Test Structure

```python
# tests/test_document_processor.py
import pytest
from nexus.processor import DocumentProcessor

class TestDocumentProcessor:
    def test_process_valid_document(self):
        """Test processing of valid document."""
        processor = DocumentProcessor()
        result = processor.process("test document")
        assert len(result) > 0

    def test_process_empty_document(self):
        """Test handling of empty document."""
        processor = DocumentProcessor()
        with pytest.raises(ValueError):
            processor.process("")
```

### Coverage Requirements

- Minimum 80% code coverage
- All new features must include tests
- Bug fixes must include regression tests

## 🌟 Recognition

Contributors are recognized in:
- [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Release notes
- Project documentation

## 💬 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Discord**: Real-time chat (coming soon)

### Getting Help

- Check documentation first
- Search existing issues
- Ask in discussions
- Be patient and respectful

## 🎯 Areas for Contribution

### High Priority

- 🚀 **Performance**: Query optimization, caching strategies
- 🔒 **Security**: Authentication, encryption, audit logging
- 📚 **Document Loaders**: Support for more file formats
- 🌍 **Internationalization**: Multi-language support
- 🧪 **Testing**: Increase test coverage

### Feature Ideas

- Voice input/output
- Web UI improvements
- API endpoints
- Cloud storage integration
- Advanced analytics

## 📚 Resources

### Learning Materials

- [LangChain Documentation](https://langchain.com)
- [Ollama Documentation](https://ollama.ai)
- [ChromaDB Documentation](https://www.trychroma.com)
- [Streamlit Documentation](https://streamlit.io)

### Development Tools

- [Black Formatter](https://black.readthedocs.io)
- [Pytest](https://pytest.org)
- [MyPy](https://mypy-lang.org)

## 🙏 Thank You!

Every contribution, no matter how small, helps make NEXUS better. We appreciate your time and effort in improving this project.

Together, we're building the future of private, autonomous document intelligence! 🚀

---

*Questions? Open an issue or start a discussion!*