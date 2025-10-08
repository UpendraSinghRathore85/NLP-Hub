# Contributing to NLP-Hub

We love your input! We want to make contributing to NLP-Hub as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](LICENSE) that covers the project.

## Report bugs using GitHub's [issue tracker](https://github.com/UpendraSinghRathore85/NLP-Hub/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/UpendraSinghRathore85/NLP-Hub/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
4. Install in development mode: `pip install -e .`
5. Install development dependencies: `pip install pytest black flake8`

## Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **Type hints** where possible

Run these before submitting:
```bash
black nlp_hub tests examples
flake8 nlp_hub tests examples
```

## Testing

Run tests with:
```bash
python -m pytest tests/
```

## License

By contributing, you agree that your contributions will be licensed under its MIT License.