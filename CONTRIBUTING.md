# Contributing to GoARIMA

Thank you for your interest in contributing to GoARIMA! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Pull Requests](#pull-requests)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all backgrounds and experience levels.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/goarima.git
   cd goarima
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/sartorproj/goarima.git
   ```

## Development Setup

### Prerequisites

- Go 1.21 or later
- [golangci-lint](https://golangci-lint.run/usage/install/) for linting

### Building

```bash
go build ./...
```

### Running Tests

```bash
go test ./...
```

### Running Linter

```bash
golangci-lint run
```

## Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the [code style guidelines](#code-style)

3. Add tests for new functionality

4. Ensure all tests pass:
   ```bash
   go test ./...
   ```

5. Run the linter:
   ```bash
   golangci-lint run
   ```

6. Commit your changes following the [commit message guidelines](#commit-messages)

## Testing

- Write tests for all new functionality
- Ensure existing tests still pass
- Aim for good test coverage
- Use table-driven tests where appropriate

Example test structure:

```go
func TestMyFunction(t *testing.T) {
    tests := []struct {
        name     string
        input    float64
        expected float64
    }{
        {"positive", 1.0, 2.0},
        {"negative", -1.0, -2.0},
        {"zero", 0.0, 0.0},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := MyFunction(tt.input)
            if result != tt.expected {
                t.Errorf("got %v, want %v", result, tt.expected)
            }
        })
    }
}
```

## Code Style

- Follow standard Go conventions and idioms
- Use `go fmt` to format your code
- Use meaningful variable and function names
- Add comments for exported functions and types
- Keep functions focused and small
- Handle errors explicitly

### Documentation

- All exported functions, types, and constants must have doc comments
- Doc comments should start with the name of the element being documented
- Use complete sentences with proper punctuation

```go
// Predict generates forecasts for the specified number of steps ahead.
// It returns the forecasted values and any error encountered during prediction.
func (m *Model) Predict(steps int) ([]float64, error) {
    // ...
}
```

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(arima): add support for seasonal differencing

fix(sarima): correct integration step in forecast

docs: update installation instructions

test(stats): add tests for KPSS stationarity test
```

## Pull Requests

1. Update your branch with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request on GitHub

4. Fill out the PR template completely

5. Wait for review and address any feedback

### PR Guidelines

- Keep PRs focused on a single change
- Write a clear description of what and why
- Reference any related issues
- Ensure CI checks pass
- Be responsive to review feedback

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: How to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Go version, OS, etc.
6. **Code Sample**: Minimal code to reproduce (if applicable)

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

## Questions?

If you have questions, feel free to:

- Open a GitHub issue
- Start a discussion in the repository

Thank you for contributing! 🎉
