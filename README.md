# Chebyshev Interpolation Engine (Java)

## Overview

This project implements polynomial interpolation using **Chebyshev nodes** to reduce oscillation and minimize interpolation error (Runge phenomenon).  

The program generates Chebyshev-distributed nodes over a specified interval, constructs interpolating polynomials of varying degrees, computes approximation error, and visualizes results using **JFreeChart**.

---

## Motivation

Polynomial interpolation using equally spaced nodes can lead to large oscillations near interval endpoints.  

Chebyshev nodes cluster near the boundaries of the interval, which improves numerical stability and reduces maximum interpolation error.  

This project explores that behavior through both quantitative error analysis and graphical comparison.

---

## Features

- Chebyshev node generation over interval [a, b]
- Polynomial interpolation for configurable degree
- Dense-grid error evaluation
- Maximum error reporting
- Graphical visualization of:
  - Original function
  - Interpolating polynomial(s)
  - Error comparison

---

## Example Output

### Interpolation Visualization
![Interpolation Graph](docs/Sample\ Program\ Graph\ Output.png)


---

## How It Works

1. Generate Chebyshev nodes using the cosine distribution formula.
2. Evaluate the target function at each node.
3. Construct the interpolating polynomial.
4. Evaluate error across a dense sampling grid.
5. Plot the function and approximation using JFreeChart.

---

## Compile & Run

### Mac / Linux
```bash
javac -cp ".:lib/jfreechart-1.5.6.jar" ChebyshevEngine.java
java -cp ".:lib/jfreechart-1.5.6.jar" ChebyshevEngine

### Windows
```bash
javac -cp ".;lib/jfreechart-1.5.6.jar" ChebyshevEngine.java
java -cp ".;lib/jfreechart-1.5.6.jar" ChebyshevEngine

