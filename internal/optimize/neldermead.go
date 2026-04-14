// Package optimize provides derivative-free optimization algorithms.
package optimize

import (
	"math"
	"sort"
)

// Options controls the Nelder-Mead optimizer behavior.
type Options struct {
	MaxIter int     // Maximum iterations (default: 1000*len(x0))
	Tol     float64 // Convergence tolerance on function value spread (default: 1e-8)
	Alpha   float64 // Reflection coefficient (default: 1.0)
	Gamma   float64 // Expansion coefficient (default: 2.0)
	Rho     float64 // Contraction coefficient (default: 0.5)
	Sigma   float64 // Shrink coefficient (default: 0.5)
}

// Result holds the optimization result.
type Result struct {
	X         []float64 // Optimal parameter vector
	Value     float64   // Objective function value at X
	Iters     int       // Number of iterations used
	Converged bool      // Whether convergence tolerance was met
}

// NelderMead minimizes f(x) using the Nelder-Mead simplex method.
// This is a derivative-free optimizer suitable for ARIMA likelihood optimization.
func NelderMead(f func([]float64) float64, x0 []float64, opts *Options) *Result {
	n := len(x0)
	if n == 0 {
		return &Result{X: x0, Value: f(x0), Converged: true}
	}

	// Apply defaults
	alpha := 1.0
	gamma := 2.0
	rho := 0.5
	sigma := 0.5
	maxIter := 1000 * n
	tol := 1e-8

	if opts != nil {
		if opts.Alpha > 0 {
			alpha = opts.Alpha
		}
		if opts.Gamma > 0 {
			gamma = opts.Gamma
		}
		if opts.Rho > 0 {
			rho = opts.Rho
		}
		if opts.Sigma > 0 {
			sigma = opts.Sigma
		}
		if opts.MaxIter > 0 {
			maxIter = opts.MaxIter
		}
		if opts.Tol > 0 {
			tol = opts.Tol
		}
	}

	// Initialize simplex: n+1 vertices
	type vertex struct {
		x []float64
		f float64
	}
	simplex := make([]vertex, n+1)

	// First vertex is x0
	simplex[0].x = make([]float64, n)
	copy(simplex[0].x, x0)
	simplex[0].f = f(x0)

	// Other vertices: perturb each dimension
	for i := 0; i < n; i++ {
		xi := make([]float64, n)
		copy(xi, x0)
		step := 0.05
		if math.Abs(xi[i]) > 1e-10 {
			step = 0.05 * math.Abs(xi[i])
		}
		xi[i] += step
		simplex[i+1] = vertex{x: xi, f: f(xi)}
	}

	sortSimplex := func() {
		sort.Slice(simplex, func(i, j int) bool {
			return simplex[i].f < simplex[j].f
		})
	}

	// Helper to compute centroid of all points except the worst
	centroid := func() []float64 {
		c := make([]float64, n)
		for i := 0; i < n; i++ { // all except last (worst)
			for j := 0; j < n; j++ {
				c[j] += simplex[i].x[j]
			}
		}
		for j := range c {
			c[j] /= float64(n)
		}
		return c
	}

	// Helper to create point: c + coeff*(c - worst)
	transform := func(c, worst []float64, coeff float64) ([]float64, float64) {
		p := make([]float64, n)
		for j := range p {
			p[j] = c[j] + coeff*(c[j]-worst[j])
		}
		return p, f(p)
	}

	iters := 0
	for iters < maxIter {
		sortSimplex()
		iters++

		// Check convergence: spread of function values
		fRange := math.Abs(simplex[n].f - simplex[0].f)
		if fRange < tol {
			return &Result{X: simplex[0].x, Value: simplex[0].f, Iters: iters, Converged: true}
		}

		c := centroid()
		worst := simplex[n]

		// Reflection
		xr, fr := transform(c, worst.x, alpha)

		if fr >= simplex[0].f && fr < simplex[n-1].f {
			// Accept reflection
			simplex[n] = vertex{x: xr, f: fr}
			continue
		}

		if fr < simplex[0].f {
			// Try expansion
			xe, fe := transform(c, worst.x, gamma)
			if fe < fr {
				simplex[n] = vertex{x: xe, f: fe}
			} else {
				simplex[n] = vertex{x: xr, f: fr}
			}
			continue
		}

		// Contraction
		xc, fc := transform(c, worst.x, -rho)
		if fc < worst.f {
			simplex[n] = vertex{x: xc, f: fc}
			continue
		}

		// Shrink: move all points toward the best
		best := simplex[0]
		for i := 1; i <= n; i++ {
			for j := 0; j < n; j++ {
				simplex[i].x[j] = best.x[j] + sigma*(simplex[i].x[j]-best.x[j])
			}
			simplex[i].f = f(simplex[i].x)
		}
	}

	sortSimplex()
	return &Result{X: simplex[0].x, Value: simplex[0].f, Iters: iters, Converged: false}
}
