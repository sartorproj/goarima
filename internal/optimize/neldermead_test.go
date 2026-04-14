package optimize

import (
	"math"
	"testing"
)

func TestNelderMeadQuadratic(t *testing.T) {
	// Minimize (x-3)^2 + (y-4)^2
	f := func(x []float64) float64 {
		return (x[0]-3)*(x[0]-3) + (x[1]-4)*(x[1]-4)
	}

	result := NelderMead(f, []float64{0, 0}, nil)

	if !result.Converged {
		t.Fatal("Did not converge")
	}
	if math.Abs(result.X[0]-3) > 1e-4 || math.Abs(result.X[1]-4) > 1e-4 {
		t.Errorf("Expected (3,4), got (%.6f, %.6f)", result.X[0], result.X[1])
	}
	if math.Abs(result.Value) > 1e-6 {
		t.Errorf("Expected f=0, got %e", result.Value)
	}
}

func TestNelderMeadRosenbrock(t *testing.T) {
	// Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
	// Minimum at (1, 1)
	f := func(x []float64) float64 {
		return (1-x[0])*(1-x[0]) + 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])
	}

	result := NelderMead(f, []float64{-1, -1}, &Options{MaxIter: 50000, Tol: 1e-10})

	if math.Abs(result.X[0]-1) > 1e-3 || math.Abs(result.X[1]-1) > 1e-3 {
		t.Errorf("Expected (1,1), got (%.6f, %.6f) after %d iters", result.X[0], result.X[1], result.Iters)
	}
	t.Logf("Rosenbrock: x=(%.6f, %.6f), f=%.2e, iters=%d, converged=%v",
		result.X[0], result.X[1], result.Value, result.Iters, result.Converged)
}

func TestNelderMeadSingleDim(t *testing.T) {
	// Minimize (x-5)^2
	f := func(x []float64) float64 {
		return (x[0] - 5) * (x[0] - 5)
	}

	result := NelderMead(f, []float64{0}, nil)

	if math.Abs(result.X[0]-5) > 0.1 {
		t.Errorf("Expected 5, got %.6f", result.X[0])
	}
}

func TestNelderMeadHighDim(t *testing.T) {
	// Sum of (x_i - i)^2 in 5 dimensions
	f := func(x []float64) float64 {
		sum := 0.0
		for i, v := range x {
			d := v - float64(i+1)
			sum += d * d
		}
		return sum
	}

	result := NelderMead(f, []float64{0, 0, 0, 0, 0}, nil)

	for i, v := range result.X {
		if math.Abs(v-float64(i+1)) > 1e-3 {
			t.Errorf("Dim %d: expected %d, got %.4f", i, i+1, v)
		}
	}
}
