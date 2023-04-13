package layer

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSoftmax(t *testing.T) {
	// Test case 1: Basic Softmax activation
	input := mat.NewDense(1, 4, []float64{
		1, 2, 3, 4,
	})

	softmax := NewSoftmax()

	expectedOutput := mat.NewDense(1, 4, []float64{
		0.0320586, 0.08714432, 0.23688282, 0.64391426,
	})

	err := softmax.Apply(input)
	if err != nil {
		t.Fatalf("failed to apply softmax activation: %v", err)
	}

	if !mat.EqualApprox(input, expectedOutput, 1e-6) {
		t.Errorf("unexpected output from Softmax activation: got %v, want %v", input, expectedOutput)
	}
}
