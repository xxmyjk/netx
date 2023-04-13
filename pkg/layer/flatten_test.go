package layer

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestFlatten(t *testing.T) {
	// Test case 1: Basic Flatten layer
	input := mat.NewDense(1, 12, []float64{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
	})

	inputWidth := 4
	inputHeight := 3
	inputDepth := 1

	flatten := NewFlatten(inputWidth, inputHeight, inputDepth)

	expectedOutput := mat.NewDense(1, 12, []float64{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
	})

	output, err := flatten.Forward(input)
	if err != nil {
		t.Fatalf("failed to perform forward pass on Flatten layer: %v", err)
	}

	if !mat.EqualApprox(output, expectedOutput, 1e-6) {
		t.Errorf("unexpected output from Flatten layer: got %v, want %v", output, expectedOutput)
	}
}
