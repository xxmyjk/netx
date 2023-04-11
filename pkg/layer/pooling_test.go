package layer

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestPooling2D(t *testing.T) {
	// Test case 1: Basic Max Pooling layer
	input := mat.NewDense(4, 4, []float64{
		1, 1, 1, 0,
		0, 1, 1, 1,
		0, 0, 1, 1,
		0, 0, 1, 1,
	})

	kernelWidth := 2
	kernelHeight := 2
	stride := 2
	padding := 0

	pooling2D := NewPooling2D(kernelWidth, kernelHeight, stride, padding)

	expectedOutput := mat.NewDense(2, 2, []float64{
		1, 1,
		0, 1,
	})

	output, err := pooling2D.Forward(input)
	if err != nil {
		t.Fatalf("failed to perform forward pass on Pooling2D layer: %v", err)
	}

	if !mat.EqualApprox(output, expectedOutput, 1e-6) {
		t.Errorf("unexpected output from Pooling2D layer: got %v, want %v", output, expectedOutput)
	}
}
