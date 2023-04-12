package layer

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestFC(t *testing.T) {
	// Test case 1: Basic FC layer
	input := mat.NewDense(1, 3, []float64{
		1,
		0,
		-1,
	})

	inputDim := 3
	outputDim := 2
	activation := NewReLU()

	fc := NewFC(inputDim, outputDim, activation)

	// Set predefined weights and biases for testing
	fc.Weights = mat.NewDense(outputDim, inputDim, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	fc.Biases = mat.NewDense(outputDim, 1, []float64{
		-1,
		1,
	})

	// Expected output after the fully connected layer and activation
	expectedOutput := mat.NewDense(2, 1, []float64{
		0,
		0,
	})

	output, err := fc.Forward(input)
	if err != nil {
		t.Fatalf("failed to perform forward pass on FC layer: %v", err)
	}

	if !mat.EqualApprox(output, expectedOutput, 1e-6) {
		t.Errorf("unexpected output from FC layer: got %v, want %v", output, expectedOutput)
	}
}
