package layer

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestConv2D(t *testing.T) {
	// Test case 1: Basic Conv2D layer
	input := mat.NewDense(4, 4, []float64{
		1, 1, 1, 0,
		0, 1, 1, 1,
		0, 0, 1, 1,
		0, 0, 1, 1,
	})

	kernelWidth := 2
	kernelHeight := 2
	inputWidth := 4
	inputHeight := 4
	stride := 1
	padding := 0
	filters := 1
	activation := NewReLU()

	conv2D := NewConv2D(kernelWidth, kernelHeight, inputWidth, inputHeight, stride, padding, filters, activation)

	// Set predefined weights and biases for testing
	conv2D.Weights[0] = mat.NewDense(kernelHeight, kernelWidth, []float64{
		1, 0,
		0, 1,
	})
	conv2D.Biases.Set(0, 0, 0)

	// Expected output after the convolution operation
	expectedOutput := mat.NewDense(3, 3, []float64{
		2, 2, 2,
		0, 2, 2,
		0, 1, 2,
	})

	output, err := conv2D.Forward(input)
	if err != nil {
		t.Fatalf("failed to perform forward pass on Conv2D layer: %v", err)
	}

	if !mat.EqualApprox(output, expectedOutput, 1e-6) {
		t.Errorf("unexpected output from Conv2D layer: got %v, want %v", output, expectedOutput)
	}

	// Add more test cases as needed
}
