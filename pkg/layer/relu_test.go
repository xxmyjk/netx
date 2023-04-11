package layer

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestReLU(t *testing.T) {
	// Test case 1: Basic ReLU activation
	input := mat.NewDense(3, 3, []float64{
		-1, 0, 1,
		-0.5, 2, -2,
		0.5, 1, -1,
	})

	expectedOutput := mat.NewDense(3, 3, []float64{
		0, 0, 1,
		0, 2, 0,
		0.5, 1, 0,
	})

	relu := NewReLU()

	err := relu.Apply(input)
	if err != nil {
		t.Fatalf("failed to apply ReLU activation: %v", err)
	}

	if !mat.EqualApprox(input, expectedOutput, 1e-6) {
		t.Errorf("unexpected output from ReLU activation: got %v, want %v", input, expectedOutput)
	}

	// // Test case 2: ApplySingle ReLU activation
	// testCases := []struct {
	// 	input          float64
	// 	expectedOutput float64
	// }{
	// 	{-1, 0},
	// 	{0, 0},
	// 	{1, 1},
	// 	{-0.5, 0},
	// 	{2, 2},
	// 	{-2, 0},
	// 	{0.5, 0.5},
	// }

	// for _, tc := range testCases {
	// 	output := relu.ApplySingle(tc.input)
	// 	if output != tc.expectedOutput {
	// 		t.Errorf("unexpected output from ApplySingle ReLU activation: got %v, want %v", output, tc.expectedOutput)
	// 	}
	// }
}
