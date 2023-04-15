package optimizer

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSGD(t *testing.T) {
	sgd := NewSGD(0.01)

	weights := mat.NewDense(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})

	gradients := mat.NewDense(3, 3, []float64{
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,
	})

	updatedWeights := sgd.Update(weights, gradients)
	expectedUpdatedWeights := mat.NewDense(3, 3, []float64{
		0.99, 1.98, 2.97,
		3.96, 4.95, 5.94,
		6.93, 7.92, 8.91,
	})

	if !mat.EqualApprox(updatedWeights, expectedUpdatedWeights, 1e-2) {
		t.Errorf("Expected updated weights:\n%v\ngot:\n%v", mat.Formatted(expectedUpdatedWeights), mat.Formatted(updatedWeights))
	}
}
