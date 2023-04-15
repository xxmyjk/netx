package loss

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCrossEntropy(t *testing.T) {
	ce := NewCrossEntropy()

	yTrue := mat.NewDense(3, 3, []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})

	yPred := mat.NewDense(3, 3, []float64{
		0.9, 0.05, 0.05,
		0.05, 0.9, 0.05,
		0.05, 0.05, 0.9,
	})

	loss := ce.Compute(yTrue, yPred)
	expectedLoss := 0.10536051565782628

	if loss != expectedLoss {
		t.Errorf("Expected loss: %v, got: %v", expectedLoss, loss)
	}

	deriv := ce.Derivative(yTrue, yPred)
	expectedDeriv := mat.NewDense(3, 3, []float64{
		-1.1111111111111112, 0, 0,
		0, -1.1111111111111112, 0,
		0, 0, -1.1111111111111112,
	})

	if !mat.EqualApprox(deriv, expectedDeriv, 1e-8) {
		t.Errorf("Expected derivative:\n%v\ngot:\n%v", mat.Formatted(expectedDeriv), mat.Formatted(deriv))
	}
}
