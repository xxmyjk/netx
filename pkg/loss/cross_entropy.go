package loss

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type CrossEntropy struct{}

func NewCrossEntropy() *CrossEntropy {
	return &CrossEntropy{}
}

func (ce *CrossEntropy) Compute(yTrue, yPred *mat.Dense) float64 {
	rows, cols := yTrue.Dims()

	var loss float64
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			loss -= yTrue.At(i, j) * math.Log(yPred.At(i, j))
		}
	}

	return loss / float64(rows)
}

func (ce *CrossEntropy) Derivative(yTrue, yPred *mat.Dense) *mat.Dense {
	rows, cols := yTrue.Dims()
	deriv := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			deriv.Set(i, j, -yTrue.At(i, j)/yPred.At(i, j))
		}
	}

	return deriv
}
