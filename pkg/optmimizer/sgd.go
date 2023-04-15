package optimizer

import (
	"gonum.org/v1/gonum/mat"
)

type SGD struct {
	LearningRate float64
}

func NewSGD(learningRate float64) *SGD {
	return &SGD{
		LearningRate: learningRate,
	}
}

func (o *SGD) Update(weights, gradients *mat.Dense) *mat.Dense {
	rows, cols := weights.Dims()
	updatedWeights := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			updatedWeights.Set(i, j, weights.At(i, j)-o.LearningRate*gradients.At(i, j))
		}
	}

	return updatedWeights
}
