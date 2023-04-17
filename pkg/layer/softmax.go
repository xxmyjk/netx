package layer

import (
	"math"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

type Softmax struct{}

func NewSoftmax() *Softmax {
	return &Softmax{}
}

func (s *Softmax) Apply(matrix *mat.Dense) error {
	rows, _ := matrix.Dims()

	for i := 0; i < rows; i++ {
		row := mat.Row(nil, i, matrix)
		expRow := make([]float64, len(row))

		// Calculate the exponentials
		for j, val := range row {
			expRow[j] = math.Exp(val)
		}

		// Calculate the sum of the exponentials
		expSum := 0.0
		for _, val := range expRow {
			expSum += val
		}

		// Normalize the row by dividing each element by the sum of the exponentials
		for j, val := range expRow {
			matrix.Set(i, j, val/expSum)
		}
	}

	return nil
}

func (s *Softmax) ApplySingle(value float64) (float64, error) {
	return 0, errors.New("cannot apply softmax to a single value")
}

func (s *Softmax) ApplyDerivative(m *mat.Dense) (*mat.Dense, error) {
	// TODO: unimplemented
	return nil, nil
}
