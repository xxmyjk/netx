package layer

import (
	"gonum.org/v1/gonum/mat"
)

type ReLU struct{}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Apply(m *mat.Dense) error {
	rows, cols := m.Dims()

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := m.At(i, j)
			if val < 0 {
				m.Set(i, j, 0)
			}
		}
	}
	return nil
}

func (r *ReLU) ApplyDerivative(m *mat.Dense) (*mat.Dense, error) {
	rows, cols := m.Dims()
	derivative := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := m.At(i, j)
			if val > 0 {
				derivative.Set(i, j, 1)
			} else {
				derivative.Set(i, j, 0)
			}
		}
	}
	return derivative, nil
}
