package layer

import (
	"math"

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
			m.Set(i, j, math.Max(0, m.At(i, j)))
		}
	}
	return nil
}
