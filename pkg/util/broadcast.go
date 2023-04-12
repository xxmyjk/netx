package util

import (
	"gonum.org/v1/gonum/mat"
)

func BroadcastAdd(matrix, bias *mat.Dense) *mat.Dense {
	rows, cols := matrix.Dims()
	_, biasCols := bias.Dims()

	result := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, matrix.At(i, j)+bias.At(0, j%biasCols))
		}
	}

	return result
}
