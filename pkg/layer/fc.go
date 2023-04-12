package layer

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

type FC struct {
	Weights    *mat.Dense
	Biases     *mat.Dense
	Activation Activation
}

func NewFC(inputDim, outputDim int, activation Activation) *FC {
	weights := mat.NewDense(outputDim, inputDim, nil)
	biases := mat.NewDense(outputDim, 1, nil)

	return &FC{
		Weights:    weights,
		Biases:     biases,
		Activation: activation,
	}
}
func (fc *FC) Forward(input *mat.Dense) (*mat.Dense, error) {
	_, inputCols := input.Dims()
	_, weightsCols := fc.Weights.Dims()

	if inputCols != weightsCols {
		return nil, errors.New("input and weights dimensions mismatch")
	}

	// Transpose the input matrix
	inputT := new(mat.Dense)
	inputT.CloneFrom(input.T())

	output := new(mat.Dense)
	output.Mul(fc.Weights, inputT)

	// Broadcast biases
	// output = util.BroadcastAdd(output, fc.Biases)
	output.Add(output, fc.Biases)

	err := fc.Activation.Apply(output)
	if err != nil {
		return nil, err
	}

	return output, nil
}
