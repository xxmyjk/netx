package layer

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

type Flatten struct {
	InputWidth  int
	InputHeight int
	InputDepth  int
}

func NewFlatten(inputWidth, inputHeight, inputDepth int) *Flatten {
	return &Flatten{
		InputWidth:  inputWidth,
		InputHeight: inputHeight,
		InputDepth:  inputDepth,
	}
}

func (f *Flatten) Forward(input *mat.Dense) (*mat.Dense, error) {
	_, cols := input.Dims()
	expectedCols := f.InputWidth * f.InputHeight * f.InputDepth

	if cols != expectedCols {
		return nil, errors.New("input dimensions mismatch")
	}

	output := mat.DenseCopyOf(input)
	return output, nil
}
