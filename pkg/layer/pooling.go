package layer

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

type Pooling2D struct {
	KernelWidth  int
	KernelHeight int
	Stride       int
	Padding      int
}

func NewPooling2D(kernelWidth, kernelHeight, stride, padding int) *Pooling2D {
	return &Pooling2D{
		KernelWidth:  kernelWidth,
		KernelHeight: kernelHeight,
		Stride:       stride,
		Padding:      padding,
	}
}

func (p *Pooling2D) Forward(input *mat.Dense) (*mat.Dense, error) {
	inputRows, inputCols := input.Dims()

	if inputRows < p.KernelHeight || inputCols < p.KernelWidth {
		return nil, errors.New("input dimensions smaller than kernel dimensions")
	}

	outputRows := int(math.Ceil(float64(inputRows-p.KernelHeight+1) / float64(p.Stride)))
	outputCols := int(math.Ceil(float64(inputCols-p.KernelWidth+1) / float64(p.Stride)))

	output := mat.NewDense(outputRows, outputCols, nil)

	for i := 0; i < outputRows; i++ {
		for j := 0; j < outputCols; j++ {
			maxVal := input.At(i*p.Stride, j*p.Stride)
			for x := 0; x < p.KernelHeight; x++ {
				for y := 0; y < p.KernelWidth; y++ {
					row := i*p.Stride + x
					col := j*p.Stride + y
					if row < inputRows && col < inputCols {
						val := input.At(row, col)
						maxVal = math.Max(maxVal, val)
					}
				}
			}
			output.Set(i, j, maxVal)
		}
	}

	return output, nil
}
