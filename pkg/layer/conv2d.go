package layer

import (
	"math/rand"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

type Conv2D struct {
	KernelWidth  int
	KernelHeight int
	InputWidth   int
	InputHeight  int
	Stride       int
	Padding      int
	Filters      int
	Weights      []*mat.Dense
	Biases       *mat.Dense
	dWeights     []*mat.Dense
	dBiases      *mat.Dense
	Activation   Activation
}

func NewConv2D(kernelWidth, kernelHeight, inputWidth, inputHeight, stride, padding, filters int, activation Activation) *Conv2D {
	if kernelWidth <= 0 || kernelHeight <= 0 || inputWidth <= 0 || inputHeight <= 0 || stride <= 0 || padding < 0 || filters <= 0 {
		panic("invalid parameters for Conv2D layer")
	}

	weights := make([]*mat.Dense, filters)
	for i := range weights {
		weights[i] = mat.NewDense(kernelHeight, kernelWidth, randomWeights(kernelWidth*kernelHeight))
	}

	biases := mat.NewDense(filters, 1, randomWeights(filters))

	dWeights := make([]*mat.Dense, filters)
	for i := range dWeights {
		dWeights[i] = mat.NewDense(kernelHeight, kernelWidth, nil)
	}
	dBiases := mat.NewDense(filters, 1, nil)

	return &Conv2D{
		KernelWidth:  kernelWidth,
		KernelHeight: kernelHeight,
		InputWidth:   inputWidth,
		InputHeight:  inputHeight,
		Stride:       stride,
		Padding:      padding,
		Filters:      filters,
		Weights:      weights,
		Biases:       biases,
		dWeights:     dWeights,
		dBiases:      dBiases,
		Activation:   activation,
	}
}

func (c *Conv2D) Forward(input *mat.Dense) (*mat.Dense, error) {
	outputWidth := (c.InputWidth-c.KernelWidth+2*c.Padding)/c.Stride + 1
	outputHeight := (c.InputHeight-c.KernelHeight+2*c.Padding)/c.Stride + 1

	output := mat.NewDense(outputHeight, outputWidth, nil)

	for f := 0; f < c.Filters; f++ {
		for y := 0; y < outputHeight; y++ {
			for x := 0; x < outputWidth; x++ {
				sum := 0.0
				for ky := 0; ky < c.KernelHeight; ky++ {
					for kx := 0; kx < c.KernelWidth; kx++ {
						iy := y*c.Stride - c.Padding + ky
						ix := x*c.Stride - c.Padding + kx

						if iy >= 0 && iy < c.InputHeight && ix >= 0 && ix < c.InputWidth {
							sum += input.At(iy, ix) * c.Weights[f].At(ky, kx)
						}
					}
				}

				sum += c.Biases.At(f, 0)
				output.Set(y, x, sum) // Change the indices here
			}
		}
	}

	// Apply activation function
	if err := c.Activation.Apply(output); err != nil {
		return nil, errors.Wrap(err, "failed to apply activation function")
	}

	return output, nil
}

func (c *Conv2D) Backward(input, gradOutput *mat.Dense) (*mat.Dense, error) {
	outputWidth := (c.InputWidth-c.KernelWidth+2*c.Padding)/c.Stride + 1
	outputHeight := (c.InputHeight-c.KernelHeight+2*c.Padding)/c.Stride + 1

	// Calculate dWeights and dBiases
	for f := 0; f < c.Filters; f++ {
		for y := 0; y < outputHeight; y++ {
			for x := 0; x < outputWidth; x++ {
				for ky := 0; ky < c.KernelHeight; ky++ {
					for kx := 0; kx < c.KernelWidth; kx++ {
						iy := y*c.Stride - c.Padding + ky
						ix := x*c.Stride - c.Padding + kx

						if iy >= 0 && iy < c.InputHeight && ix >= 0 && ix < c.InputWidth {
							inputValue := input.At(iy, ix)
							gradOutputValue := gradOutput.At(y, x)
							weightGrad := c.dWeights[f].At(ky, kx) + inputValue*gradOutputValue
							c.dWeights[f].Set(ky, kx, weightGrad)
						}
					}
				}
				biasGrad := c.dBiases.At(f, 0) + gradOutput.At(y, x)
				c.dBiases.Set(f, 0, biasGrad)
			}
		}
	}

	// Calculate input gradient
	gradInput := mat.NewDense(c.InputHeight, c.InputWidth, nil)
	for y := 0; y < c.InputHeight; y++ {
		for x := 0; x < c.InputWidth; x++ {
			for f := 0; f < c.Filters; f++ {
				for ky := 0; ky < c.KernelHeight; ky++ {
					for kx := 0; kx < c.KernelWidth; kx++ {
						oy := (y + c.Padding - ky) / c.Stride
						ox := (x + c.Padding - kx) / c.Stride

						if oy >= 0 && oy < outputHeight && ox >= 0 && ox < outputWidth {
							inputValue := gradOutput.At(oy, ox)
							weightValue := c.Weights[f].At(ky, kx)
							gradInputValue := gradInput.At(y, x) + inputValue*weightValue
							gradInput.Set(y, x, gradInputValue)
						}
					}
				}
			}
		}
	}

	// Apply the activation function's derivative
	if _, err := c.Activation.ApplyDerivative(gradInput); err != nil {
		return nil, errors.Wrap(err, "failed to apply activation function's derivative")
	}

	return gradInput, nil
}

func randomWeights(size int) []float64 {
	weights := make([]float64, size)
	for i := range weights {
		weights[i] = rand.Float64()
	}
	return weights
}
