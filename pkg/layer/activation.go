package layer

import "gonum.org/v1/gonum/mat"

type Activation interface {
	Apply(*mat.Dense) error
	ApplyDerivative(*mat.Dense) (*mat.Dense, error)
}
