package pkg

import (
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	Forward(input *mat.Dense) (*mat.Dense, error)
}

type Network struct {
	Layers []Layer
}

func (n *Network) AddLayer(layer Layer) {
	n.Layers = append(n.Layers, layer)
}

func (n *Network) Forward(input *mat.Dense) (*mat.Dense, error) {
	current := input
	var err error
	for _, layer := range n.Layers {
		current, err = layer.Forward(current)
		if err != nil {
			return nil, err
		}
	}
	return current, nil
}
