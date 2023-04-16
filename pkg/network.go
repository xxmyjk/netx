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

func (n *Network) Forward(inputs []*mat.Dense) ([]*mat.Dense, error) {
	outputs := make([]*mat.Dense, len(inputs))
	for i, input := range inputs {
		current := input
		var err error
		for _, layer := range n.Layers {
			current, err = layer.Forward(current)
			if err != nil {
				return nil, err
			}
		}
		outputs[i] = current
	}
	return outputs, nil
}

func (n *Network) Backward(grad *mat.Dense) {
	for i := len(n.Layers) - 1; i >= 0; i-- {
		layerWithBackward, ok := n.Layers[i].(layer.LayerWithBackward)
		if ok {
			grad = layerWithBackward.Backward(grad)
		}
	}
}
