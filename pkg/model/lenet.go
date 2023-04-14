package model

import (
	"github.com/xxmyjk/netx/pkg"
	"github.com/xxmyjk/netx/pkg/layer"
)

func NewLeNet() *pkg.Network {
	network := &pkg.Network{}

	// First convolutional layer with ReLU activation
	network.AddLayer(layer.NewConv2D(5, 5, 28, 28, 1, 0, 6, layer.NewReLU()))

	// First pooling layer
	network.AddLayer(layer.NewPooling2D(2, 2, 2, 0))

	// Second convolutional layer with ReLU activation
	network.AddLayer(layer.NewConv2D(5, 5, 14, 14, 1, 0, 16, layer.NewReLU()))

	// Second pooling layer
	network.AddLayer(layer.NewPooling2D(2, 2, 2, 0))

	// Flatten layer
	network.AddLayer(layer.NewFlatten(5, 5, 16))

	// First fully connected layer with ReLU activation
	network.AddLayer(layer.NewFC(400, 120, layer.NewReLU()))

	// Second fully connected layer with ReLU activation
	network.AddLayer(layer.NewFC(120, 84, layer.NewReLU()))

	// Third fully connected layer (output layer)
	network.AddLayer(layer.NewFC(84, 10, layer.NewSoftmax()))

	return network
}
