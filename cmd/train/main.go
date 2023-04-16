package main

import (
	"fmt"

	"github.com/xxmyjk/netx/pkg/layer"
	"github.com/xxmyjk/netx/pkg/loss"
	"github.com/xxmyjk/netx/pkg/model"
	"github.com/xxmyjk/netx/pkg/optimizer"
	"github.com/xxmyjk/netx/pkg/util"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// 1. Download MNIST dataset
	err := util.DownloadMNIST("./data")
	if err != nil {
		fmt.Println("Error downloading MNIST dataset:", err)
		return
	}

	// 2. Load MNIST dataset
	trainData, err := util.LoadMNIST(
		"./data/mnist/train-images/train-images-idx3-ubyte",
		"./data/mnist/train-labels/train-labels-idx1-ubyte",
	)
	if err != nil {
		fmt.Println("Error loading MNIST training dataset:", err)
		return
	}
	testData, err := util.LoadMNIST("./data/mnist/test-images/t10k-images-idx3-ubyte",
		"./data/mnist/test-labels/t10k-labels-idx1-ubyte",
	)
	if err != nil {
		fmt.Println("Error loading MNIST test dataset:", err)
		return
	}

	// 3. Preprocess data
	// Preprocess the data
	normalizedTrainImages, oneHotTrainLabels := util.PreprocessMNIST(trainData)
	normalizedTestImages, oneHotTestLabels := util.PreprocessMNIST(testData)

	// Convert the preprocessed images to *mat.Dense
	trainImages := util.MNISTDataToMatDense(normalizedTrainImages)
	testImages := util.MNISTDataToMatDense(normalizedTestImages)

	// 4. Create LeNet model
	net := model.NewLeNet()

	// 5. Define loss function and optimizer
	lossFunc := loss.NewCrossEntropy()
	opt := optimizer.NewSGD(net, 0.01)

	// 6. Train model
	epochs := 10
	batchSize := 64
	loss := 0.0
	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < len(trainImages); i += batchSize {
			// 6.1 Get batch data
			batchData := trainImages[i:min(i+batchSize, len(trainImages))]
			batchLabels := oneHotTrainLabels[i:min(i+batchSize, len(oneHotTrainLabels))]

			// 6.2 Forward pass
			output, err := net.Forward(batchData)
			if err != nil {
				fmt.Printf("Error performing forward pass: %v\n", err)
				return
			}

			// 6.3.0 convert data
			// Convert []*mat.Dense to *mat.Dense for output
			outputRows, outputCols := output[0].Dims()
			concatOutput := mat.NewDense(len(output)*outputRows, outputCols, nil)
			for i, o := range output {
				for r := 0; r < outputRows; r++ {
					concatOutput.SetRow(i*outputRows+r, o.RawRowView(r))
				}
			}

			// Convert [][]float64 to *mat.Dense for batchLabels
			batchLabelsData := make([]float64, 0)
			for _, bl := range batchLabels {
				batchLabelsData = append(batchLabelsData, bl...)
			}
			batchLabelsDense := mat.NewDense(len(batchLabels), len(batchLabels[0]), batchLabelsData)

			// 6.3 Calculate loss
			loss = lossFunc.Compute(concatOutput, batchLabelsDense)

			// 6.4 Backward pass
			grad := lossFunc.Derivative(concatOutput, batchLabelsDense)
			net.Backward(grad)

			// 6.5 Update weights
			opt.Step()
		}

		// 7. Evaluate model on test set
		testOutput := net.Forward(testImages)
		predictions := layer.GetPredictions(testOutput)
		correct := 0
		for i := range predictions {
			if predictions[i] == testLabels[i] {
				correct++
			}
		}
		accuracy := float64(correct) / float64(len(oneHotTestLabels))

		// 8. Output loss and accuracy
		fmt.Printf("Epoch %d: Loss = %.4f, Accuracy = %.2f%%\n", epoch+1, loss, accuracy*100)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
