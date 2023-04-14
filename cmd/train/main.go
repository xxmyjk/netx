package main

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/xxmyjk/netx/pkg/util"
)

func main_() {
	downloadDir := "data"
	err := util.DownloadMNIST(downloadDir)
	if err != nil {
		log.Fatalf("Error downloading MNIST dataset: %v", err)
	} else {
		fmt.Printf("MNIST dataset successfully downloaded to %s\n", downloadDir)
	}

	// Load and preprocess the MNIST dataset
	trainImagesPath := filepath.Join(downloadDir, "mnist", "train-images", "train-images-idx3-ubyte.gz")
	trainLabelsPath := filepath.Join(downloadDir, "mnist", "train-labels", "train-labels-idx1-ubyte.gz")
	testImagesPath := filepath.Join(downloadDir, "mnist", "test-images", "t10k-images-idx3-ubyte.gz")
	testLabelsPath := filepath.Join(downloadDir, "mnist", "test-labels", "t10k-labels-idx1-ubyte.gz")

	trainData, err := util.LoadMNIST(trainImagesPath, trainLabelsPath)
	if err != nil {
		log.Fatalf("Error loading training data: %v", err)
	}

	testData, err := util.LoadMNIST(testImagesPath, testLabelsPath)
	if err != nil {
		log.Fatalf("Error loading test data: %v", err)
	}

	// Continue with the training and testing of your LeNet model
	println("trainData", trainData)
	println("testData", testData)
}

func main() {
	downloadDir := "data"
	err := util.DownloadMNIST(downloadDir)
	if err != nil {
		println("Error downloading MNIST dataset:", err)
		return
	}

	trainImagesPath := filepath.Join(downloadDir, "mnist", "train-images", "train-images-idx3-ubyte.gz")
	trainLabelsPath := filepath.Join(downloadDir, "mnist", "train-labels", "train-labels-idx1-ubyte.gz")
	testImagesPath := filepath.Join(downloadDir, "mnist", "test-images", "t10k-images-idx3-ubyte.gz")
	testLabelsPath := filepath.Join(downloadDir, "mnist", "test-labels", "t10k-labels-idx1-ubyte.gz")

	trainData, err := util.LoadMNIST(trainImagesPath, trainLabelsPath)
	if err != nil {
		fmt.Println("Error loading training data:", err)
		return
	}

	testData, err := util.LoadMNIST(testImagesPath, testLabelsPath)
	if err != nil {
		fmt.Println("Error loading test data:", err)
		return
	}

	normalizedTrainImages, oneHotTrainLabels := util.PreprocessMNIST(trainData)
	normalizedTestImages, oneHotTestLabels := util.PreprocessMNIST(testData)

	// Now you can use the preprocessed data for training and testing your model.
	println("normalizedTrainImages", normalizedTrainImages)
	println("oneHotTrainLabels", oneHotTrainLabels)
	println("normalizedTestImages", normalizedTestImages)
	println("oneHotTestLabels", oneHotTestLabels)
}
