package util

import (
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"gonum.org/v1/gonum/mat"
)

const (
	trainImagesURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
	trainLabelsURL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
	testImagesURL  = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
	testLabelsURL  = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
)

// MNISTData holds the image and label data for the MNIST dataset.
type MNISTData struct {
	Images [][][]byte
	Labels []byte
}

// DownloadMNIST downloads the MNIST dataset from the official website.
func DownloadMNIST(downloadDir string) error {
	mnistDir := filepath.Join(downloadDir, "mnist")

	trainImagesPath := filepath.Join(mnistDir, "train-images")
	trainLabelsPath := filepath.Join(mnistDir, "train-labels")
	testImagesPath := filepath.Join(mnistDir, "test-images")
	testLabelsPath := filepath.Join(mnistDir, "test-labels")

	err := os.MkdirAll(trainImagesPath, os.ModePerm)
	if err != nil {
		return err
	}

	err = os.MkdirAll(trainLabelsPath, os.ModePerm)
	if err != nil {
		return err
	}

	err = os.MkdirAll(testImagesPath, os.ModePerm)
	if err != nil {
		return err
	}

	err = os.MkdirAll(testLabelsPath, os.ModePerm)
	if err != nil {
		return err
	}

	urls := map[string]string{
		filepath.Join(trainImagesPath, "train-images-idx3-ubyte.gz"): trainImagesURL,
		filepath.Join(trainLabelsPath, "train-labels-idx1-ubyte.gz"): trainLabelsURL,
		filepath.Join(testImagesPath, "t10k-images-idx3-ubyte.gz"):   testImagesURL,
		filepath.Join(testLabelsPath, "t10k-labels-idx1-ubyte.gz"):   testLabelsURL,
	}

	for path, url := range urls {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			err = downloadFile(path, url)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// downloadFile downloads a file from the specified URL and saves it to the given path.
func downloadFile(filepath, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

// LoadMNIST reads the MNIST data from the specified paths and returns a MNISTData struct.
func LoadMNIST(imagesPath, labelsPath string) (*MNISTData, error) {
	imagesFile, err := os.Open(imagesPath)
	if err != nil {
		return nil, err
	}
	defer imagesFile.Close()

	labelsFile, err := os.Open(labelsPath)
	if err != nil {
		return nil, err
	}
	defer labelsFile.Close()

	var imagesMagic, labelsMagic, nImages, nLabels, rows, cols uint32
	err = binary.Read(imagesFile, binary.BigEndian, &imagesMagic)
	if err != nil {
		return nil, err
	}

	err = binary.Read(labelsFile, binary.BigEndian, &labelsMagic)
	if err != nil {
		return nil, err
	}

	err = binary.Read(imagesFile, binary.BigEndian, &nImages)
	if err != nil {
		return nil, err
	}

	err = binary.Read(labelsFile, binary.BigEndian, &nLabels)
	if err != nil {
		return nil, err
	}

	err = binary.Read(imagesFile, binary.BigEndian, &rows)
	if err != nil {
		return nil, err
	}

	err = binary.Read(imagesFile, binary.BigEndian, &cols)
	if err != nil {
		return nil, err
	}

	if nImages != nLabels {
		return nil, fmt.Errorf("number of images (%d) and labels (%d) do not match", nImages, nLabels)
	}

	imagesData := make([][][]byte, nImages)
	for i := uint32(0); i < nImages; i++ {
		imagesData[i] = make([][]byte, rows)
		for j := uint32(0); j < rows; j++ {
			imagesData[i][j] = make([]byte, cols)
			_, err = imagesFile.Read(imagesData[i][j])
			if err != nil {
				return nil, err
			}
		}
	}

	labelsData := make([]byte, nLabels)
	_, err = labelsFile.Read(labelsData)
	if err != nil {
		return nil, err
	}

	return &MNISTData{
		Images: imagesData,
		Labels: labelsData,
	}, nil
}

// PreprocessMNIST normalizes the MNIST images and one-hot encodes the labels.
func PreprocessMNIST(data *MNISTData) ([][][]float64, [][]float64) {
	nImages := len(data.Images)
	rows := len(data.Images[0])
	cols := len(data.Images[0][0])
	normalizedImages := make([][][]float64, nImages)
	for i := 0; i < nImages; i++ {
		normalizedImages[i] = make([][]float64, rows)
		for j := 0; j < rows; j++ {
			normalizedImages[i][j] = make([]float64, cols)
			for k := 0; k < cols; k++ {
				normalizedImages[i][j][k] = float64(data.Images[i][j][k]) / 255.0
			}
		}
	}

	oneHotLabels := make([][]float64, nImages)
	for i := 0; i < nImages; i++ {
		oneHotLabels[i] = make([]float64, 10)
		oneHotLabels[i][data.Labels[i]] = 1.0
	}

	return normalizedImages, oneHotLabels
}

func MNISTDataToMatDense(images [][][]float64) []*mat.Dense {
	numImages := len(images)
	rows := len(images[0])
	cols := len(images[0][0])
	matrices := make([]*mat.Dense, numImages)

	for i := 0; i < numImages; i++ {
		data := make([]float64, rows*cols)
		for j := 0; j < rows; j++ {
			for k := 0; k < cols; k++ {
				data[j*cols+k] = images[i][j][k]
			}
		}
		matrices[i] = mat.NewDense(rows, cols, data)
	}

	return matrices
}
