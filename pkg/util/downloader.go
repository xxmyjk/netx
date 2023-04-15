package util

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
)

type progressWriter struct {
	totalSize      int64
	currentSize    int64
	lastUpdate     int64
	showPercentage bool
}

func (pw *progressWriter) Write(p []byte) (int, error) {
	n := len(p)
	pw.currentSize += int64(n)

	if pw.showPercentage {
		percentage := int64(float64(pw.currentSize) / float64(pw.totalSize) * 100)

		if percentage != pw.lastUpdate {
			pw.lastUpdate = int64(percentage)
			fmt.Printf("\rDownloading: %d%%", percentage)
		}
	} else {
		pw.lastUpdate += int64(n)
		if pw.lastUpdate >= 1024 {
			fmt.Printf("\rDownloaded: %d bytes", pw.currentSize)
			pw.lastUpdate = 0
		}
	}

	return n, nil
}

func DownloadDataset(datasetName string, urls map[string]string, downloadDir string) error {
	if downloadDir == "" {
		downloadDir = "data"
	}

	for dataType, url := range urls {
		filename := filepath.Base(url)
		filePath := filepath.Join(downloadDir, datasetName, dataType, filename)

		err := os.MkdirAll(filepath.Dir(filePath), os.ModePerm)
		if err != nil {
			return fmt.Errorf("unable to create directory for dataset: %v", err)
		}

		err = downloadFileWithResume(filePath, url)
		if err != nil {
			return fmt.Errorf("unable to download %s file: %v", dataType, err)
		}
	}

	return nil
}

func downloadFileWithResume(filePath, url string) error {
	tempFilePath := filePath + ".tmp"

	outFile, err := os.OpenFile(tempFilePath, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer outFile.Close()

	stat, err := outFile.Stat()
	if err != nil {
		return err
	}

	startByte := stat.Size()

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}

	if startByte > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startByte))
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("non-200 status code: %d", resp.StatusCode)
	}

	// 获取文件总大小
	contentLength := resp.Header.Get("Content-Length")
	var fileSize int64
	if contentLength != "" {
		fileSize, err = strconv.ParseInt(contentLength, 10, 64)
		if err != nil {
			return err
		}
		fileSize += startByte
	}

	// 创建 progressWriter 实例
	pw := &progressWriter{
		totalSize:      fileSize,
		currentSize:    startByte,
		showPercentage: contentLength != "",
	}

	// 使用 io.TeeReader 将数据从 resp.Body 写入 outFile 和 progressWriter
	_, err = io.Copy(outFile, io.TeeReader(resp.Body, pw))
	if err != nil {
		return err
	}

	_, err = io.Copy(outFile, resp.Body)
	if err != nil {
		return err
	}

	// check if outFile Close
	if err := outFile.Close(); err != nil {
		return err
	}

	err = os.Rename(tempFilePath, filePath)
	if err != nil {
		return err
	}

	return nil

}
