package util

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestDownloadFileWithResume(t *testing.T) {
	// 创建一个临时文件
	tempFile, err := os.CreateTemp("", "test_download")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tempFile.Name())

	// 写入临时文件
	content := []byte("This is a test file for download")
	_, err = tempFile.Write(content)
	if err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tempFile.Close()

	// 创建一个简单的文件服务器
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, tempFile.Name())
	}))
	defer ts.Close()

	// 创建一个临时目录存放下载的文件
	tempDir, err := os.MkdirTemp("", "test_download_destination")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// 使用 downloadFileWithResume 函数下载文件
	destFile := filepath.Join(tempDir, "downloaded_file")
	err = downloadFileWithResume(destFile, ts.URL)
	if err != nil {
		t.Fatalf("Failed to download file with resume: %v", err)
	}

	// 检查下载的文件内容是否正确
	downloadedContent, err := os.ReadFile(destFile)
	if err != nil {
		t.Fatalf("Failed to read downloaded file: %v", err)
	}

	if string(downloadedContent) != string(content) {
		t.Errorf("Downloaded file content mismatch: expected %s, got %s", string(content), string(downloadedContent))
	}
}
