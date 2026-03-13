package main

import (
	"fmt"
	"os"
	"path/filepath"

	imageprocessing "image-processing"

	"strings"

	"gocv.io/x/gocv"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <image-path> [target-dimension]\n", os.Args[0])
		os.Exit(1)
	}

	inputPath := os.Args[1]
	outputPath := "data/output-images"
	if !imageprocessing.IsSupportedFormat(inputPath) {
		fmt.Fprintf(os.Stderr, "unsupported format: %s\n", filepath.Ext(inputPath))
		os.Exit(1)
	}

	info, err := imageprocessing.GetImageInfo(inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to get image info: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("color space: %s, has alpha: %v, channels: %d\n", info.ColorSpace, info.HasAlpha, info.Channels)

	var modelPath string
	libPath := "/opt/homebrew/Cellar/onnxruntime/1.24.3_1/lib/libonnxruntime.dylib"
	modelName := "u2net"

	switch modelName {
	case "u2net":
		modelPath = "data/models/u2net.onnx"
	case "isnet":
		modelPath = "data/models/isnet-general-use.onnx"
	case "biref":
		modelPath = "data/models/BiRefNet_HR-matting-epoch_135.onnx"
	}

	img := gocv.IMRead(inputPath, gocv.IMReadUnchanged)
	if img.Empty() {
		fmt.Fprintf(os.Stderr, "failed to read image: %s\n", inputPath)
		os.Exit(1)
	}
	defer img.Close()

	fmt.Printf("loaded %s (%dx%d, %d channels)\n", inputPath, img.Cols(), img.Rows(), img.Channels())

	var noBg gocv.Mat
	if imageprocessing.HasTransparency(img) {
		fmt.Println("skipping background removal (image has transparency)")
		noBg = img.Clone()
	} else if imageprocessing.HasWhiteBackground(img) {
		fmt.Println("skipping background removal (white background detected)")
		if img.Channels() == 4 {
			noBg = gocv.NewMat()
			gocv.CvtColor(img, &noBg, gocv.ColorBGRAToBGR)
		} else {
			noBg = img.Clone()
		}
	} else {
		fmt.Println("removing background...")
		var bgErr error
		noBg, bgErr = imageprocessing.RemoveBackground(img, modelPath, libPath, modelName)
		if bgErr != nil {
			fmt.Fprintf(os.Stderr, "failed to remove background: %v\n", bgErr)
			os.Exit(1)
		}
		fmt.Printf("background removed (%dx%d, %d channels)\n", noBg.Cols(), noBg.Rows(), noBg.Channels())
	}
	defer noBg.Close()

	contours := imageprocessing.GetContours(noBg)
	if len(contours) == 0 {
		fmt.Fprintf(os.Stderr, "no contours found\n")
		os.Exit(1)
	}
	contour := contours[0]
	fmt.Printf("contour: %v\n", contour)

	composite := imageprocessing.CompositeImageOnCanvas(noBg, contour, imageprocessing.CerveBorderRatio)
	defer composite.Close()
	fmt.Printf("composited to %dx%d\n", composite.Cols(), composite.Rows())

	resized := imageprocessing.SetImageSize(composite, imageprocessing.CerveTargetDimension)
	defer resized.Close()

	fmt.Printf("resized to %dx%d\n", resized.Cols(), resized.Rows())

	//dir := filepath.Dir(inputPath)
	base := strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))
	outputPath = filepath.Join(outputPath, base+"_resized.png")

	if err := imageprocessing.SaveAsPNG(resized, outputPath); err != nil {
		fmt.Fprintf(os.Stderr, "failed to save image: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("saved to %s\n", outputPath)
}
