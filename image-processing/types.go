package imageprocessing

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

const (
	ThresholdIntensity   = 200.0
	ISNetTargetDimension = 1024                      // ISNet requires 1024x1024 pixel images
	U2NetTargetDimension = 320                       // U2Net requires 320x320 pixel images
	BirefTargetDimension = 2048                      // Biref requires 2048x2048 pixel images
	CerveTargetDimension = 1200                      // cerve requires 1200x1200 pixel images
	CerveTargetSize      = 32 * 1024 * 1024          // 32 MB as required by Cerve
	CerveBorderRatio     = float64(40) / float64(28) // The image should have a ratio from the border to the object equal to 40/28 derrived from original image of dimension 400x400 with a padding of 40
)

var supportedFormats = map[string]bool{
	".jpg":  true,
	".jpeg": true,
	".png":  true,
	".webp": true,
	".avif": true,
	".tiff": true,
	".tif":  true,
	// ".psd":  true,
}

var supportedModels = map[string]bool{
	"isnet": true, // needs more post processing tuning before using
	"u2net": true, // works good enough
	"biref": true, // very large- uses approx 30gb of data and takes a long time needs more tuning
}

type ModelInfo struct {
	Name           string
	InputDimension int
	ChannelCount   int
	OutputChannels int
}

var ISNetInfo = ModelInfo{
	Name:           "isnet",
	InputDimension: ISNetTargetDimension,
	ChannelCount:   3, // RGB
	OutputChannels: 1,
}

var U2NetInfo = ModelInfo{
	Name:           "u2net",
	InputDimension: U2NetTargetDimension,
	ChannelCount:   3, // RGB
	OutputChannels: 1,
}

var BirefInfo = ModelInfo{
	Name:           "biref",
	InputDimension: BirefTargetDimension,
	ChannelCount:   3, // RGB
	OutputChannels: 1,
}

type ImageInfo struct {
	ColorSpace string
	HasAlpha   bool
	Channels   int
}

func GetImageInfo(filePath string) (ImageInfo, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return ImageInfo{}, err
	}
	defer f.Close()

	ext := strings.ToLower(filepath.Ext(filePath))
	switch ext {
	case ".psd":
		return parsePSDInfo(f)
	case ".tiff", ".tif":
		return parseTIFFInfo(f)
	case ".jpg", ".jpeg":
		return parseJPEGInfo(f)
	case ".png":
		return parsePNGInfo(f)
	default:
		// WebP, AVIF — these don't support CMYK
		return ImageInfo{ColorSpace: "RGB", HasAlpha: false, Channels: 3}, nil
	}
}

// parsePSDInfo reads the PSD header to extract color mode and channel count.
// PSD header: bytes 0-3 "8BPS", 4-5 version, 6-11 reserved, 12-13 channels, 14-17 height, 18-21 width, 22-23 depth, 24-25 color mode
func parsePSDInfo(f *os.File) (ImageInfo, error) {
	header := make([]byte, 26)
	if _, err := f.ReadAt(header, 0); err != nil {
		return ImageInfo{}, fmt.Errorf("failed to read PSD header: %w", err)
	}
	if string(header[0:4]) != "8BPS" {
		return ImageInfo{}, fmt.Errorf("not a valid PSD file")
	}

	channels := int(binary.BigEndian.Uint16(header[12:14]))
	colorMode := binary.BigEndian.Uint16(header[24:26])

	// PSD color modes: 0=Bitmap, 1=Grayscale, 3=RGB, 4=CMYK
	cs := "Unknown"
	baseChannels := 0
	switch colorMode {
	case 0:
		cs = "Bitmap"
		baseChannels = 1
	case 1:
		cs = "Grayscale"
		baseChannels = 1
	case 3:
		cs = "RGB"
		baseChannels = 3
	case 4:
		cs = "CMYK"
		baseChannels = 4
	}

	hasAlpha := channels > baseChannels
	return ImageInfo{ColorSpace: cs, HasAlpha: hasAlpha, Channels: channels}, nil
}

// parseTIFFInfo reads TIFF IFD tags to detect color space and alpha.
// PhotometricInterpretation (tag 262): 1=Grayscale, 2=RGB, 5=CMYK
// ExtraSamples (tag 338): presence indicates alpha
func parseTIFFInfo(f *os.File) (ImageInfo, error) {
	header := make([]byte, 8)
	if _, err := f.ReadAt(header, 0); err != nil {
		return ImageInfo{}, fmt.Errorf("failed to read TIFF header: %w", err)
	}

	var bo binary.ByteOrder
	switch string(header[0:2]) {
	case "II":
		bo = binary.LittleEndian
	case "MM":
		bo = binary.BigEndian
	default:
		return ImageInfo{}, fmt.Errorf("not a valid TIFF file")
	}

	ifdOffset := bo.Uint32(header[4:8])

	countBuf := make([]byte, 2)
	if _, err := f.ReadAt(countBuf, int64(ifdOffset)); err != nil {
		return ImageInfo{}, err
	}
	numEntries := bo.Uint16(countBuf)

	cs := "RGB"
	hasAlpha := false
	channels := 3

	for i := 0; i < int(numEntries); i++ {
		entry := make([]byte, 12)
		offset := int64(ifdOffset) + 2 + int64(i)*12
		if _, err := f.ReadAt(entry, offset); err != nil {
			break
		}
		tag := bo.Uint16(entry[0:2])
		value := bo.Uint32(entry[8:12])

		switch tag {
		case 262: // PhotometricInterpretation
			switch value {
			case 1:
				cs = "Grayscale"
				channels = 1
			case 2:
				cs = "RGB"
				channels = 3
			case 5:
				cs = "CMYK"
				channels = 4
			}
		case 338: // ExtraSamples
			hasAlpha = true
			channels++
		}
	}

	return ImageInfo{ColorSpace: cs, HasAlpha: hasAlpha, Channels: channels}, nil
}

// parseJPEGInfo scans for a SOF marker to read the number of color components.
// 1=Grayscale, 3=RGB/YCbCr, 4=CMYK. JPEG never has alpha.
func parseJPEGInfo(f *os.File) (ImageInfo, error) {
	f.Seek(0, 0)
	buf := make([]byte, 2)
	for {
		if _, err := f.Read(buf); err != nil {
			return ImageInfo{}, fmt.Errorf("failed to parse JPEG: %w", err)
		}
		if buf[0] != 0xFF {
			continue
		}
		marker := buf[1]
		// SOF markers: 0xC0-0xC3, 0xC5-0xC7, 0xC9-0xCB, 0xCD-0xCF
		isSOF := (marker >= 0xC0 && marker <= 0xCF) && marker != 0xC4 && marker != 0xC8 && marker != 0xCC
		if isSOF {
			// SOF segment: length(2) + precision(1) + height(2) + width(2) + numComponents(1)
			header := make([]byte, 8)
			if _, err := f.Read(header); err != nil {
				return ImageInfo{}, fmt.Errorf("failed to read SOF: %w", err)
			}
			numComponents := int(header[7])
			cs := "RGB"
			if numComponents == 1 {
				cs = "Grayscale"
			} else if numComponents == 4 {
				cs = "CMYK"
			}
			return ImageInfo{ColorSpace: cs, HasAlpha: false, Channels: numComponents}, nil
		}
		// Skip non-SOF segment
		if marker != 0x00 && marker != 0xFF && marker != 0xD0 && marker != 0xD8 {
			lenBuf := make([]byte, 2)
			if _, err := f.Read(lenBuf); err != nil {
				return ImageInfo{}, fmt.Errorf("failed to read segment length: %w", err)
			}
			segLen := int(binary.BigEndian.Uint16(lenBuf)) - 2
			f.Seek(int64(segLen), 1)
		}
	}
}

// parsePNGInfo reads the IHDR chunk to determine color type.
// Color type bits: 1=palette, 2=color, 4=alpha
func parsePNGInfo(f *os.File) (ImageInfo, error) {
	// PNG header is 8 bytes, then IHDR chunk: 4 len + 4 "IHDR" + 13 data
	header := make([]byte, 29)
	if _, err := f.ReadAt(header, 0); err != nil {
		return ImageInfo{}, fmt.Errorf("failed to read PNG header: %w", err)
	}
	colorType := header[25]
	hasAlpha := colorType&4 != 0
	cs := "RGB"
	channels := 3
	switch colorType {
	case 0: // Grayscale
		cs = "Grayscale"
		channels = 1
	case 4: // Grayscale + Alpha
		cs = "Grayscale"
		channels = 2
	case 2: // RGB
		channels = 3
	case 6: // RGBA
		channels = 4
	case 3: // Palette
		cs = "Palette"
		channels = 1
	}
	return ImageInfo{ColorSpace: cs, HasAlpha: hasAlpha, Channels: channels}, nil
}

func GetImageInfoImageMagick(filePath string) (ImageInfo, error) {
	out, err := exec.Command("magick", "identify", "-format", "%[colorspace] %[channels]", filePath).Output()
	if err != nil {
		return ImageInfo{}, fmt.Errorf("magick identify failed: %w", err)
	}

	parts := strings.Fields(strings.TrimSpace(string(out)))
	if len(parts) < 2 {
		return ImageInfo{}, fmt.Errorf("unexpected magick output: %s", out)
	}

	colorSpace := parts[0]
	channelStr := parts[1]
	hasAlpha := strings.HasSuffix(channelStr, "a")
	channels := len(channelStr)

	return ImageInfo{
		ColorSpace: colorSpace,
		HasAlpha:   hasAlpha,
		Channels:   channels,
	}, nil
}

func IsSupportedFormat(filePath string) bool {
	suffix := strings.ToLower(filepath.Ext(filePath))
	return supportedFormats[suffix]
}

func HasBlackBackground(img gocv.Mat) bool {
	return false
}

func HasWhiteBackground(img gocv.Mat) bool {
	// Convert to grayscale
	grey := gocv.NewMat()
	defer grey.Close()
	if img.Channels() == 4 {
		gocv.CvtColor(img, &grey, gocv.ColorBGRAToGray)
	} else {
		gocv.CvtColor(img, &grey, gocv.ColorBGRToGray)
	}

	// Create mask by thresholding to find the object (non-white areas)
	threshed := gocv.NewMat()
	defer threshed.Close()
	gocv.Threshold(grey, &threshed, ThresholdIntensity, 255, gocv.ThresholdBinaryInv)

	// Find contours to get the bounding rectangle of the object
	contours := gocv.FindContours(threshed, gocv.RetrievalExternal, gocv.ChainApproxNone)
	defer contours.Close()
	if contours.Size() == 0 {
		return true // entirely white
	}

	largest := contours.At(0)
	for i := 1; i < contours.Size(); i++ {
		if gocv.ContourArea(contours.At(i)) > gocv.ContourArea(largest) {
			largest = contours.At(i)
		}
	}
	rect := gocv.BoundingRect(largest)

	// If the bounding box covers the entire image, background is not white
	if rect.Dy() == grey.Rows() && rect.Dx() == grey.Cols() {
		return false
	}

	// Create a mask that excludes the object area, keeping only the background
	mask := gocv.NewMatWithSize(grey.Rows(), grey.Cols(), gocv.MatTypeCV8U)
	defer mask.Close()
	gocv.Rectangle(&mask, rect, color.RGBA{255, 255, 255, 255}, -1)
	gocv.BitwiseNot(mask, &mask)

	// Get average intensity of the background pixels
	bgPixels := gocv.NewMat()
	defer bgPixels.Close()
	grey.CopyToWithMask(&bgPixels, mask)

	// Calculate mean of non-zero pixels (the background)
	mean := bgPixels.Mean()
	avgIntensity := mean.Val1

	return avgIntensity > ThresholdIntensity
}

func HasAlphaChanel(filePath string) bool {
	image := gocv.IMRead(filePath, gocv.IMReadUnchanged)
	defer image.Close()
	channels := image.Channels()
	return channels == 4
}

// HasTransparency checks if a 4-channel image actually contains transparent pixels.
func HasTransparency(img gocv.Mat) bool {
	if img.Channels() != 4 {
		return false
	}
	channels := gocv.Split(img)
	defer func() {
		for _, ch := range channels {
			ch.Close()
		}
	}()
	alpha := channels[3]
	minVal, _, _, _ := gocv.MinMaxLoc(alpha)
	return minVal < 255
}

func GetImageSize(img gocv.Mat) float64 {
	return float64(img.Total()*img.ElemSize()) / (1024 * 1024)
}

// WhiteToTransparent converts white/near-white pixels to transparent.
// Returns a BGRA image where the background is transparent and the object is opaque.
func WhiteToTransparent(img gocv.Mat) gocv.Mat {
	// Convert to BGRA
	bgra := AddAlphaChannel(img)

	// Convert to grayscale
	grey := gocv.NewMat()
	defer grey.Close()
	if img.Channels() == 4 {
		gocv.CvtColor(img, &grey, gocv.ColorBGRAToGray)
	} else {
		gocv.CvtColor(img, &grey, gocv.ColorBGRToGray)
	}

	// Threshold: white pixels = 255 (background), non-white = 0 (object)
	bgMask := gocv.NewMat()
	defer bgMask.Close()
	gocv.Threshold(grey, &bgMask, ThresholdIntensity, 255, gocv.ThresholdBinary)

	// BFS flood fill from all white edge pixels to mark connected background.
	// Interior white pixels (inside the object) stay unmarked.
	w, h := img.Cols(), img.Rows()
	bgBytes, _ := bgMask.DataPtrUint8()
	visited := make([]bool, w*h)
	queue := make([]int, 0, w*h/4)

	// Seed from all edges
	for x := 0; x < w; x++ {
		if bgBytes[x] == 255 {
			queue = append(queue, x)
			visited[x] = true
		}
		idx := (h-1)*w + x
		if bgBytes[idx] == 255 {
			queue = append(queue, idx)
			visited[idx] = true
		}
	}
	for y := 1; y < h-1; y++ {
		idx := y * w
		if bgBytes[idx] == 255 {
			queue = append(queue, idx)
			visited[idx] = true
		}
		idx = y*w + w - 1
		if bgBytes[idx] == 255 {
			queue = append(queue, idx)
			visited[idx] = true
		}
	}

	// BFS
	for len(queue) > 0 {
		idx := queue[0]
		queue = queue[1:]
		x, y := idx%w, idx/w
		neighbors := [4]int{idx - w, idx + w, idx - 1, idx + 1}
		valid := [4]bool{y > 0, y < h-1, x > 0, x < w-1}
		for i, n := range neighbors {
			if valid[i] && !visited[n] && bgBytes[n] == 255 {
				visited[n] = true
				queue = append(queue, n)
			}
		}
	}

	// Build alpha: visited = background = transparent (0), rest = opaque (255)
	alphaMask := gocv.NewMatWithSizeFromScalar(gocv.NewScalar(255, 0, 0, 0), img.Rows(), img.Cols(), gocv.MatTypeCV8U)
	alphaBytes, _ := alphaMask.DataPtrUint8()
	for i, v := range visited {
		if v {
			alphaBytes[i] = 0
		}
	}

	// Apply as alpha channel
	channels := gocv.Split(bgra)
	alphaMask.CopyTo(&channels[3])
	result := gocv.NewMat()
	gocv.Merge(channels, &result)
	for _, ch := range channels {
		ch.Close()
	}
	alphaMask.Close()
	bgra.Close()

	return result
}

func AddAlphaChannel(img gocv.Mat) gocv.Mat {
	if img.Channels() == 4 {
		return img.Clone()
	} else if img.Channels() == 3 {
		alpha := gocv.NewMatWithSize(img.Rows(), img.Cols(), gocv.MatTypeCV8UC1)
		alpha.SetTo(gocv.NewScalar(255, 0, 0, 0))
		channels := gocv.Split(img)
		channels = append(channels, alpha)
		imgWithAlpha := gocv.NewMat()
		gocv.Merge(channels, &imgWithAlpha)
		for _, ch := range channels {
			ch.Close()
		}
		return imgWithAlpha
	}
	panic("unsupported number of channels")
}

func GetContours(img gocv.Mat) []image.Rectangle {
	// Use the alpha channel to find the object, not the color data
	if img.Channels() == 4 {
		channels := gocv.Split(img)
		defer func() {
			for _, ch := range channels {
				ch.Close()
			}
		}()
		alpha := channels[3]

		// Threshold alpha: any pixel with alpha > 0 is part of the object
		binaryMask := gocv.NewMat()
		defer binaryMask.Close()
		gocv.Threshold(alpha, &binaryMask, 1, 255, gocv.ThresholdBinary)

		contours := gocv.FindContours(binaryMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
		defer contours.Close()
		allPoints := gocv.NewPointVector()
		defer allPoints.Close()
		for i := 0; i < contours.Size(); i++ {
			contour := contours.At(i)
			for j := 0; j < contour.Size(); j++ {
				allPoints.Append(contour.At(j))
			}
		}
		boundingRectangle := gocv.BoundingRect(allPoints)
		return []image.Rectangle{boundingRectangle}
	}

	// Fallback for 3-channel images: use grayscale thresholding
	greyImage := gocv.NewMat()
	defer greyImage.Close()
	gocv.CvtColor(img, &greyImage, gocv.ColorBGRToGray)
	blurredImage := gocv.NewMat()
	defer blurredImage.Close()
	gocv.GaussianBlur(greyImage, &blurredImage, image.Point{X: 15, Y: 15}, 50, 100, gocv.BorderDefault)
	binaryMask := gocv.NewMat()
	defer binaryMask.Close()
	gocv.Threshold(blurredImage, &binaryMask, 250, 255, gocv.ThresholdBinaryInv)
	contours := gocv.FindContours(binaryMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()
	allPoints := gocv.NewPointVector()
	defer allPoints.Close()
	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		for j := 0; j < contour.Size(); j++ {
			allPoints.Append(contour.At(j))
		}
	}
	boundingRectangle := gocv.BoundingRect(allPoints)
	return []image.Rectangle{boundingRectangle}
}

func CompositeImageOnCanvas(img gocv.Mat, contour image.Rectangle, borderRatio float64) gocv.Mat {
	width := contour.Dx()
	height := contour.Dy()
	longestSideLength := math.Max(float64(width), float64(height))
	newSideLength := int(math.Round(longestSideLength * borderRatio))

	// Create white canvas (3-channel, CV8U)
	canvas := gocv.NewMatWithSizeFromScalar(gocv.NewScalar(255, 255, 255, 0), newSideLength, newSideLength, gocv.MatTypeCV8UC3)

	// Crop the object from the image
	crop := img.Region(image.Rect(contour.Min.X, contour.Min.Y, contour.Max.X, contour.Max.Y))

	// Center the crop on the canvas
	offsetX := (newSideLength - width) / 2
	offsetY := (newSideLength - height) / 2
	roi := canvas.Region(image.Rect(offsetX, offsetY, offsetX+width, offsetY+height))

	if img.Channels() == 4 {
		// Alpha-blend the crop onto the white canvas
		channels := gocv.Split(crop)
		defer func() {
			for _, ch := range channels {
				ch.Close()
			}
		}()

		// Convert alpha to float32 normalized to [0, 1]
		alphaF := gocv.NewMat()
		defer alphaF.Close()
		channels[3].ConvertTo(&alphaF, gocv.MatTypeCV32F)
		alphaF.DivideFloat(255.0)

		// Convert each BGR channel and the ROI to float32 for blending
		roiChannels := gocv.Split(roi)
		defer func() {
			for _, ch := range roiChannels {
				ch.Close()
			}
		}()

		for i := 0; i < 3; i++ {
			srcF := gocv.NewMat()
			dstF := gocv.NewMat()
			channels[i].ConvertTo(&srcF, gocv.MatTypeCV32F)
			roiChannels[i].ConvertTo(&dstF, gocv.MatTypeCV32F)

			// result = src * alpha + dst * (1 - alpha)
			onesMat := gocv.NewMatWithSizeFromScalar(gocv.NewScalar(1, 0, 0, 0), alphaF.Rows(), alphaF.Cols(), gocv.MatTypeCV32FC1)
			invAlpha := gocv.NewMat()
			gocv.Subtract(onesMat, alphaF, &invAlpha)
			onesMat.Close()

			srcWeighted := gocv.NewMat()
			dstWeighted := gocv.NewMat()
			gocv.Multiply(srcF, alphaF, &srcWeighted)
			gocv.Multiply(dstF, invAlpha, &dstWeighted)

			blended := gocv.NewMat()
			gocv.Add(srcWeighted, dstWeighted, &blended)

			// Convert back to uint8 and write into the ROI channel
			blended.ConvertTo(&roiChannels[i], gocv.MatTypeCV8U)

			srcF.Close()
			dstF.Close()
			invAlpha.Close()
			srcWeighted.Close()
			dstWeighted.Close()
			blended.Close()
		}

		// Merge blended channels back into the ROI
		gocv.Merge(roiChannels, &roi)
	} else {
		// 3-channel: copy entire contour region directly onto canvas
		crop.CopyTo(&roi)
	}

	return canvas
}

func ResizeImageFileSize(img gocv.Mat, targetSize int) gocv.Mat {
	for GetImageSize(img) > float64(targetSize) {
		reducedRatio := float64(targetSize) / GetImageSize(img)
		newWidth := int(float64(img.Cols()) * math.Sqrt(reducedRatio))
		newHeight := int(float64(img.Rows()) * math.Sqrt(reducedRatio))
		resizedImage := gocv.NewMat()
		gocv.Resize(img, &resizedImage, image.Point{X: newWidth, Y: newHeight}, 0, 0, gocv.InterpolationArea)
		img.Close()
		img = resizedImage
	}
	return img
}

func UpscaleImage(img gocv.Mat, targetDimension int) gocv.Mat {
	scaleRatio := math.Ceil(math.Max(float64(targetDimension)/float64(img.Cols()), float64(targetDimension)/float64(img.Rows())))
	newWidth := int(float64(img.Cols()) * scaleRatio)
	newHeight := int(float64(img.Rows()) * scaleRatio)
	resizedImage := gocv.NewMat()
	gocv.Resize(img, &resizedImage, image.Point{X: newWidth, Y: newHeight}, 0, 0, gocv.InterpolationLanczos4)
	noAlphaImage := gocv.NewMat()
	gocv.CvtColor(resizedImage, &noAlphaImage, gocv.ColorBGRAToBGR)
	resizedImage.Close()
	smoothImage := gocv.NewMat()
	gocv.BilateralFilter(noAlphaImage, &smoothImage, 9, 75, 75)
	noAlphaImage.Close()
	alphaData := gocv.NewMatWithSizeFromScalar(gocv.NewScalar(255, 0, 0, 0), smoothImage.Rows(), smoothImage.Cols(), gocv.MatTypeCV8UC1)
	newImage := gocv.NewMat()
	gocv.CvtColor(smoothImage, &newImage, gocv.ColorBGRToBGRA)
	smoothImage.Close()
	channels := gocv.Split(newImage)
	channels[3] = alphaData
	gocv.Merge(channels, &newImage)
	for _, ch := range channels {
		ch.Close()
	}
	return newImage
}

func DownscaleImage(img gocv.Mat, targetDimension int) gocv.Mat {
	scaleRatio := math.Min(float64(targetDimension)/float64(img.Cols()), float64(targetDimension)/float64(img.Rows()))
	newWidth := int(float64(img.Cols()) * scaleRatio)
	newHeight := int(float64(img.Rows()) * scaleRatio)
	resizedImage := gocv.NewMat()
	gocv.Resize(img, &resizedImage, image.Point{X: newWidth, Y: newHeight}, 0, 0, gocv.InterpolationArea)
	return resizedImage

}

func SetImageSize(img gocv.Mat, targetDimension int) gocv.Mat {
	if img.Cols() < targetDimension || img.Rows() < targetDimension {
		img = UpscaleImage(img, targetDimension)
	}
	return DownscaleImage(img, targetDimension)
}

// By the time an image reaches the remove background step, it should be a 3 channel image
func PreprocessImageForModel(img gocv.Mat, model string) ([]float32, int, int) {
	originalHeight := img.Rows()
	originalWidth := img.Cols()

	var size int
	switch model {
	case "isnet":
		size = ISNetInfo.InputDimension
	case "u2net":
		size = U2NetInfo.InputDimension
	case "biref":
		size = BirefInfo.InputDimension
	default:
		panic("unsupported model: " + model)
	}

	// Resize first
	resized := gocv.NewMat()
	defer resized.Close()
	gocv.Resize(img, &resized, image.Pt(size, size), 0, 0, gocv.InterpolationLinear)

	// Then convert BGR -> RGB on the resized image
	rgb := gocv.NewMat()
	defer rgb.Close()
	gocv.CvtColor(resized, &rgb, gocv.ColorBGRToRGB)

	// Build CHW float32 tensor
	mean := [3]float32{0.485, 0.456, 0.406}
	std := [3]float32{0.229, 0.224, 0.225}

	pixels, _ := rgb.DataPtrUint8()
	tensor := make([]float32, 3*size*size)
	planeSize := size * size

	for i := 0; i < planeSize; i++ {
		for ch := 0; ch < 3; ch++ {
			tensor[ch*planeSize+i] = (float32(pixels[i*3+ch])/255.0 - mean[ch]) / std[ch]
		}
	}

	return tensor, originalHeight, originalWidth
}

func PostProcessMask(mask gocv.Mat) gocv.Mat {
	// Step 1: Morphological opening (erosion followed by dilation)
	// removes small noise pixels and speckles
	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer kernel.Close()

	opened := gocv.NewMat()
	defer opened.Close()
	gocv.MorphologyEx(mask, &opened, gocv.MorphOpen, kernel)

	// Step 2: Gaussian blur to soften edges
	blurred := gocv.NewMat()
	defer blurred.Close()
	gocv.GaussianBlur(opened, &blurred, image.Pt(0, 0), 2, 2, gocv.BorderDefault)
	// Pt(0,0) tells OpenCV to auto-calculate kernel size from sigma

	// Step 3: Re-binarise at midpoint threshold
	result := gocv.NewMat()
	// threshold different for differnt models
	gocv.Threshold(blurred, &result, 200, 255, gocv.ThresholdBinary)

	return result
}

func Postprocessing(maskData []float32, original gocv.Mat, modelInputSize int) gocv.Mat {
	origH := original.Rows()
	origW := original.Cols()

	// Find min/max for normalisation
	min, max := maskData[0], maskData[0]
	for _, v := range maskData[1:] {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	rng := max - min + 1e-8

	// Build normalised grayscale mask at model resolution
	mask := gocv.NewMatWithSize(modelInputSize, modelInputSize, gocv.MatTypeCV8U)
	defer mask.Close()
	maskBytes, _ := mask.DataPtrUint8()
	for i, v := range maskData {
		normalised := (v - min) / rng
		maskBytes[i] = uint8(normalised * 255)
	}

	// Resize mask back to original image dimensions
	maskResized := gocv.NewMat()
	defer maskResized.Close()
	gocv.Resize(mask, &maskResized, image.Pt(origW, origH), 0, 0, gocv.InterpolationLinear)

	// Fill holes before cleaning (closing should happen on raw mask before opening)
	kernel := gocv.GetStructuringElement(gocv.MorphEllipse, image.Pt(25, 25))
	defer kernel.Close()
	maskClosed := gocv.NewMat()
	defer maskClosed.Close()
	gocv.MorphologyEx(maskResized, &maskClosed, gocv.MorphClose, kernel)

	// Clean edges: opening + gaussian + re-binarise
	cleaned := PostProcessMask(maskClosed)
	defer cleaned.Close()

	// Convert original to BGRA
	rgba := gocv.NewMat()
	gocv.CvtColor(original, &rgba, gocv.ColorBGRToBGRA)

	// Replace alpha channel with cleaned mask
	channels := gocv.Split(rgba)
	rgba.Close()
	defer func() {
		for _, ch := range channels {
			ch.Close()
		}
	}()

	cleaned.CopyTo(&channels[3])

	result := gocv.NewMat()
	gocv.Merge(channels, &result)
	return result
}

func RemoveBackground(img gocv.Mat, modelPath string, libPath string, model string) (gocv.Mat, error) {
	// Preprocess
	tensor, origH, origW := PreprocessImageForModel(img, model)

	var size int
	var input string
	var output string
	switch model {
	case "isnet":
		size = ISNetInfo.InputDimension
		input = "input_image"
		output = "output_image"
	case "u2net":
		size = U2NetInfo.InputDimension
		input = "input.1"
		output = "1959"
	case "biref":
		size = BirefInfo.InputDimension
		input = "input_image"
		output = "output_image"
	default:
		return gocv.NewMat(), fmt.Errorf("unsupported model: %s", model)
	}

	// Initialise OnnxRuntime
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to initialise onnxruntime: %w", err)
	}
	defer ort.DestroyEnvironment()

	// Create input tensor (1, 3, size, size)
	inputShape := ort.NewShape(1, 3, int64(size), int64(size))
	inputTensor, err := ort.NewTensor(inputShape, tensor)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor (1, 1, size, size)
	outputShape := ort.NewShape(1, 1, int64(size), int64(size))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Create session and run
	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{input},
		[]string{output},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		nil,
	)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to create onnx session: %w", err)
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to run onnx session: %w", err)
	}

	// Save raw model output as grayscale image for debugging
	rawMask := outputTensor.GetData()
	rawMin, rawMax := rawMask[0], rawMask[0]
	for _, v := range rawMask[1:] {
		if v < rawMin {
			rawMin = v
		}
		if v > rawMax {
			rawMax = v
		}
	}
	rawRng := rawMax - rawMin + 1e-8
	debugMask := gocv.NewMatWithSize(size, size, gocv.MatTypeCV8U)
	debugBytes, _ := debugMask.DataPtrUint8()
	for i, v := range rawMask {
		debugBytes[i] = uint8((v - rawMin) / rawRng * 255)
	}
	gocv.IMWrite("data/output-images/debug_raw_mask.png", debugMask)
	debugMask.Close()
	fmt.Println("saved raw model output to data/output-images/debug_raw_mask.png")

	// Postprocess and return
	result := Postprocessing(rawMask, img, size)
	_ = origH
	_ = origW
	return result, nil
}

func SaveAsPNG(img gocv.Mat, outPath string) error {
	// Convert Mat to image.Image
	goImg, err := img.ToImage()
	if err != nil {
		return fmt.Errorf("failed to convert mat to image: %w", err)
	}

	f, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer f.Close()

	if err := png.Encode(f, goImg); err != nil {
		return fmt.Errorf("failed to encode png: %w", err)
	}
	return nil
}

func MatToBytes(img gocv.Mat) ([]byte, error) {
	buf, err := gocv.IMEncode(gocv.PNGFileExt, img)
	if err != nil {
		return nil, fmt.Errorf("failed to encode mat to bytes: %w", err)
	}
	defer buf.Close()
	return buf.GetBytes(), nil
}
