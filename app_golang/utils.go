package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image"
	"image/draw"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"strings"

	triton "github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
)

const (
	outputSize = 1000
)

type MaxConfident struct {
	Class     string
	Confident float32
}

func validExtension(imageByte []byte) error {
    formatted := strings.Split(http.DetectContentType(imageByte), "/")
    ext := formatted[len(formatted)-1]

    for _, validExt := range []string{"png", "jpeg"} {
        if ext == validExt {
            return nil
        }
    }

    return errors.New("unknown Format")

}

func fileToBytes(files []*multipart.FileHeader) ([]byte, error) {

	var ImageByte []byte

	// Iterate over each file
	for _, file := range files {
		// Open the uploaded file
		src, err := file.Open()
		if err != nil {
			return ImageByte, err
		}
		
		defer src.Close()

		// Read the file content into a byte slice
		fileBytes, err := io.ReadAll(src)
		if err != nil {
			return ImageByte, err
		}

		ImageByte = append(ImageByte, fileBytes...)

	}

	return ImageByte, nil

}

func imageToRGBA(src image.Image) *image.RGBA {

    // No conversion needed if image is an *image.RGBA.
    if dst, ok := src.(*image.RGBA); ok {
        return dst
    }

    // Use the image/draw package to convert to *image.RGBA.
    b := src.Bounds()
    dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
    draw.Draw(dst, dst.Bounds(), src, b.Min, draw.Src)
    return dst
}

func ImageToRGBSlice(img image.Image) ([][][]uint8, error) {
	// Get the dimensions of the image
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// Create a slice to store the RGB values
	rgbSlice := make([]uint8, width*height*3)

	// Iterate over each pixel and store its RGB values in the slice
	index := 0
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// Convert RGBA to RGB by discarding the alpha channel
			rgbSlice[index] = uint8(r >> 8)
			rgbSlice[index+1] = uint8(g >> 8)
			rgbSlice[index+2] = uint8(b >> 8)
			index += 3
		}
	}

	// Reshape the slice to have the shape [height][width][3]
	reshapedSlice := make([][][]uint8, height)
	for y := 0; y < height; y++ {
		reshapedSlice[y] = make([][]uint8, width)
		for x := 0; x < width; x++ {
			pixelIndex := y*width*3 + x*3
			reshapedSlice[y][x] = rgbSlice[pixelIndex : pixelIndex+3]
		}
	}

	return reshapedSlice, nil
}

func RGBSliceToBytes(rgbSlice [][][]uint8) []byte {
	var buf []byte
	for _, row := range rgbSlice {
		for _, pixel := range row {
			buf = append(buf, pixel...)
		}
	}
	return buf
}

// Preprocess image (image.Image -> image.RGBA -> RGB slices -> [][][]byte)
func Preprocess(img image.Image) ([]byte, error) {
	
	img = imageToRGBA(img)

	rgbSlice, err := ImageToRGBSlice(img)
	if err != nil {
		panic(err)
	}

	rgbByte := RGBSliceToBytes(rgbSlice)

	return rgbByte, nil

}

// Reader 4 chunk of byte to single float
func readFloat32(fourBytes []byte) float32 {
	buf := bytes.NewBuffer(fourBytes)
	var retval float32
	binary.Read(buf, binary.LittleEndian, &retval)
	return retval
}

// Find argmax from logits
func argMax(logits [][]float32) []MaxConfident {

	var out []MaxConfident

	classes := map[uint16]string{
		0: "DAISY",
		1: "DANDELION",
		2: "ROSE",
		3: "SUNFLOWER",
		4: "TULIP",
		5: "NONE",
	}
	

	for _, logit := range logits {
		maxValue := float32(math.Inf(-1))
		var maxIndex uint16
		for index, confident := range logit {
			if confident > maxValue {
				maxValue = confident
				maxIndex = uint16(index)
			}
		}
		if maxIndex > 4 {
			maxIndex = 5
		}
		max := MaxConfident{Class: classes[maxIndex], Confident: maxValue}
		out = append(out, max)
	}
	return out
}

// Convert output's raw bytes into float32 data (assumes Little Endian)
func Postprocess(inferResponse *triton.ModelInferResponse) []MaxConfident {

	var logits [][]float32

	for _, outputBytes := range inferResponse.RawOutputContents {

		outputData := make([]float32, outputSize)

		for i := 0; i < outputSize; i++ {
			outputData[i] = readFloat32(outputBytes[i*4 : i*4+4])
		}

		logits = append(logits, outputData)

	}

	out := argMax(logits)

	return out
}
