package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"image"
	"log"
	"os"
	"time"

	"github.com/gofiber/fiber/v2"
	triton "github.com/sunhailin-Leo/triton-service-go/v2/nvidia_inferenceserver"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)


type Flags struct {
	ModelName    string
	ModelVersion string
	BatchSize    int
	URL          string
}

func parseFlags() Flags {

	var flags Flags

	flag.StringVar(&flags.ModelName, "model_name", "ensemble_resnet50", "Name of model being served. (Required)")
	flag.StringVar(&flags.ModelVersion, "model_version", "1", "Version of model. Default: Latest Version.")
	flag.IntVar(&flags.BatchSize, "batch_size", 1, "Batch size. Default: 1.")
	flag.StringVar(&flags.URL, "url", "localhost:8001", "Inference Server URL. Default: localhost:8001")
	flag.Parse()

	return flags

}

func ServerLiveRequest(client triton.GRPCInferenceServiceClient) *triton.ServerLiveResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverLiveRequest := triton.ServerLiveRequest{}
	// Submit ServerLive request to server
	serverLiveResponse, err := client.ServerLive(ctx, &serverLiveRequest)
	if err != nil {
		log.Fatalf("Couldn't get server live: %v", err)
	}
	return serverLiveResponse
}

func ServerReadyRequest(client triton.GRPCInferenceServiceClient) *triton.ServerReadyResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverReadyRequest := triton.ServerReadyRequest{}
	// Submit ServerReady request to server
	serverReadyResponse, err := client.ServerReady(ctx, &serverReadyRequest)
	if err != nil {
		log.Fatalf("Couldn't get server ready: %v", err)
	}
	return serverReadyResponse
}

func ModelMetadataRequest(client triton.GRPCInferenceServiceClient, modelName string, modelVersion string) *triton.ModelMetadataResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create status request for a given model
	modelMetadataRequest := triton.ModelMetadataRequest{
		Name:    modelName,
		Version: modelVersion,
	}
	// Submit modelMetadata request to server
	modelMetadataResponse, err := client.ModelMetadata(ctx, &modelMetadataRequest)
	if err != nil {
		log.Fatalf("Couldn't get server model metadata: %v", err)
	}
	return modelMetadataResponse
}

func ModelInferRequest(client triton.GRPCInferenceServiceClient, rawInput []byte, inputShape []int64, modelName string, modelVersion string) *triton.ModelInferResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create request input tensors
	inferInputs := []*triton.ModelInferRequest_InferInputTensor{
		{
			Name:     "INPUT",
			Datatype: "UINT8",
			Shape:    inputShape,
		},
	}

	// Create request input output tensors
	inferOutputs := []*triton.ModelInferRequest_InferRequestedOutputTensor{
		{
			Name: "OUTPUT",
		},
	}

	// Create inference request for specific model/version
	modelInferRequest := triton.ModelInferRequest{
		ModelName:    modelName,
		ModelVersion: modelVersion,
		Inputs:       inferInputs,
		Outputs:      inferOutputs,
	}

	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, rawInput)

	// Submit inference request to server
	modelInferResponse, err := client.ModelInfer(ctx, &modelInferRequest)

	if err != nil {
		log.Fatalf("Error processing InferRequest: %v", err)
	}
	return modelInferResponse
}

func ConnectTritonServer(FLAGS Flags) (triton.GRPCInferenceServiceClient, *grpc.ClientConn) {

		// Create gRPC server connection
		conn, err := grpc.Dial(FLAGS.URL, grpc.WithTransportCredentials(insecure.NewCredentials()))

		if err != nil {
			log.Fatalf("Couldn't connect to endpoint %s: %v", FLAGS.URL, err)
		}
	
		// Create client from gRPC server connection
		client := triton.NewGRPCInferenceServiceClient(conn)
	
		// Check triton is Live
		serverLiveResponse := ServerLiveRequest(client)
		fmt.Printf("Triton Health - Live: %v\n", serverLiveResponse.Live)
	
		// Check triton is Ready
		serverReadyResponse := ServerReadyRequest(client)
		fmt.Printf("Triton Health - Ready: %v\n", serverReadyResponse.Ready)
	
		// Try send request Meta data
		modelMetadataResponse := ModelMetadataRequest(client, FLAGS.ModelName, FLAGS.ModelVersion)
		fmt.Println(modelMetadataResponse)

		return client, conn
}

func PipeClassify(c *fiber.Ctx) error {

	// Parse flags triton
	FLAGS := parseFlags()
	FLAGS.URL = os.Getenv("TRITON_SERVER_URL")

	client, connector := ConnectTritonServer(FLAGS)
	defer connector.Close()

	// Retrieve the file
	form, err := c.MultipartForm()
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	files := form.File["ImageByte"]
	if len(files) == 0 {
		return fiber.NewError(fiber.StatusBadRequest, "No file uploaded")
	}

	ImageBytes, err := fileToBytes(files)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	// Validate extension
	err = validExtension(ImageBytes[:512])
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error extension not supported": err.Error()})
	}

	img, _, err := image.Decode(bytes.NewReader(ImageBytes))
    if err != nil {
        return c.Status(500).JSON(fiber.Map{"error": err.Error()})
    }

	// Convert image.Image to RGB Slices
	inputShape := []int64 {int64(FLAGS.BatchSize), int64(img.Bounds().Dy()), int64(img.Bounds().Dx()), 3}
	rgbByte, err := Preprocess(img)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	// Model inference from server
	inferResponse := ModelInferRequest(client, rgbByte, inputShape, FLAGS.ModelName, FLAGS.ModelVersion)

	return c.Status(200).JSON(Postprocess(inferResponse))

}