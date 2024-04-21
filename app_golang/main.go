package main

import (
	_ "image/jpeg"
	_ "image/png"

	"github.com/gofiber/fiber/v2"
	"github.com/joho/godotenv"
)

func main() {

	// Load environment
	if err := godotenv.Load("../.env"); err != nil {
		panic(err.Error())
	}

	app := fiber.New()

	app.Post("/classify", PipeClassify)

	app.Listen(":8888")

}