package pkg

import (
	"os"
)

func GetPort() string {
	if port := os.Getenv("PORT"); port != "" {
		return port // Render automatically sets this
	}
	if port := os.Getenv("FI_MCP_PORT"); port != "" {
		return port
	}
	return "8080"
}
