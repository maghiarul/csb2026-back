package main

import (
	"encoding/json"
	"net/http"
)

type Response struct {
	Message string `json:"message"`
}

func handler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")
	
	json.NewEncoder(w).Encode(Response{Message: "Backend connected"})
}

func main() {
	http.HandleFunc("/api/data", handler)
	http.ListenAndServe(":8080", nil)
}