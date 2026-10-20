watch -n 3 "curl -ks http://localhost:8080/v1/metrics|jq '.active_requests[0]'"
