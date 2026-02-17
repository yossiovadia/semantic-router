// BBR ExtProc server with vSR plugin loaded in-process.
// Implements the Envoy External Processor gRPC protocol.
// The vSR BBR plugin runs as an in-process function call — no separate service.
//
// Usage: go run cmd/server/main.go [-port 9002]
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"

	basepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extprocpb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typepb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/examples/bbr-plugin/pkg/plugin"
	bbrplugins "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins"
)

var port = flag.Int("port", 9002, "gRPC port")

// server implements the Envoy ExtProc gRPC service with BBR plugins
type server struct {
	extprocpb.UnimplementedExternalProcessorServer
	plugins []bbrplugins.BBRPlugin
}

func (s *server) Process(srv extprocpb.ExternalProcessor_ProcessServer) error {
	for {
		req, err := srv.Recv()
		if err == io.EOF || err == context.Canceled {
			return nil
		}
		if err != nil {
			return status.Errorf(codes.Unknown, "recv error: %v", err)
		}

		var resp *extprocpb.ProcessingResponse

		switch v := req.Request.(type) {
		case *extprocpb.ProcessingRequest_RequestHeaders:
			// Pass headers through — we process on body
			resp = &extprocpb.ProcessingResponse{
				Response: &extprocpb.ProcessingResponse_RequestHeaders{
					RequestHeaders: &extprocpb.HeadersResponse{},
				},
			}

		case *extprocpb.ProcessingRequest_RequestBody:
			body := v.RequestBody.GetBody()

			// Run all BBR plugins in chain
			var allHeaders []*basepb.HeaderValueOption
			for _, p := range s.plugins {
				mutatedBody, headers, execErr := p.Execute(body)
				if execErr != nil {
					// Plugin blocked the request — return error response
					errBody, _ := json.Marshal(map[string]interface{}{
						"error": map[string]interface{}{
							"code":    403,
							"message": execErr.Error(),
							"type":    "invalid_request_error",
						},
					})
					resp = &extprocpb.ProcessingResponse{
						Response: &extprocpb.ProcessingResponse_ImmediateResponse{
							ImmediateResponse: &extprocpb.ImmediateResponse{
								Status: &typepb.HttpStatus{Code: typepb.StatusCode_Forbidden},
								Body:   errBody,
								Headers: &extprocpb.HeaderMutation{
									SetHeaders: []*basepb.HeaderValueOption{
										{Header: &basepb.HeaderValue{Key: "content-type", RawValue: []byte("application/json")}},
									},
								},
							},
						},
					}
					if sendErr := srv.Send(resp); sendErr != nil {
						return sendErr
					}
					return nil
				}

				body = mutatedBody

				// Collect headers from plugin
				for k, vals := range headers {
					for _, v := range vals {
						allHeaders = append(allHeaders, &basepb.HeaderValueOption{
							Header: &basepb.HeaderValue{
								Key:      k,
								RawValue: []byte(v),
							},
						})
					}
				}
			}

			// Build response with mutated body + plugin headers
			resp = &extprocpb.ProcessingResponse{
				Response: &extprocpb.ProcessingResponse_RequestBody{
					RequestBody: &extprocpb.BodyResponse{
						Response: &extprocpb.CommonResponse{
							ClearRouteCache: true,
							HeaderMutation: &extprocpb.HeaderMutation{
								SetHeaders: allHeaders,
							},
							BodyMutation: &extprocpb.BodyMutation{
								Mutation: &extprocpb.BodyMutation_Body{
									Body: body,
								},
							},
						},
					},
				},
			}

		case *extprocpb.ProcessingRequest_ResponseHeaders:
			resp = &extprocpb.ProcessingResponse{
				Response: &extprocpb.ProcessingResponse_ResponseHeaders{
					ResponseHeaders: &extprocpb.HeadersResponse{},
				},
			}

		case *extprocpb.ProcessingRequest_ResponseBody:
			resp = &extprocpb.ProcessingResponse{
				Response: &extprocpb.ProcessingResponse_ResponseBody{
					ResponseBody: &extprocpb.BodyResponse{},
				},
			}

		default:
			continue
		}

		if err := srv.Send(resp); err != nil {
			return err
		}
	}
}

func main() {
	flag.Parse()

	// Create vSR BBR plugin
	vsrPlugin := plugin.NewSemanticRouterPluginWithConfig(plugin.DefaultConfig())
	log.Printf("Loaded plugin: %s/%s", vsrPlugin.TypedName().Type, vsrPlugin.TypedName().Name)

	// Create gRPC server
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	extprocpb.RegisterExternalProcessorServer(grpcServer, &server{
		plugins: []bbrplugins.BBRPlugin{vsrPlugin},
	})

	log.Printf("BBR ExtProc server with vSR plugin listening on :%d", *port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
