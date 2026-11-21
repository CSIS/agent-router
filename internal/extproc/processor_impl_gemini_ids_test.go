// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package extproc

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"log/slog"
	"strings"
	"testing"

	extprocv3 "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/envoyproxy/ai-gateway/internal/endpointspec"
	"github.com/envoyproxy/ai-gateway/internal/filterapi"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
)

const geminiIDTestUUID = "6f1c2a9e-4b7d-4e2a-9c3f-1a2b3c4d5e6f"

func signedGeminiToolCallID(base string, signature []byte) string {
	payload := binary.BigEndian.AppendUint32(nil, crc32.ChecksumIEEE(signature))
	return base + "_ts_" + base64.RawURLEncoding.EncodeToString(append(payload, signature...))
}

func emittedRequestBody(t *testing.T, resp *extprocv3.ProcessingResponse) []byte {
	t.Helper()
	return resp.Response.(*extprocv3.ProcessingResponse_RequestHeaders).RequestHeaders.Response.GetBodyMutation().GetBody()
}

func Test_chatCompletionProcessorUpstreamFilter_geminiToolCallIDSignatures(t *testing.T) {
	signature := []byte("real-thought-signature-\x00\xff-bytes")
	signedID := signedGeminiToolCallID(geminiIDTestUUID, signature)
	rawBody := fmt.Appendf(nil, `{"model":"some-model","messages":[`+
		`{"role":"user","content":"list files"},`+
		`{"role":"assistant","tool_calls":[{"id":%q,"type":"function","function":{"name":"ls","arguments":"{}"}}]},`+
		`{"role":"tool","tool_call_id":%q,"content":"a.txt"},`+
		`{"role":"user","content":"which is largest?"}]}`, signedID, signedID)

	newRouter := func(t *testing.T) *chatCompletionProcessorRouterFilter {
		_, body, _, _, err := endpointspec.ChatCompletionsEndpointSpec{}.ParseBody(rawBody, false)
		require.NoError(t, err)
		return &chatCompletionProcessorRouterFilter{
			eh:                     endpointspec.ChatCompletionsEndpointSpec{},
			config:                 &filterapi.RuntimeConfig{},
			logger:                 slog.Default(),
			requestHeaders:         map[string]string{":path": "/v1/chat/completions"},
			originalRequestBodyRaw: rawBody,
			originalRequestBody:    body,
			originalModel:          body.Model,
		}
	}
	send := func(t *testing.T, rp *chatCompletionProcessorRouterFilter, schema filterapi.VersionedAPISchema) []byte {
		t.Helper()
		rp.upstreamFilterCount = 0
		p := &chatCompletionProcessorUpstreamFilter{
			requestHeaders: map[string]string{":path": "/v1/chat/completions", internalapi.ModelNameHeaderKeyDefault: "some-model"},
			metrics:        &mockMetrics{},
			logger:         slog.Default(),
		}
		require.NoError(t, p.SetBackend(t.Context(), &filterapi.RuntimeBackend{
			Backend: &filterapi.Backend{Name: "backend", Schema: schema},
		}, "route", rp))
		resp, err := p.ProcessRequestHeaders(t.Context(), nil)
		require.NoError(t, err)
		return emittedRequestBody(t, resp)
	}

	for _, tc := range []struct {
		name   string
		schema filterapi.VersionedAPISchema
		ids    func(body []byte) []string
	}{
		{
			name:   "OpenAI pass-through",
			schema: filterapi.VersionedAPISchema{Name: filterapi.APISchemaOpenAI},
			ids:    openAIChatToolCallIDs,
		},
		{
			name:   "AzureOpenAI pass-through",
			schema: filterapi.VersionedAPISchema{Name: filterapi.APISchemaAzureOpenAI, Version: "2025-01-01"},
			ids:    openAIChatToolCallIDs,
		},
		{
			name:   "GCPAnthropic translated",
			schema: filterapi.VersionedAPISchema{Name: filterapi.APISchemaGCPAnthropic, Version: "vertex-2023-10-16"},
			ids: func(body []byte) []string {
				return []string{
					gjson.GetBytes(body, `messages.#(role=="assistant").content.#(type=="tool_use").id`).String(),
					gjson.GetBytes(body, `messages.#(content.#(type=="tool_result")).content.#(type=="tool_result").tool_use_id`).String(),
				}
			},
		},
	} {
		t.Run(tc.name+" gets the base UUID", func(t *testing.T) {
			body := send(t, newRouter(t), tc.schema)
			require.NotEmpty(t, body)
			require.Equal(t, []string{geminiIDTestUUID, geminiIDTestUUID}, tc.ids(body))
			require.NotContains(t, string(body), "_ts_")
		})
	}

	t.Run("GCPVertexAI gets the signature", func(t *testing.T) {
		body := send(t, newRouter(t), filterapi.VersionedAPISchema{Name: filterapi.APISchemaGCPVertexAI})
		require.Equal(t, base64.StdEncoding.EncodeToString(signature),
			gjson.GetBytes(body, `contents.#(role=="model").parts.0.thoughtSignature`).String())
	})

	t.Run("GCPVertexAI retry after an OpenAI attempt still gets the signature", func(t *testing.T) {
		rp := newRouter(t)
		require.Equal(t, []string{geminiIDTestUUID, geminiIDTestUUID},
			openAIChatToolCallIDs(send(t, rp, filterapi.VersionedAPISchema{Name: filterapi.APISchemaOpenAI})))
		body := send(t, rp, filterapi.VersionedAPISchema{Name: filterapi.APISchemaGCPVertexAI})
		require.Equal(t, base64.StdEncoding.EncodeToString(signature),
			gjson.GetBytes(body, `contents.#(role=="model").parts.0.thoughtSignature`).String())
	})

	t.Run("damaged ID is sent unchanged", func(t *testing.T) {
		truncated := signedID[:len(signedID)-3]
		raw := []byte(strings.ReplaceAll(string(rawBody), signedID, truncated))
		rp := newRouter(t)
		_, body, _, _, err := endpointspec.ChatCompletionsEndpointSpec{}.ParseBody(raw, false)
		require.NoError(t, err)
		rp.originalRequestBodyRaw, rp.originalRequestBody = raw, body
		require.Nil(t, send(t, rp, filterapi.VersionedAPISchema{Name: filterapi.APISchemaOpenAI}))
	})
}

func openAIChatToolCallIDs(body []byte) []string {
	return []string{
		gjson.GetBytes(body, `messages.#(role=="assistant").tool_calls.0.id`).String(),
		gjson.GetBytes(body, `messages.#(role=="tool").tool_call_id`).String(),
	}
}

func Test_messagesProcessorUpstreamFilter_geminiToolCallIDSignatures(t *testing.T) {
	// OpenCode keeps the Gemini tool-call IDs when a session switches to a Claude model on the
	// Anthropic Messages endpoint; it only replaces characters outside [A-Za-z0-9_-].
	signedID := signedGeminiToolCallID(geminiIDTestUUID, []byte("sig"))
	raw := fmt.Appendf(nil, `{"model":"claude","max_tokens":64,"messages":[`+
		`{"role":"user","content":"list files"},`+
		`{"role":"assistant","content":[{"type":"tool_use","id":%q,"name":"ls","input":{}}]},`+
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":%q,"content":"a.txt"}]}]}`, signedID, signedID)
	_, body, _, _, err := endpointspec.MessagesEndpointSpec{}.ParseBody(raw, false)
	require.NoError(t, err)

	p := &messagesProcessorUpstreamFilter{
		requestHeaders: map[string]string{":path": "/v1/messages", internalapi.ModelNameHeaderKeyDefault: "claude"},
		metrics:        &mockMetrics{},
		logger:         slog.Default(),
	}
	r := &messagesProcessorRouterFilter{
		eh:                     endpointspec.MessagesEndpointSpec{},
		config:                 &filterapi.RuntimeConfig{},
		logger:                 slog.Default(),
		requestHeaders:         map[string]string{":path": "/v1/messages"},
		originalRequestBodyRaw: raw,
		originalRequestBody:    body,
		originalModel:          body.Model,
	}
	require.NoError(t, p.SetBackend(t.Context(), &filterapi.RuntimeBackend{
		Backend: &filterapi.Backend{Name: "anthropic", Schema: filterapi.VersionedAPISchema{Name: filterapi.APISchemaAnthropic}},
	}, "route", r))
	resp, err := p.ProcessRequestHeaders(t.Context(), nil)
	require.NoError(t, err)

	out := emittedRequestBody(t, resp)
	require.Equal(t, geminiIDTestUUID, gjson.GetBytes(out, `messages.1.content.0.id`).String())
	require.Equal(t, geminiIDTestUUID, gjson.GetBytes(out, `messages.2.content.0.tool_use_id`).String())
}
