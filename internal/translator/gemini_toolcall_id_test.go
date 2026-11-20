// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math/rand/v2"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/json"
)

// signedToolCallID builds an ID in the documented format, independently of newGeminiToolCallID:
// <uuid>_ts_<base64url(crc32(signature) big-endian || signature)>.
func signedToolCallID(base string, signature []byte) string {
	payload := binary.BigEndian.AppendUint32(nil, crc32.ChecksumIEEE(signature))
	return base + "_ts_" + base64.RawURLEncoding.EncodeToString(append(payload, signature...))
}

func testSignature(n int) []byte {
	var seed [32]byte
	copy(seed[:], strconv.Itoa(n))
	sig := make([]byte, n)
	_, _ = rand.NewChaCha8(seed).Read(sig) // Deterministic test data; never fails.
	return sig
}

var idCharset = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

func TestNewGeminiToolCallID_carriesSignature(t *testing.T) {
	// 204 and 471 bytes are the smallest and largest signatures seen from Gemini 4 Hydrogen and Neon.
	for _, n := range []int{1, 204, 471, maxToolCallIDSignatureBytes} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			sig := testSignature(n)
			id := newGeminiToolCallID(sig)

			require.Regexp(t, idCharset, id)
			require.NoError(t, uuid.Validate(id[:36]))
			require.Equal(t, signedToolCallID(id[:36], sig), id)
			base, got, ok := thoughtSignatureFromToolCallID(id)
			require.True(t, ok)
			require.Equal(t, id[:36], base)
			require.Equal(t, sig, got)
		})
	}
	require.Len(t, newGeminiToolCallID(testSignature(471)), 674)
}

func TestNewGeminiToolCallID_plainUUIDWithoutUsableSignature(t *testing.T) {
	for name, sig := range map[string][]byte{
		"nil":       nil,
		"empty":     {},
		"over cap":  testSignature(maxToolCallIDSignatureBytes + 1),
		"far above": testSignature(64 << 10),
	} {
		t.Run(name, func(t *testing.T) {
			id := newGeminiToolCallID(sig)
			require.NoError(t, uuid.Validate(id))
			require.Len(t, id, 36)
		})
	}
}

func TestThoughtSignatureFromToolCallID_rejectsDamagedAndForeignIDs(t *testing.T) {
	sig := testSignature(300)
	id := signedToolCallID("6f1c2a9e-4b7d-4e2a-9c3f-1a2b3c4d5e6f", sig)

	t.Run("every truncation", func(t *testing.T) {
		// pi truncates IDs to 40 or 64 characters when a session switches provider.
		for n := range len(id) {
			_, _, ok := thoughtSignatureFromToolCallID(id[:n])
			require.False(t, ok, "truncated to %d", n)
		}
	})
	t.Run("every single-character change in the payload", func(t *testing.T) {
		for i := 40; i < len(id); i++ {
			c := byte('A')
			if id[i] == 'A' {
				c = 'B'
			}
			_, _, ok := thoughtSignatureFromToolCallID(id[:i] + string(c) + id[i+1:])
			require.False(t, ok, "changed at %d", i)
		}
	})
	for name, foreign := range map[string]string{
		"empty":                    "",
		"plain UUID":               "6f1c2a9e-4b7d-4e2a-9c3f-1a2b3c4d5e6f",
		"separator only":           "6f1c2a9e-4b7d-4e2a-9c3f-1a2b3c4d5e6f_ts_",
		"OpenAI":                   "call_ts_abcdefghijklmnopqrstuvwxyz0123",
		"Anthropic":                "toolu_01A09q90qw90lq917835lq9",
		"non-UUID base":            "6f1c2a9e_4b7d_4e2a_9c3f_1a2b3c4d5e6f" + id[36:],
		"padded base64":            id + "==",
		"standard base64 alphabet": strings.ReplaceAll(strings.ReplaceAll(id[:40], "-", "+"), "_", "/") + id[40:],
		"over the size cap": signedToolCallID("6f1c2a9e-4b7d-4e2a-9c3f-1a2b3c4d5e6f",
			testSignature(maxToolCallIDSignatureBytes+1)),
	} {
		t.Run(name, func(t *testing.T) {
			_, _, ok := thoughtSignatureFromToolCallID(foreign)
			require.False(t, ok)
		})
	}
}

func TestStripGeminiToolCallIDSignatures(t *testing.T) {
	const base = "6f1c2a9e-4b7d-4e2a-9c3f-1a2b3c4d5e6f"
	const other = "0b8e9f3c-2d1a-4c5b-8e7f-6a5b4c3d2e1f"
	id1 := signedToolCallID(base, testSignature(200))
	id2 := signedToolCallID(other, testSignature(400))
	damaged := id2[:len(id2)-1]

	t.Run("replaces whole-string IDs and keeps everything else", func(t *testing.T) {
		in := fmt.Sprintf(`{"messages":[{"tool_calls":[{"id":%q},{"id":%q}]},{"tool_call_id":%q},`+
			`{"content":"mentions %s inline"},{"content":"quoted \"%s\""},{"id":%q}]}`,
			id1, id2, id1, id1, id1, damaged)
		out, changed := StripGeminiToolCallIDSignatures([]byte(in))
		require.True(t, changed)
		require.True(t, gjson.ValidBytes(out))
		want := fmt.Sprintf(`{"messages":[{"tool_calls":[{"id":%q},{"id":%q}]},{"tool_call_id":%q},`+
			`{"content":"mentions %s inline"},{"content":"quoted \"%s\""},{"id":%q}]}`,
			base, other, base, id1, id1, damaged)
		require.Equal(t, want, string(out))
	})
	t.Run("unchanged body is returned as is", func(t *testing.T) {
		for _, in := range []string{
			`{"messages":[{"role":"user","content":"hi"}]}`,
			fmt.Sprintf(`{"id":%q}`, damaged),
			`"_ts_"`,
			`_ts_`,
		} {
			out, changed := StripGeminiToolCallIDSignatures([]byte(in))
			require.False(t, changed, in)
			require.Equal(t, in, string(out))
		}
	})
}

func FuzzThoughtSignatureFromToolCallID(f *testing.F) {
	id := signedToolCallID("6f1c2a9e-4b7d-4e2a-9c3f-1a2b3c4d5e6f", testSignature(300))
	for _, seed := range []string{id, id[:64], id[:40], "", "call_abc", "6f1c2a9e-4b7d-4e2a-9c3f-1a2b3c4d5e6f_ts_AAAAAAA"} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, id string) {
		base, sig, ok := thoughtSignatureFromToolCallID(id)
		if !ok {
			return
		}
		// Only an ID in the documented format, with an in-range signature, decodes.
		require.Equal(t, signedToolCallID(base, sig), id)
		require.NotEmpty(t, sig)
		require.LessOrEqual(t, len(sig), maxToolCallIDSignatureBytes)
	})
}

func FuzzGeminiToolCallIDRoundTrip(f *testing.F) {
	f.Add([]byte("signature"))
	f.Add(testSignature(471))
	f.Fuzz(func(t *testing.T, sig []byte) {
		id := newGeminiToolCallID(sig)
		require.Regexp(t, idCharset, id)
		_, got, ok := thoughtSignatureFromToolCallID(id)
		if len(sig) == 0 || len(sig) > maxToolCallIDSignatureBytes {
			require.False(t, ok)
			return
		}
		require.True(t, ok)
		require.Equal(t, sig, got)
	})
}

func FuzzStripGeminiToolCallIDSignatures(f *testing.F) {
	id := signedToolCallID("6f1c2a9e-4b7d-4e2a-9c3f-1a2b3c4d5e6f", testSignature(64))
	f.Add([]byte(fmt.Sprintf(`{"tool_calls":[{"id":%q}],"tool_call_id":%q}`, id, id)))
	f.Add([]byte(fmt.Sprintf(`{"c":"\"%s\"","d":"\\"%s"}`, id, id)))
	f.Add([]byte(`{"messages":[]}`))
	f.Fuzz(func(t *testing.T, body []byte) {
		out, changed := StripGeminiToolCallIDSignatures(body)
		if !changed {
			require.Equal(t, body, out)
			return
		}
		require.Less(t, len(out), len(body))
		if gjson.ValidBytes(body) {
			require.True(t, gjson.ValidBytes(out))
		}
		again, changedAgain := StripGeminiToolCallIDSignatures(out)
		require.False(t, changedAgain)
		require.Equal(t, out, again)
	})
}

// geminiFunctionCallResponse is a generateContent response with two parallel function calls;
// Gemini puts the thought signature on the first one only.
func geminiFunctionCallResponse(sig []byte, finishReason string) string {
	return fmt.Sprintf(`{"candidates":[{"content":{"role":"model","parts":[`+
		`{"functionCall":{"name":"ls","args":{"path":"."}},"thoughtSignature":%q},`+
		`{"functionCall":{"name":"cat","args":{"path":"a.txt"}}}]},"finishReason":%q}],`+
		`"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
		base64.StdEncoding.EncodeToString(sig), finishReason)
}

// replayRequest is the next request of a client that drops thinking_blocks but echoes tool-call
// IDs, as pi 1.5.2 and OpenCode 1.19.6 do.
func replayRequest(t *testing.T, ids []string) *openai.ChatCompletionRequest {
	t.Helper()
	raw := fmt.Sprintf(`{"model":"gemini-4-hydrogen","messages":[`+
		`{"role":"user","content":"list and read"},`+
		`{"role":"assistant","tool_calls":[`+
		`{"id":%q,"type":"function","function":{"name":"ls","arguments":"{\"path\":\".\"}"}},`+
		`{"id":%q,"type":"function","function":{"name":"cat","arguments":"{\"path\":\"a.txt\"}"}}]},`+
		`{"role":"tool","tool_call_id":%q,"content":"a.txt"},`+
		`{"role":"tool","tool_call_id":%q,"content":"hello"},`+
		`{"role":"user","content":"now read b.txt"}]}`, ids[0], ids[1], ids[0], ids[1])
	var req openai.ChatCompletionRequest
	require.NoError(t, json.Unmarshal([]byte(raw), &req))
	return &req
}

func geminiRequestFor(t *testing.T, req *openai.ChatCompletionRequest) []byte {
	t.Helper()
	_, body, err := NewChatCompletionOpenAIToGCPVertexAITranslator("").RequestBody(nil, req, false)
	require.NoError(t, err)
	return body
}

func modelPartSignatures(body []byte) []string {
	var sigs []string
	for _, p := range gjson.GetBytes(body, `contents.#(role=="model").parts`).Array() {
		sigs = append(sigs, p.Get("thoughtSignature").String())
	}
	return sigs
}

func TestOpenAIToGCPVertexAITranslator_thoughtSignatureRoundTripsThroughToolCallIDs(t *testing.T) {
	sig := testSignature(471)
	wantSig := base64.StdEncoding.EncodeToString(sig)

	t.Run("non-streaming", func(t *testing.T) {
		tr := NewChatCompletionOpenAIToGCPVertexAITranslator("")
		_, _, err := tr.RequestBody(nil, &openai.ChatCompletionRequest{Model: "gemini-4-hydrogen"}, false)
		require.NoError(t, err)
		_, body, _, _, err := tr.ResponseBody(nil, strings.NewReader(geminiFunctionCallResponse(sig, "STOP")), true, nil)
		require.NoError(t, err)

		var resp openai.ChatCompletionResponse
		require.NoError(t, json.Unmarshal(body, &resp))
		msg := resp.Choices[0].Message
		require.Len(t, msg.ToolCalls, 2)
		ids := []string{*msg.ToolCalls[0].ID, *msg.ToolCalls[1].ID}
		require.Regexp(t, idCharset, ids[0])
		require.NoError(t, uuid.Validate(ids[1]), "the unsigned call keeps a plain UUID")
		require.Equal(t, wantSig, msg.ThinkingBlocks[0].Signature, "thinking_blocks is unchanged")

		require.Equal(t, []string{wantSig, ""}, modelPartSignatures(geminiRequestFor(t, replayRequest(t, ids))))
	})

	t.Run("streaming", func(t *testing.T) {
		tr := NewChatCompletionOpenAIToGCPVertexAITranslator("")
		_, _, err := tr.RequestBody(nil, &openai.ChatCompletionRequest{Model: "gemini-4-hydrogen", Stream: true}, false)
		require.NoError(t, err)
		sse := "data: " + geminiFunctionCallResponse(sig, "STOP") + "\n\n"
		_, body, _, _, err := tr.ResponseBody(nil, strings.NewReader(sse), true, nil)
		require.NoError(t, err)

		var ids []string
		for line := range bytes.Lines(body) {
			data, ok := bytes.CutPrefix(bytes.TrimSpace(line), []byte("data: "))
			if !ok || string(data) == "[DONE]" {
				continue
			}
			for _, tc := range gjson.GetBytes(data, "choices.0.delta.tool_calls").Array() {
				ids = append(ids, tc.Get("id").String())
			}
		}
		require.Len(t, ids, 2)
		require.NoError(t, uuid.Validate(ids[1]))

		require.Equal(t, []string{wantSig, ""}, modelPartSignatures(geminiRequestFor(t, replayRequest(t, ids))))
	})
}

func TestAssistantMsgToGeminiParts_signaturePrecedence(t *testing.T) {
	idSig := testSignature(64)
	signedID := signedToolCallID("6f1c2a9e-4b7d-4e2a-9c3f-1a2b3c4d5e6f", idSig)
	secondID := signedToolCallID("0b8e9f3c-2d1a-4c5b-8e7f-6a5b4c3d2e1f", testSignature(32))
	dummy := base64.StdEncoding.EncodeToString(dummyThoughtSignature)

	t.Run("thinking_blocks wins for the first call", func(t *testing.T) {
		req := replayRequest(t, []string{signedID, "plain-2"})
		req.Messages[1].OfAssistant.ThinkingBlocks = []openai.ThinkingBlock{{Type: "thinking", Signature: "YmxvY2s="}}
		require.Equal(t, []string{"YmxvY2s=", ""}, modelPartSignatures(geminiRequestFor(t, req)))
	})
	t.Run("each call gets the signature in its own ID", func(t *testing.T) {
		require.Equal(t,
			[]string{base64.StdEncoding.EncodeToString(idSig), base64.StdEncoding.EncodeToString(testSignature(32))},
			modelPartSignatures(geminiRequestFor(t, replayRequest(t, []string{signedID, secondID}))))
	})
	t.Run("truncated ID falls back to the dummy", func(t *testing.T) {
		// pi truncates to 40 characters for OpenAI and 64 for Anthropic-style providers.
		for _, n := range []int{40, 64, len(signedID) - 1} {
			require.Equal(t, []string{dummy, ""},
				modelPartSignatures(geminiRequestFor(t, replayRequest(t, []string{signedID[:n], "plain-2"}))))
		}
	})
	t.Run("charset-sanitized client ID falls back to the dummy", func(t *testing.T) {
		require.Equal(t, []string{dummy, ""},
			modelPartSignatures(geminiRequestFor(t, replayRequest(t, []string{"call_abc.def", "plain-2"}))))
	})
}

func TestOpenAIToGCPVertexAITranslator_malformedResponseFinishReason(t *testing.T) {
	malformed := `{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"MALFORMED_RESPONSE"}],` +
		`"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":8,"totalTokenCount":77,"thoughtsTokenCount":64}}`

	t.Run("non-streaming", func(t *testing.T) {
		tr := NewChatCompletionOpenAIToGCPVertexAITranslator("")
		_, _, err := tr.RequestBody(nil, &openai.ChatCompletionRequest{Model: "gemini-4-hydrogen"}, false)
		require.NoError(t, err)
		_, body, _, _, err := tr.ResponseBody(nil, strings.NewReader(malformed), true, nil)
		require.NoError(t, err)
		require.Equal(t, "malformed_response", gjson.GetBytes(body, "choices.0.finish_reason").String())
	})
	t.Run("streaming", func(t *testing.T) {
		tr := NewChatCompletionOpenAIToGCPVertexAITranslator("")
		_, _, err := tr.RequestBody(nil, &openai.ChatCompletionRequest{Model: "gemini-4-hydrogen", Stream: true}, false)
		require.NoError(t, err)
		_, body, _, _, err := tr.ResponseBody(nil, strings.NewReader("data: "+malformed+"\n\n"), true, nil)
		require.NoError(t, err)
		require.Contains(t, string(body), `"finish_reason":"malformed_response"`)
	})
}
