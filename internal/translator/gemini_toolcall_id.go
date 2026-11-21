// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"hash/crc32"

	"github.com/google/uuid"
)

// Gemini returns a thought signature on functionCall parts and needs it back on the replayed
// parts. OpenAI-schema clients that drop thinking_blocks still echo the tool-call ID verbatim,
// so the translator carries the signature inside the ID:
//
//	<uuid>_ts_<base64url(crc32(signature) || signature)>
//
// Only [A-Za-z0-9_-] is used, so clients that sanitize IDs to that set keep it intact. The
// CRC-32 makes a truncated or rewritten ID decode as "no signature" instead of a corrupted
// signature, which Vertex rejects with 400 "Invalid thought signature".
const (
	toolCallIDSignatureSeparator = "_ts_"
	// maxToolCallIDSignatureBytes caps the signature carried in an ID. Larger signatures are
	// not encoded, and payloads that would decode to more are rejected.
	maxToolCallIDSignatureBytes = 8 << 10
	toolCallIDChecksumBytes     = 4
	toolCallIDUUIDLen           = 36
	toolCallIDPrefixLen         = toolCallIDUUIDLen + len(toolCallIDSignatureSeparator)
)

var maxToolCallIDPayloadLen = base64.RawURLEncoding.EncodedLen(toolCallIDChecksumBytes + maxToolCallIDSignatureBytes)

// newGeminiToolCallID returns a fresh tool-call ID that carries signature, or a plain UUID when
// signature is empty or larger than maxToolCallIDSignatureBytes.
func newGeminiToolCallID(signature []byte) string {
	id := uuid.New().String()
	if len(signature) == 0 || len(signature) > maxToolCallIDSignatureBytes {
		return id
	}
	payload := make([]byte, toolCallIDChecksumBytes+len(signature))
	binary.BigEndian.PutUint32(payload, crc32.ChecksumIEEE(signature))
	copy(payload[toolCallIDChecksumBytes:], signature)
	return id + toolCallIDSignatureSeparator + base64.RawURLEncoding.EncodeToString(payload)
}

// thoughtSignatureFromToolCallID returns the base UUID and the signature carried by id. ok is
// false when id carries no intact signature: it does not start with a UUID and the separator, the
// payload is too long or not base64url, or the checksum does not match (for example after a
// client truncated the ID).
func thoughtSignatureFromToolCallID(id string) (base string, signature []byte, ok bool) {
	if len(id) <= toolCallIDPrefixLen || id[toolCallIDUUIDLen:toolCallIDPrefixLen] != toolCallIDSignatureSeparator {
		return "", nil, false
	}
	base, payload := id[:toolCallIDUUIDLen], id[toolCallIDPrefixLen:]
	if len(payload) > maxToolCallIDPayloadLen || uuid.Validate(base) != nil {
		return "", nil, false
	}
	raw, err := base64.RawURLEncoding.Strict().DecodeString(payload)
	if err != nil || len(raw) <= toolCallIDChecksumBytes {
		return "", nil, false
	}
	signature = raw[toolCallIDChecksumBytes:]
	if binary.BigEndian.Uint32(raw) != crc32.ChecksumIEEE(signature) {
		return "", nil, false
	}
	return base, signature, true
}

// StripGeminiToolCallIDSignatures replaces every JSON string in body that is exactly a tool-call
// ID minted by the GCPVertexAI translator with an intact signature by its base UUID. It reports
// whether body changed; when it did not, body is returned as is. Other providers reject the
// long IDs (OpenAI, Bedrock and Mistral limit their length), so the upstream filter applies this
// to every request it sends to a non-Gemini backend.
func StripGeminiToolCallIDSignatures(body []byte) ([]byte, bool) {
	sep := []byte(toolCallIDSignatureSeparator)
	if !bytes.Contains(body, sep) {
		return body, false
	}
	var out []byte
	copied := 0 // body[:copied] has been written to out.
	for from := 0; ; {
		i := bytes.Index(body[from:], sep)
		if i < 0 {
			break
		}
		i += from
		from = i + len(sep)
		start := i - toolCallIDUUIDLen
		if start < 1 || start < copied || body[start-1] != '"' || escapedQuote(body, start-1) {
			continue
		}
		end := from
		for end < len(body) && isBase64URLByte(body[end]) {
			end++
		}
		if end == len(body) || body[end] != '"' {
			continue
		}
		base, _, ok := thoughtSignatureFromToolCallID(string(body[start:end]))
		if !ok {
			continue
		}
		if out == nil {
			out = make([]byte, 0, len(body))
		}
		out = append(out, body[copied:start]...)
		out = append(out, base...)
		copied, from = end, end
	}
	if out == nil {
		return body, false
	}
	return append(out, body[copied:]...), true
}

// escapedQuote reports whether the quote at body[q] is preceded by an odd number of backslashes,
// that is, whether it is part of a string's content rather than a delimiter.
func escapedQuote(body []byte, q int) bool {
	n := 0
	for j := q - 1; j >= 0 && body[j] == '\\'; j-- {
		n++
	}
	return n%2 == 1
}

func isBase64URLByte(c byte) bool {
	return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '-' || c == '_'
}
