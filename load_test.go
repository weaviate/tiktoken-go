package tiktoken

import (
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type noNetwork struct{}

func (noNetwork) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("network access is not allowed")
}

func TestEncodingsLoadWithoutNetwork(t *testing.T) {
	defaultTransport := http.DefaultTransport
	http.DefaultTransport = noNetwork{}
	t.Cleanup(func() { http.DefaultTransport = defaultTransport })

	tests := []struct {
		encoding   string
		wantTokens []int
	}{
		{encoding: MODEL_O200K_BASE, wantTokens: []int{24912, 2375}},
		{encoding: MODEL_CL100K_BASE, wantTokens: []int{15339, 1917}},
		{encoding: MODEL_P50K_BASE, wantTokens: []int{31373, 995}},
		{encoding: MODEL_P50K_EDIT, wantTokens: []int{31373, 995}},
		{encoding: MODEL_R50K_BASE, wantTokens: []int{31373, 995}},
	}
	for _, tt := range tests {
		t.Run(tt.encoding, func(t *testing.T) {
			tke, err := GetEncoding(tt.encoding)
			require.NoError(t, err)
			assert.Equal(t, tt.wantTokens, tke.Encode("hello world", nil, nil))
		})
	}
}

func TestParseTiktokenBpe(t *testing.T) {
	tests := []struct {
		name     string
		contents string
		want     map[string]int
		wantErr  bool
	}{
		{name: "valid", contents: "YQ== 0\nYg== 1\n", want: map[string]int{"a": 0, "b": 1}},
		{name: "line without rank", contents: "AccessDenied\n", wantErr: true},
		{name: "token not base64", contents: "<?xml 0\n", wantErr: true},
		{name: "rank not a number", contents: "YQ== x\n", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseTiktokenBpe([]byte(tt.contents))
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}
