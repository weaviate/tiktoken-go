package tiktoken

import (
	"bytes"
	"compress/gzip"
	"embed"
	"encoding/base64"
	"fmt"
	"io"
	"path"
	"strconv"
	"strings"
)

// The vocabularies are the files OpenAI publishes under
// https://openaipublic.blob.core.windows.net/encodings/, gzipped. They are
// embedded so that loading an encoding never touches the network.
//
//go:embed assets/*.tiktoken.gz
var assets embed.FS

type BpeLoader interface {
	LoadTiktokenBpe(tiktokenBpeFile string) (map[string]int, error)
}

// defaultBpeLoader serves the embedded vocabulary named by the base of
// tiktokenBpeFile, e.g. ".../encodings/cl100k_base.tiktoken".
type defaultBpeLoader struct{}

func (l *defaultBpeLoader) LoadTiktokenBpe(tiktokenBpeFile string) (map[string]int, error) {
	compressed, err := assets.ReadFile("assets/" + path.Base(tiktokenBpeFile) + ".gz")
	if err != nil {
		return nil, fmt.Errorf("no embedded vocabulary for %s: %w", tiktokenBpeFile, err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		return nil, fmt.Errorf("decompress vocabulary %s: %w", tiktokenBpeFile, err)
	}
	defer zr.Close()
	contents, err := io.ReadAll(zr)
	if err != nil {
		return nil, fmt.Errorf("decompress vocabulary %s: %w", tiktokenBpeFile, err)
	}
	return parseTiktokenBpe(contents)
}

func NewDefaultBpeLoader() BpeLoader {
	return &defaultBpeLoader{}
}

// parseTiktokenBpe decodes "<base64 token> <rank>" lines.
func parseTiktokenBpe(contents []byte) (map[string]int, error) {
	bpeRanks := make(map[string]int)
	for _, line := range strings.Split(string(contents), "\n") {
		if line == "" {
			continue
		}
		encodedToken, encodedRank, ok := strings.Cut(line, " ")
		if !ok {
			return nil, fmt.Errorf("malformed vocabulary line %q", line)
		}
		token, err := base64.StdEncoding.DecodeString(encodedToken)
		if err != nil {
			return nil, fmt.Errorf("malformed vocabulary line %q: %w", line, err)
		}
		rank, err := strconv.Atoi(encodedRank)
		if err != nil {
			return nil, fmt.Errorf("malformed vocabulary line %q: %w", line, err)
		}
		bpeRanks[string(token)] = rank
	}
	return bpeRanks, nil
}
