package schemasv1

import (
	"strings"

	"github.com/huandu/xstrings"
)

const ValueQMe = "@me"
const KeyQKeywords = "__keywords"
const KeyQIn = "in"

type Q string

func (q Q) ToMap() map[string]interface{} {
	res := map[string]interface{}{}
	for _, piece := range strings.Split(string(q), " ") {
		piece = strings.TrimSpace(piece)
		if piece == "" {
			continue
		}
		var k string
		var v string
		if !strings.Contains(piece, ":") {
			k = KeyQKeywords
			v = piece
		} else {
			k, _, v = xstrings.Partition(piece, ":")
			if v == "" {
				continue
			}
			if k == "is" {
				res[v] = true
				continue
			}
			if k == "not" {
				res[v] = false
				continue
			}
		}
		v_, ok := res[k]
		if !ok {
			v_ = make([]string, 0)
		}
		v_ = append(v_.([]string), v)
		res[k] = v_
	}
	return res
}
