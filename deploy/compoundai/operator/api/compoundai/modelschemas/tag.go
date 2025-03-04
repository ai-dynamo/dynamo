package modelschemas

import (
	"github.com/huandu/xstrings"
	"github.com/pkg/errors"
)

type Tag string

func (t Tag) Parse() (name, version string, err error) {
	name, _, version = xstrings.Partition(string(t), ":")
	if version == "" {
		err = errors.Errorf("tag %s is invalid", t)
		return
	}
	return
}
