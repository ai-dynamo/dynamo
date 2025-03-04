package modelschemas

type LabelItemSchema struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

func (in *LabelItemSchema) DeepCopy() (out *LabelItemSchema) {
	if in == nil {
		return nil
	}

	out = new(LabelItemSchema)
	in.DeepCopyInto(out)
	return
}

func (in *LabelItemSchema) DeepCopyInto(out *LabelItemSchema) {
	*out = *in
}

type LabelItemsSchema []LabelItemSchema
