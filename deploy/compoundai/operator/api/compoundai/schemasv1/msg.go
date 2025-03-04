package schemasv1

type MsgSchema struct {
	Message string `json:"message"`
}

type WsMsgSchema struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}
