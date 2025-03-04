package schemasv1

type WsReqType string

const (
	WsReqTypeData      WsReqType = "data"
	WsReqTypeHeartbeat WsReqType = "heartbeat"
)

type WsReqSchema struct {
	Type    WsReqType   `json:"type"`
	Payload interface{} `json:"payload"`
}

type WsRespType string

const (
	WsRespTypeSuccess WsRespType = "success"
	WsRespTypeError   WsRespType = "error"
)

type WsRespSchema struct {
	Type    WsRespType  `json:"type"`
	Message string      `json:"message"`
	Payload interface{} `json:"payload"`
}
