import numpy as np
import tritonserver

model_repository = "model_repo"
model_name = "identity"
backend_dir = "backends"

server = tritonserver.Server(
    model_repository=model_repository,
    backend_directory=backend_dir,
    log_verbose=6,
    model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
    startup_models=[model_name],
)
server.start(wait_until_ready=True)
print("Server started")

model = server.model(model_name)
print("Model loaded")


input_data = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
inference_responses = model.infer(inputs={"INPUT0": input_data})

for inference_response in inference_responses:
    output_data = np.from_dlpack(inference_response.outputs["OUTPUT0"])
    print(output_data)

server.unload(model_name)
print("Model unloaded")

server.stop()
print("Server stopped")
