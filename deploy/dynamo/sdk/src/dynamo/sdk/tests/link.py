import sys
from dynamo.sdk.tests.pipeline import Frontend, Middle, Backend

print("INITIAL DEPENDENCIES")
print("Frontend dependencies", Frontend.dependencies)
print("Middle dependencies", Middle.dependencies)
print("Backend dependencies", Backend.dependencies)

# print("--------------------------------")

pipeline = Frontend.link(Backend).link(Middle)
pipeline.apply()

print("FINAL DEPENDENCIES")
print("Frontend dependencies", Frontend.dependencies)
print("Middle dependencies", Middle.dependencies)
print("Backend dependencies", Backend.dependencies)


# dynamo serve pipeline:Frontend --kv-mode="random"
