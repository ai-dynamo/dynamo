import sys
from dynamo.sdk.tests.pipeline import Frontend, Middle, Backend
from dynamo.sdk.lib.dependency import DynamoDependency, create_pipeline

print("INITIAL DEPENDENCIES")
print("Frontend dependencies", Frontend.dependencies)
print("Middle dependencies", Middle.dependencies)
print("Backend dependencies", Backend.dependencies)

print("--------------------------------")

# Check command line arguments for pipeline configuration
if "--reverse" in sys.argv:
    # Create a reversed pipeline (Frontend -> Backend -> Middle)
    print("Creating REVERSED pipeline: Frontend -> Backend -> Middle")
    sys.argv.remove("--reverse")

    pipeline = Frontend.link(Backend).link(Middle)
    pipeline.apply()

    pipeline.run() # bento apis 

    # this script is run via python3

    print("Pipeline Dependencies:")
    print("Frontend dependencies:", Frontend.dependencies)
    print("Middle dependencies:", Middle.dependencies)
    print("Backend dependencies:", Backend.dependencies)