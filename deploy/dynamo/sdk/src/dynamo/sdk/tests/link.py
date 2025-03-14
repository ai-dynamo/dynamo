# linking syntax example

import sys
from dynamo.sdk.tests.pipeline import Frontend, Middle, Backend, Backend2, End

# print("INITIAL DEPENDENCIES")
# print("Frontend dependencies", Frontend.dependencies)
# print("Middle dependencies", Middle.dependencies)
# print("Backend dependencies", Backend.dependencies)

# print("\n\n\n")

print(Middle.dependencies)
print()
Frontend.link(Middle).link(Backend).build()

Middle.link(Backend2).build()
print()
print(Middle.dependencies)
print(Backend.dependencies)