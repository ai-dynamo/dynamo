#!/bin/bash
#
# Runs inside the SHARED environment image (environment/Dockerfile) — canonical TB2 has no
# separate verifier image. pytest is baked into environment/Dockerfile, so do NOT install or
# download anything here — verify-time setup is rejected by the static checks.
#
# Put your pytest files (e.g. test_outputs.py) in tests/ and run them below. Harbor overlays
# tests/ at /tests only at verify time, so keep ground truth / expected outputs in tests/
# (never in environment/, where the agent could read them).
# --ctrf writes a standard JSON report; write 1/0 to /logs/verifier/reward.txt.
#!/bin/bash
#
# 1. Create the test data
echo "date,product,quantity,price" > /workspace/sales.csv
echo "2024-01-01,Widget,10,5.0" >> /workspace/sales.csv
echo "2024-01-15,Gadget,5,20.0" >> /workspace/sales.csv
echo "2024-02-01,Widget,8,5.0" >> /workspace/sales.csv

# 2. Run the agent's solution
bash solution/solve.sh > /tmp/agent_output.json

# 3. Run the verifier tests on the output
cat /tmp/agent_output.json | python3 /tests/test_outputs.py
RESULT=$?

# 4. Write reward for harbor
if [ $RESULT -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

exit $RESULT
