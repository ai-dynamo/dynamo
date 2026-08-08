import json, sys
output = json.load(sys.stdin)
assert "total_revenue" in output
assert "top_product" in output
assert "monthly_breakdown" in output
assert output["total_revenue"] > 0
print("All tests passed")
