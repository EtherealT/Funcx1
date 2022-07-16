# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from funcx.sdk.client import FuncXClient

fxc = FuncXClient()

def add_func(a, b):
  return a + b

func_uuid = fxc.register_function(add_func)
tutorial_endpoint = '942cf606-bf1c-41bd-a68e-9e35eb2b588e' # Public tutorial endpoint
res = fxc.run(5, 10, function_id=func_uuid, endpoint_id=tutorial_endpoint)
print(fxc.get_result(res))
