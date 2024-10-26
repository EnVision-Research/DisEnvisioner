 # copied from https://github.com/daohu527/imap/blob/main/imap/global_var.py

def _init():
  global _global_var_dist
  _global_var_dist = {}

def set_value(key, value):
  _global_var_dist[key] = value

def get_value(key):
  return _global_var_dist.get(key)

def is_set(key):
  return key in _global_var_dist
