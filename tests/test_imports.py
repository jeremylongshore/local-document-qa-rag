import importlib.util, pathlib

def test_app_imports():
    app_path = pathlib.Path("app.py")
    spec = importlib.util.spec_from_file_location("app", app_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # should not raise on import
