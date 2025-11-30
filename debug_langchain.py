import sys
print(sys.path)
try:
    import langchain
    print(f"Langchain version: {langchain.__version__}")
    print(f"Langchain file: {langchain.__file__}")
except ImportError as e:
    print(f"Error importing langchain: {e}")

try:
    from langchain import chains
    print("Successfully imported langchain.chains")
except ImportError as e:
    print(f"Error importing langchain.chains: {e}")

try:
    import langchain.chains
    print("Successfully imported langchain.chains directly")
except ImportError as e:
    print(f"Error importing langchain.chains directly: {e}")
