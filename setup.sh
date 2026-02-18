cat > setup.sh << 'EOF'
#!/usr/bin/env bash
set -e

echo "=============================="
echo " Setting up Python environment"
echo "=============================="

# 1. check python3
if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found"
  exit 1
fi

echo "[INFO] Using python: $(python3 --version)"

# 2. create fresh venv (always)
echo "[INFO] Creating fresh virtual environment..."
rm -rf venv
python3 -m venv venv

# 3. activate venv
if [ ! -f venv/bin/activate ]; then
  echo "[ERROR] venv/bin/activate not found"
  exit 1
fi

source venv/bin/activate

# 4. upgrade pip
echo "[INFO] Upgrading pip..."
python -m pip install --upgrade pip

# 5. install requirements
REQ="requirements_exact.txt"
if [ ! -f "$REQ" ]; then
  echo "[ERROR] $REQ not found"
  exit 1
fi

echo "[INFO] Installing dependencies..."
pip install -r "$REQ"

echo "=============================="
echo " Environment setup complete ✅"
echo "=============================="
echo ""
echo "Next:"
echo "  source venv/bin/activate"
echo "  python main.py"
EOF