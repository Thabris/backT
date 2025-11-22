# Streamlit Launch Troubleshooting

## Quick Fix

The paths have been fixed! Now you need to ensure Streamlit is installed.

### Step 1: Install Streamlit (if not already installed)

**Option A: Run the setup script**
```
Double-click: SETUP_STREAMLIT.bat
```

**Option B: Manual installation**
```bash
conda activate backt
pip install streamlit
```

### Step 2: Launch the app
```
Double-click: launch_streamlit.bat
```

---

## What Was Fixed

✅ **Path issue resolved** - All Streamlit files now correctly point to parent directory
- Changed: `sys.path.insert(0, str(Path(__file__).parent))`
- To: `sys.path.insert(0, str(Path(__file__).parent.parent))`

✅ **Launcher improved** - Better error handling and conda activation

---

## Common Issues & Solutions

### Issue 1: "Could not activate backt environment"

**Solution:**
```bash
# Check if environment exists
conda env list

# If backt doesn't exist, create it:
conda create -n backt python=3.10
conda activate backt
cd C:\Users\maxim\Documents\Projects\backtester2
pip install -e .
pip install streamlit
```

### Issue 2: "Streamlit not found"

**Solution:**
```bash
conda activate backt
pip install streamlit
```

### Issue 3: "No module named 'backt'"

**Solution:**
```bash
conda activate backt
cd C:\Users\maxim\Documents\Projects\backtester2
pip install -e .
```

### Issue 4: Port already in use

**Error:** `Address already in use`

**Solution:**
```bash
# Kill existing Streamlit process
taskkill /F /IM streamlit.exe

# Or use a different port
streamlit run streamlit_backtest_runner.py --server.port 8502
```

### Issue 5: Browser doesn't open automatically

**Solution:**
- Manually open: `http://localhost:8501`
- Or check the terminal output for the URL

---

## Manual Launch (If Batch File Doesn't Work)

```bash
# Open Anaconda Prompt
conda activate backt
cd C:\Users\maxim\Documents\Projects\backtester2\streamlit_apps
streamlit run streamlit_backtest_runner.py
```

---

## Verify Installation

Test that everything is installed correctly:

```bash
conda activate backt
python -c "import backt; import streamlit; print('SUCCESS: All modules found')"
```

Expected output:
```
SUCCESS: All modules found
```

---

## Full Setup from Scratch

If nothing works, start fresh:

```bash
# 1. Create environment
conda create -n backt python=3.10 -y
conda activate backt

# 2. Install BackT
cd C:\Users\maxim\Documents\Projects\backtester2
pip install -e .

# 3. Install Streamlit
pip install streamlit

# 4. Test
python -c "import backt; import streamlit; print('SUCCESS')"

# 5. Launch
cd streamlit_apps
streamlit run streamlit_backtest_runner.py
```

---

## Check What's Installed

```bash
conda activate backt
pip list | findstr streamlit
pip list | findstr backt
```

Should show:
```
backt                0.1.0
streamlit            1.x.x
```

---

## Alternative: Use Python Directly

If the batch file is problematic, create a Python launcher:

**`launch.py`:**
```python
import subprocess
import sys

# Activate conda and run streamlit
conda_path = r"C:\Users\maxim\.conda\condabin\conda.bat"
subprocess.run([
    conda_path, "activate", "backt", "&&",
    "streamlit", "run", "streamlit_apps/streamlit_backtest_runner.py"
], shell=True)
```

Then run:
```bash
python launch.py
```

---

## Files That Were Fixed

All these files now have the correct path:
- ✅ `streamlit_apps/streamlit_backtest_runner.py`
- ✅ `streamlit_apps/streamlit_book_manager.py`
- ✅ `streamlit_apps/streamlit_app.py`
- ✅ `streamlit_backtest_runner_cpcv.py` (no path needed)

---

## Test Each Component

**Test 1: Can Python find backt?**
```bash
cd streamlit_apps
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd().parent)); import backt; print('OK')"
```

**Test 2: Can Streamlit run?**
```bash
conda activate backt
streamlit --version
```

**Test 3: Can the app import everything?**
```bash
cd streamlit_apps
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd().parent)); from backt.utils import BookEditor; print('OK')"
```

---

## Summary

**The issue was:** Files moved to subfolder but still looking for backt in current directory

**The fix was:** Changed path from `.parent` to `.parent.parent`

**Next steps:**
1. Run `SETUP_STREAMLIT.bat` (if Streamlit not installed)
2. Run `launch_streamlit.bat`
3. Browser opens to `http://localhost:8501`

---

## Still Not Working?

**Share this information:**
1. Output of: `conda activate backt && python -c "import sys; print(sys.executable)"`
2. Output of: `conda activate backt && pip list | findstr streamlit`
3. Error message from `launch_streamlit.bat`

Then we can diagnose further!
