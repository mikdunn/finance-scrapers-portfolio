# Installing Python and Packages for Android

This guide provides instructions for running the Finance Scrapers Portfolio toolkit on Android devices.

## Overview

There are several approaches to running Python on Android, each with different trade-offs:

1. **Termux** (Recommended) - Full Linux environment with package manager
2. **Pydroid 3** - User-friendly IDE with pip support
3. **Chaquopy** - For embedding Python in Android apps (advanced)

## Method 1: Termux (Recommended)

Termux provides a full Linux terminal environment on Android with access to Python and most standard packages.

### Installation Steps

1. **Install Termux**
   
   Download from [F-Droid](https://f-droid.org/en/packages/com.termux/) (recommended) or GitHub releases.
   
   ⚠️ **Important**: Do NOT use the Google Play Store version - it's outdated and no longer maintained.

2. **Update Package Lists**
   
   ```bash
   pkg update && pkg upgrade
   ```

3. **Install Python and Dependencies**
   
   ```bash
   pkg install python python-pip git
   ```

4. **Install Build Tools** (required for some Python packages)
   
   ```bash
   pkg install build-essential clang libffi openssl
   ```

5. **Clone the Repository**
   
   ```bash
   git clone https://github.com/mikdunn/finance-scrapers-portfolio.git
   cd finance-scrapers-portfolio
   ```

6. **Install Python Dependencies**
   
   Some packages may need special handling on Android:
   
   ```bash
   # Install NumPy and pandas (core dependencies)
   pip install numpy pandas
   
   # Install the rest from requirements.txt
   pip install -r requirements.txt
   ```

### Troubleshooting for Termux

**Issue: Compilation errors for packages like `numpy` or `pandas`**

Solution: Some packages are available pre-compiled through Termux's package manager:

```bash
pkg install python-numpy python-pandas
pip install -r requirements.txt --no-build-isolation
```

**Issue: `lxml` or `beautifulsoup4` fails to install**

Solution: Install system library first:

```bash
pkg install libxml2 libxslt
pip install lxml beautifulsoup4
```

**Issue: Memory errors during installation**

Solution: Install packages one at a time or increase swap:

```bash
# Create swap file in Termux home directory (more reliable)
cd $HOME
fallocate -l 1G swapfile
chmod 600 swapfile
mkswap swapfile
swapon swapfile

# Alternatively, try /data/local/tmp (may require root on some devices)
# fallocate -l 1G /data/local/tmp/swapfile
```

Note: Swap file creation may not work on all devices. If you get permission errors, just install packages one at a time instead.

**Issue: `selenium` and browser automation**

Selenium requires a web browser driver, which is challenging on Android. Consider:
- Using the API/scraping features that don't require Selenium
- Using Termux-API for notifications instead
- Running browser automation on a desktop/server and using the Android device for viewing results

**Issue: Storage permissions**

Termux has its own isolated storage. To access shared storage:

```bash
termux-setup-storage
```

This creates `~/storage/` symlinks to your device's shared storage.

### Running the Project in Termux

```bash
cd ~/finance-scrapers-portfolio

# Example: Market analyzer
python main.py --project market --symbols "AAPL,MSFT" --period 6mo --interval 1d --out-dir outputs

# View generated HTML files
termux-open outputs/AAPL_6mo_1d.html
```

## Method 2: Pydroid 3

Pydroid 3 is a user-friendly Python IDE for Android with built-in pip support.

### Installation Steps

1. **Install Pydroid 3**
   
   Download from [Google Play Store](https://play.google.com/store/apps/details?id=ru.iiec.pydroid3)
   
   Note: Unlike Termux, Pydroid 3 on Play Store is actively maintained. However, it's a commercial app with some features requiring payment.

2. **Install Pip Packages**
   
   Open Pydroid 3, go to Menu → Pip → Install
   
   Install packages one at a time:
   - requests
   - beautifulsoup4
   - pandas
   - plotly
   - yfinance
   - scikit-learn
   - statsmodels
   - joblib
   - networkx

3. **Download Project Files**
   
   Use a file manager app or download the repository as a ZIP file and extract it.

4. **Open and Run**
   
   - Navigate to the project folder in Pydroid 3
   - Open `main.py`
   - Run the script

### Limitations of Pydroid 3

- Some packages may not be available or may fail to install
- Limited access to system commands
- Selenium/webdriver-manager won't work
- Performance may be slower than Termux

## Method 3: Chaquopy (For App Development)

Chaquopy allows embedding Python in Android apps. This is only relevant if you want to build a standalone Android application.

See [Chaquopy documentation](https://chaquo.com/chaquopy/) for details.

## Package-Specific Notes

### Core Dependencies Status on Android

| Package | Termux | Pydroid 3 | Notes |
|---------|--------|-----------|-------|
| requests | ✅ | ✅ | Works fine |
| beautifulsoup4 | ✅ | ✅ | Works fine |
| selenium | ⚠️ | ❌ | Requires webdriver; limited support |
| pandas | ✅ | ✅ | May need pre-compiled version |
| plotly | ✅ | ✅ | Works fine |
| yfinance | ✅ | ✅ | Works fine |
| statsmodels | ✅ | ⚠️ | May take time to install |
| scikit-learn | ✅ | ⚠️ | Pre-compile in Termux recommended |
| joblib | ✅ | ✅ | Works fine |
| networkx | ✅ | ✅ | Works fine |
| tensorly | ✅ | ⚠️ | Requires NumPy |

### Selenium and WebDriver

Browser automation (Selenium) is challenging on Android:

**Workarounds:**
1. Use projects that don't require `--project collector` or `--project sentiment_heatmap` with browser modes
2. Use RSS mode for sentiment: `--source rss` (doesn't require Selenium)
3. Run browser automation on a remote server/desktop and use Android for viewing results

### Large Dependencies

Some packages like `scikit-learn` and `statsmodels` are large. On devices with limited storage:

```bash
# Check available space
df -h

# Install only essential packages
pip install requests beautifulsoup4 pandas plotly yfinance
```

## Performance Considerations

### CPU and Memory

- Modern Android devices can run these scripts, but expect slower performance than desktop
- Use shorter periods and fewer symbols to reduce memory usage:
  ```bash
  python main.py --project market --symbols "AAPL" --period 1mo --interval 1d --out-dir outputs
  ```

### Battery Usage

- Long-running analyses will drain battery
- Consider running while charging
- Use Termux wake locks if needed:
  ```bash
  termux-wake-lock
  # Your long-running command here
  termux-wake-unlock
  ```

### Storage

- HTML chart files and CSV outputs can accumulate
- Regularly clean old outputs:
  ```bash
  rm -rf outputs/* ml_outputs/* hub_outputs/*
  ```

## Recommended Workflow for Android

1. **Data Collection** (on desktop/server)
   
   Run resource-intensive operations on more powerful hardware:
   ```bash
   python main.py --project hub --universe sp500 --max-symbols 100 --period 1y --interval 1d --out-dir hub_outputs
   ```

2. **Transfer Artifacts** (to Android)
   
   Copy generated CSV and HTML files to your Android device

3. **Analysis and Viewing** (on Android)
   
   - View interactive HTML charts
   - Run targeted analysis on specific symbols
   - Train small ML models on individual assets

## Testing Your Installation

Run this minimal test to verify everything works:

```bash
python -c "import pandas, numpy, plotly; print('All core packages imported successfully!')"
```

Test scikit-learn separately (the package name is `scikit-learn` but imports as `sklearn`):

```bash
python -c "import sklearn; print('scikit-learn imported successfully!')"
```

Run a quick market analysis:

```bash
python main.py --project market --symbols "AAPL" --period 1mo --interval 1d --out-dir test_output
```

If the command completes without errors and generates files in `test_output/`, your installation is working correctly.

## Getting Help

If you encounter issues:

1. Check the [Termux Wiki](https://wiki.termux.com/) for Termux-specific questions
2. Check the [project README](README.md) for general usage
3. Open an issue on [GitHub](https://github.com/mikdunn/finance-scrapers-portfolio/issues)

## Further Reading

- [Termux Documentation](https://wiki.termux.com/)
- [Python on Android: A Brief History](https://wiki.python.org/moin/Android)
- [Pydroid 3 Documentation](https://pydroid.com/)
- [Chaquopy Documentation](https://chaquo.com/chaquopy/)
