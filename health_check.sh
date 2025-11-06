#!/bin/bash

# Kaelum Frontend Health Check Script
echo "🏥 Kaelum Health Check"
echo "===================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check counters
PASSED=0
FAILED=0

# Helper function
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $1"
        ((FAILED++))
    fi
}

# 1. Check directory structure
echo "📁 Checking directory structure..."
[ -d "backend" ] && check_status "backend/ exists" || check_status "backend/ exists"
[ -d "frontend" ] && check_status "frontend/ exists" || check_status "frontend/ exists"
[ -d "core" ] && check_status "core/ exists" || check_status "core/ exists"
[ -d "runtime" ] && check_status "runtime/ exists" || check_status "runtime/ exists"

# 2. Check required files
echo ""
echo "📄 Checking required files..."
[ -f "backend/app.py" ] && check_status "backend/app.py exists" || check_status "backend/app.py exists"
[ -f "frontend/package.json" ] && check_status "frontend/package.json exists" || check_status "frontend/package.json exists"
[ -f "kaelum.py" ] && check_status "kaelum.py exists" || check_status "kaelum.py exists"
[ -f "start_demo.sh" ] && check_status "start_demo.sh exists" || check_status "start_demo.sh exists"

# 3. Check new frontend components
echo ""
echo "🎨 Checking new frontend components..."
[ -f "frontend/app/components/LogViewer.tsx" ] && check_status "LogViewer.tsx exists" || check_status "LogViewer.tsx exists"
[ -f "frontend/app/components/ConfigPanel.tsx" ] && check_status "ConfigPanel.tsx exists" || check_status "ConfigPanel.tsx exists"
[ -f "frontend/app/components/FineTuningPanel.tsx" ] && check_status "FineTuningPanel.tsx exists" || check_status "FineTuningPanel.tsx exists"

# 4. Check Python dependencies
echo ""
echo "🐍 Checking Python environment..."
if command -v python3 &> /dev/null; then
    check_status "Python 3 installed"
    
    # Check key packages
    python3 -c "import flask" 2>/dev/null && check_status "Flask installed" || check_status "Flask installed"
    python3 -c "import flask_cors" 2>/dev/null && check_status "Flask-CORS installed" || check_status "Flask-CORS installed"
    python3 -c "import sentence_transformers" 2>/dev/null && check_status "sentence-transformers installed" || check_status "sentence-transformers installed"
else
    check_status "Python 3 installed"
fi

# 5. Check Node.js and npm
echo ""
echo "📦 Checking Node.js environment..."
if command -v node &> /dev/null; then
    check_status "Node.js installed ($(node --version))"
else
    check_status "Node.js installed"
fi

if command -v npm &> /dev/null; then
    check_status "npm installed ($(npm --version))"
else
    check_status "npm installed"
fi

# 6. Check if frontend dependencies are installed
echo ""
echo "🔧 Checking frontend dependencies..."
if [ -d "frontend/node_modules" ]; then
    check_status "node_modules exists"
else
    echo -e "${YELLOW}!${NC} node_modules not found - run 'cd frontend && npm install'"
fi

# 7. Check .kaelum directory
echo ""
echo "💾 Checking data directories..."
if [ -d ".kaelum" ]; then
    check_status ".kaelum/ exists"
    [ -d ".kaelum/routing" ] && check_status ".kaelum/routing/ exists" || check_status ".kaelum/routing/ exists"
    [ -d ".kaelum/cache" ] && check_status ".kaelum/cache/ exists" || check_status ".kaelum/cache/ exists"
else
    echo -e "${YELLOW}!${NC} .kaelum/ not found - will be created on first run"
fi

# 8. Check if services are running
echo ""
echo "🚀 Checking running services..."
if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    check_status "Backend API (port 5000) is running"
else
    echo -e "${YELLOW}!${NC} Backend not running - start with './start_demo.sh' or 'cd backend && python app.py'"
fi

if curl -s http://localhost:3000 > /dev/null 2>&1; then
    check_status "Frontend (port 3000) is running"
else
    echo -e "${YELLOW}!${NC} Frontend not running - start with './start_demo.sh' or 'cd frontend && npm run dev'"
fi

# Summary
echo ""
echo "===================="
echo "📊 Health Check Summary"
echo "===================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✨ All checks passed! System is healthy.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Start the system: ./start_demo.sh"
    echo "2. Open browser: http://localhost:3000"
    echo "3. Try a query and explore the dashboard"
else
    echo -e "${YELLOW}⚠️  Some checks failed. Please review errors above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "1. Install dependencies: pip install -r requirements.txt"
    echo "2. Install frontend: cd frontend && npm install"
    echo "3. Create data dirs: mkdir -p .kaelum/{routing,cache,cache_validation,calibration}"
fi

echo ""
