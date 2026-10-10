#!/bin/bash
# Security check script for Research Vault
# Run before commits to catch potential security issues

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0

echo "=========================================="
echo "Research Vault Security Check"
echo "=========================================="
echo ""

# Get the repo root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1. Check for real API keys (not placeholders)
echo "Checking for hardcoded API keys..."
if grep -rn --include="*.py" --include="*.ts" --include="*.tsx" --include="*.js" --include="*.json" --include="*.md" --include="*.yml" --include="*.yaml" \
    -E "sk-ant-api3-[a-zA-Z0-9]{20,}|sk-proj-[a-zA-Z0-9]{20,}" \
    --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ \
    . 2>/dev/null; then
    echo -e "${RED}ERROR: Found potential real API keys${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}OK: No real API keys found${NC}"
fi
echo ""

# 2. Check for personal paths (outside build artifacts)
echo "Checking for personal paths..."
if grep -rn --include="*.py" --include="*.ts" --include="*.tsx" --include="*.js" --include="*.json" --include="*.yml" --include="*.yaml" \
    -E "/Users/[a-zA-Z]+/|/home/[a-zA-Z]+/" \
    --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ --exclude-dir=.next --exclude-dir=.claude \
    . 2>/dev/null; then
    echo -e "${RED}ERROR: Found personal paths${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}OK: No personal paths found${NC}"
fi
echo ""

# 3. Check for .env files tracked by git
echo "Checking for tracked .env files..."
if git ls-files | grep -E "^\.env$|/\.env$" 2>/dev/null; then
    echo -e "${RED}ERROR: .env files are tracked by git${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}OK: No .env files tracked${NC}"
fi
echo ""

# 4. Check for SRAL references in code (migration artifact)
echo "Checking for SRAL references in code..."
SRAL_FOUND=$(grep -rn --include="*.py" --include="*.ts" --include="*.tsx" \
    -E "sral|SRAL|state_content|reason_content|act_content|learn_content" \
    --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ \
    . 2>/dev/null | grep -v "# SRAL" | grep -v "devlog" | grep -v ".md:" || true)

if [ -n "$SRAL_FOUND" ]; then
    echo "$SRAL_FOUND"
    echo -e "${YELLOW}WARNING: Found SRAL references (may be migration artifacts)${NC}"
    # Not counting as error, just warning
else
    echo -e "${GREEN}OK: No SRAL references in code${NC}"
fi
echo ""

# 5. Check for debug/development artifacts
echo "Checking for debug artifacts..."
if grep -rn --include="*.py" --include="*.ts" --include="*.tsx" \
    -E "console\.log\(.*password|console\.log\(.*secret|console\.log\(.*key|print\(.*password|print\(.*secret|print\(.*api_key" \
    --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ \
    . 2>/dev/null; then
    echo -e "${RED}ERROR: Found potential secret logging${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}OK: No secret logging found${NC}"
fi
echo ""

# 6. Check that .env.example exists and doesn't have real keys
echo "Checking .env.example files..."
for envfile in .env.example backend/.env.example; do
    if [ -f "$envfile" ]; then
        if grep -E "sk-ant-api3-[a-zA-Z0-9]{20,}|sk-proj-[a-zA-Z0-9]{20,}" "$envfile" 2>/dev/null; then
            echo -e "${RED}ERROR: $envfile contains real API keys${NC}"
            ERRORS=$((ERRORS + 1))
        else
            echo -e "${GREEN}OK: $envfile has placeholder keys${NC}"
        fi
    else
        echo -e "${YELLOW}WARNING: $envfile not found${NC}"
    fi
done
echo ""

# Summary
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All security checks passed!${NC}"
    exit 0
else
    echo -e "${RED}Found $ERRORS security issue(s)${NC}"
    exit 1
fi
